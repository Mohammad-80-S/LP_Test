from typing import Union, Tuple, List
from pathlib import Path
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

from configs import SuperResolutionConfig
from utils.logger import get_logger
from .model import SuperResolutionModel


class SuperResolutionInference:
    """Super Resolution inference module."""
    
    def __init__(self, config: SuperResolutionConfig):
        self.config = config
        self.logger = get_logger()
        self.model = None
        self.device = None
        self._load_model()
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
    
    def _load_model(self):
        """Load super resolution model."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.device == "cuda" else "cpu"
        )
        
        try:
            self.model = SuperResolutionModel(
                num_blocks=self.config.num_blocks,
                in_channels=self.config.in_channels,
                growth_channels=self.config.growth_channels,
                scale_factor=self.config.scale_factor
            ).to(self.device)
            
            checkpoint = torch.load(self.config.model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            self.logger.debug(f"Loaded SR model: {self.config.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load SR model: {e}")
            raise
    
    def should_apply(self, image: Union[Image.Image, np.ndarray]) -> bool:
        """Check if SR should be applied based on image size."""
        if not self.config.apply_threshold:
            return True
        
        if isinstance(image, Image.Image):
            w, h = image.size
        else:
            h, w = image.shape[:2]
        
        # Apply SR if image is smaller than threshold
        return w < self.config.min_width or h < self.config.min_height
    
    def _resize_to_training_size(self, image: Image.Image) -> Image.Image:
        """
        Resize image to the dimensions the model was trained on.
        
        Args:
            image: Input image
            
        Returns:
            Resized image with dimensions (training_width, training_height)
        """
        target_w = self.config.training_width
        target_h = self.config.training_height
        
        # Use BICUBIC for upscaling small images
        resized = image.resize((target_w, target_h), Image.BICUBIC)
        
        self.logger.debug(f"Resized image from {image.size} to {resized.size} (training size)")
        return resized
    
    def enhance(
        self, 
        image: Union[Image.Image, np.ndarray, str, Path]
    ) -> Image.Image:
        """
        Apply super resolution to image.
        
        The image is first resized to the training dimensions (38x13) if needed,
        then the SR model is applied to upscale it.
        
        Args:
            image: Input low-resolution image
            
        Returns:
            Enhanced high-resolution image
        """
        # Load image
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image.copy()
        
        original_size = pil_img.size
        
        # Check if SR should be applied
        if self.config.apply_threshold and not self.should_apply(pil_img):
            self.logger.debug(f"Image size {original_size} above threshold, skipping SR")
            return pil_img
        
        # Resize to training dimensions
        if self.config.resize_to_training_size:
            pil_img = self._resize_to_training_size(pil_img)
        
        # Convert to tensor
        lr_tensor = self.to_tensor(pil_img).unsqueeze(0).to(self.device)
        
        # Apply SR model directly (no patching needed for small 38x13 images)
        with torch.no_grad():
            sr_tensor = self.model(lr_tensor)
        
        # Convert back to PIL
        sr_tensor = sr_tensor.squeeze(0).clamp(0, 1).cpu()
        sr_img = self.to_pil(sr_tensor)
        
        scale = self.config.scale_factor
        expected_w = self.config.training_width * scale
        expected_h = self.config.training_height * scale
        
        self.logger.debug(
            f"Applied SR: {original_size} -> {self.config.training_width}x{self.config.training_height} "
            f"-> {sr_img.size[0]}x{sr_img.size[1]}"
        )
        
        return sr_img
    
    def enhance_simple(
        self,
        image: Union[Image.Image, np.ndarray, str, Path]
    ) -> Image.Image:
        """
        Simple enhancement without patching - for small plate images.
        
        This method:
        1. Resizes input to training size (38x13)
        2. Applies SR model
        3. Returns upscaled image (304x104 for 8x scale)
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Load image
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image.copy()
        
        original_size = pil_img.size
        
        # Resize to exact training dimensions
        target_w = self.config.training_width
        target_h = self.config.training_height
        resized_img = pil_img.resize((target_w, target_h), Image.BICUBIC)
        
        # Convert to tensor and add batch dimension
        lr_tensor = self.to_tensor(resized_img).unsqueeze(0).to(self.device)
        
        # Apply model
        with torch.no_grad():
            sr_tensor = self.model(lr_tensor)
        
        # Convert back to PIL
        sr_tensor = sr_tensor.squeeze(0).clamp(0, 1).cpu()
        sr_img = self.to_pil(sr_tensor)
        
        self.logger.debug(
            f"SR enhancement: {original_size} -> ({target_w}, {target_h}) -> {sr_img.size}"
        )
        
        return sr_img