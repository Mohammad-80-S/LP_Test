from typing import Optional, Union, Tuple
from pathlib import Path
import torch
from PIL import Image
import numpy as np

from configs import PlateDetectionConfig
from utils.logger import get_logger
from .wpodnet import WPODNet, Predictor, Prediction
from .wpodnet.model import load_wpodnet_from_checkpoint


class PlateDetector:
    """License plate detection using WPOD-NET."""
    
    def __init__(self, config: PlateDetectionConfig):
        self.config = config
        self.logger = get_logger()
        self.model = None
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """Load WPOD-NET model."""
        device = "cuda" if torch.cuda.is_available() and self.config.device == "cuda" else "cpu"
        try:
            self.model = load_wpodnet_from_checkpoint(self.config.model_path).to(device)
            self.predictor = Predictor(self.model)
            self.logger.debug(f"Loaded plate detection model: {self.config.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load plate detection model: {e}")
            raise
    
    def detect(
        self, 
        image: Union[Image.Image, np.ndarray, str, Path]
    ) -> Tuple[Optional[Prediction], Optional[Image.Image]]:
        """
        Detect license plate in image.
        
        Args:
            image: Input image (car crop)
            
        Returns:
            Tuple of (prediction, warped_plate_image)
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image
        
        # Run prediction
        prediction = self.predictor.predict(
            pil_img,
            scaling_ratio=self.config.scale,
            dim_min=self.config.dim_min,
            dim_max=self.config.dim_max,
            confidence_threshold=self.config.confidence_threshold,
        )
        
        if prediction is None:
            self.logger.debug("No plate detected (confidence below threshold)")
            return None, None
        
        self.logger.debug(f"Plate detected with confidence: {prediction.confidence:.4f}")
        
        # Warp the plate region
        if self.config.use_fixed_size:
            # Fixed size output (old behavior)
            warped = prediction.warp(
                pil_img,
                width=self.config.fixed_width,
                height=self.config.fixed_height,
                use_auto_size=False
            )
            self.logger.debug(f"Warped plate to fixed size: {self.config.fixed_width}x{self.config.fixed_height}")
        else:
            # Variable size based on detected corners
            warped = prediction.warp(
                pil_img,
                use_auto_size=True,
                min_width=self.config.min_plate_width,
                min_height=self.config.min_plate_height,
                max_width=self.config.max_plate_width,
                max_height=self.config.max_plate_height
            )
            self.logger.debug(f"Warped plate to auto size: {warped.size[0]}x{warped.size[1]}")
        
        return prediction, warped
    
    def annotate(
        self, 
        image: Image.Image, 
        prediction: Prediction,
        outline: str = "red",
        width: int = 2
    ) -> Image.Image:
        """Draw plate bounds on image."""
        canvas = image.copy()
        prediction.annotate(canvas, outline=outline, width=width)
        return canvas