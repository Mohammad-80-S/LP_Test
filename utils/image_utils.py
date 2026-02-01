import os
from pathlib import Path
from typing import List, Union, Tuple, Optional
from PIL import Image
import numpy as np


class ImageUtils:
    """Utility functions for image handling."""
    
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    
    @staticmethod
    def is_image_file(path: Union[str, Path]) -> bool:
        """Check if a file is a supported image."""
        return Path(path).suffix.lower() in ImageUtils.SUPPORTED_EXTENSIONS
    
    @staticmethod
    def get_image_paths(input_path: Union[str, Path]) -> List[Path]:
        """Get list of image paths from file or directory."""
        input_path = Path(input_path)
        
        if input_path.is_file():
            if ImageUtils.is_image_file(input_path):
                return [input_path]
            else:
                raise ValueError(f"Not a supported image file: {input_path}")
        
        elif input_path.is_dir():
            image_paths = []
            for ext in ImageUtils.SUPPORTED_EXTENSIONS:
                image_paths.extend(input_path.glob(f"*{ext}"))
                image_paths.extend(input_path.glob(f"*{ext.upper()}"))
            return sorted(image_paths)
        
        else:
            raise FileNotFoundError(f"Path not found: {input_path}")
    
    @staticmethod
    def load_image(path: Union[str, Path]) -> Image.Image:
        """Load image as PIL Image."""
        return Image.open(path).convert("RGB")
    
    @staticmethod
    def save_image(
        image: Union[Image.Image, np.ndarray],
        path: Union[str, Path],
        create_dir: bool = True
    ) -> Path:
        """Save image to path."""
        path = Path(path)
        
        if create_dir:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image.save(path)
        return path
    
    @staticmethod
    def get_image_size(image: Union[Image.Image, np.ndarray]) -> Tuple[int, int]:
        """Get image dimensions (width, height)."""
        if isinstance(image, Image.Image):
            return image.size
        else:
            return image.shape[1], image.shape[0]
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """Ensure directory exists."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path