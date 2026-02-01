from typing import Union
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from configs import HistogramConfig
from utils.logger import get_logger


class HistogramEqualizer:
    """CLAHE-based histogram equalization."""
    
    def __init__(self, config: HistogramConfig):
        self.config = config
        self.logger = get_logger()
        self.clahe = cv2.createCLAHE(
            clipLimit=config.clip_limit,
            tileGridSize=config.tile_grid_size
        )
    
    def equalize(
        self, 
        image: Union[Image.Image, np.ndarray, str, Path]
    ) -> Image.Image:
        """
        Apply CLAHE histogram equalization.
        
        Args:
            image: Input image
            
        Returns:
            Equalized image as PIL Image
        """
        # Convert to numpy array
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        # Check if grayscale or color
        if len(img.shape) == 2:
            # Grayscale
            equalized = self.clahe.apply(img)
        else:
            # Color - use YCrCb color space
            # Convert to BGR for OpenCV processing
            if img.shape[2] == 3:
                bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                bgr_img = img
            
            ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
            y_channel, cr, cb = cv2.split(ycrcb)
            equalized_y = self.clahe.apply(y_channel)
            merged = cv2.merge([equalized_y, cr, cb])
            equalized_bgr = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
            equalized = cv2.cvtColor(equalized_bgr, cv2.COLOR_BGR2RGB)
        
        self.logger.debug("Applied histogram equalization")
        return Image.fromarray(equalized)