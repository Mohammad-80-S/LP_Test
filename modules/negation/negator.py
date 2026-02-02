from typing import Union, Tuple, List
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.signal import find_peaks

from configs import NegationConfig
from utils.logger import get_logger


class HistogramNegator:
    """
    Histogram-based image negation.
    
    Analyzes the histogram of an upscaled image to determine if the background
    is darker than the characters. If so, negates the image.
    """
    
    def __init__(self, config: NegationConfig):
        self.config = config
        self.logger = get_logger()
    
    def _upscale_for_analysis(self, image: Image.Image) -> Image.Image:
        """Upscale image using bicubic interpolation for better histogram analysis."""
        w, h = image.size
        new_w = w * self.config.upscale_factor
        new_h = h * self.config.upscale_factor
        return image.resize((new_w, new_h), Image.BICUBIC)
    
    def _compute_grayscale_histogram(self, image: Image.Image) -> np.ndarray:
        """Compute histogram of grayscale image."""
        gray = image.convert('L')
        gray_array = np.array(gray)
        histogram, _ = np.histogram(gray_array.flatten(), bins=256, range=(0, 256))
        return histogram
    
    def _find_two_peaks(self, histogram: np.ndarray) -> Tuple[int, int]:
        """
        Find the two highest peaks in the histogram.
        
        Returns:
            Tuple of (highest_peak_position, second_highest_peak_position)
        """
        # Smooth histogram to reduce noise
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(histogram, kernel, mode='same')
        
        # Find all peaks with minimum distance
        peaks, properties = find_peaks(
            smoothed, 
            distance=self.config.peak_distance,
            height=np.max(smoothed) * 0.05  # At least 5% of max height
        )
        
        if len(peaks) < 2:
            # If less than 2 peaks found, use simple approach
            # Find global maximum
            highest_idx = np.argmax(smoothed)
            
            # Mask around the highest peak and find second
            mask_start = max(0, highest_idx - self.config.peak_distance)
            mask_end = min(256, highest_idx + self.config.peak_distance)
            smoothed_masked = smoothed.copy()
            smoothed_masked[mask_start:mask_end] = 0
            second_idx = np.argmax(smoothed_masked)
            
            return highest_idx, second_idx
        
        # Get peak heights
        peak_heights = smoothed[peaks]
        
        # Sort peaks by height (descending)
        sorted_indices = np.argsort(peak_heights)[::-1]
        
        # Get top 2 peaks
        highest_peak = peaks[sorted_indices[0]]
        second_peak = peaks[sorted_indices[1]]
        
        return highest_peak, second_peak
    
    def _should_negate(self, image: Image.Image) -> bool:
        """
        Determine if image should be negated based on histogram analysis.
        
        If the highest peak (background) is on the left side of the histogram,
        it means the background is dark and the image should be negated.
        
        Returns:
            True if image should be negated, False otherwise
        """
        histogram = self._compute_grayscale_histogram(image)
        highest_peak, second_peak = self._find_two_peaks(histogram)
        
        self.logger.debug(
            f"Histogram peaks - Highest (background): {highest_peak}, "
            f"Second (characters): {second_peak}"
        )
        
        # If highest peak is on the left (dark), negate
        # Background should be brighter than characters for proper OCR
        should_negate = highest_peak < second_peak
        
        if should_negate:
            self.logger.debug("Background is dark, image will be negated")
        else:
            self.logger.debug("Background is light, no negation needed")
        
        return should_negate
    
    def _negate_image(self, image: Image.Image) -> Image.Image:
        """Negate (invert) the image."""
        img_array = np.array(image)
        negated_array = 255 - img_array
        return Image.fromarray(negated_array)
    
    def process(
        self, 
        image: Union[Image.Image, np.ndarray, str, Path]
    ) -> Image.Image:
        """
        Process image and negate if necessary.
        
        The image is upscaled for histogram analysis, but the output
        is the original size image (negated if necessary).
        
        Args:
            image: Input image
            
        Returns:
            Original size image, negated if background was dark
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image.copy()
        
        original_size = pil_img.size
        
        # Upscale for histogram analysis
        upscaled = self._upscale_for_analysis(pil_img)
        
        self.logger.debug(
            f"Upscaled image from {original_size} to {upscaled.size} for histogram analysis"
        )
        
        # Determine if negation is needed
        if self._should_negate(upscaled):
            # Negate the ORIGINAL image, not the upscaled one
            result = self._negate_image(pil_img)
            self.logger.debug("Applied negation to original image")
        else:
            result = pil_img
        
        return result
    
    def process_batch(
        self, 
        images: List[Union[Image.Image, np.ndarray]]
    ) -> List[Image.Image]:
        """Process multiple images."""
        return [self.process(img) for img in images]