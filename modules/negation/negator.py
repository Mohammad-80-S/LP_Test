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
    
    Logic:
    - Find two highest peaks in histogram
    - The highest peak represents the background (more pixels)
    - The second highest peak represents the characters
    - If background peak position < character peak position → dark background → INVERT
    - If background peak position > character peak position → light background → NO CHANGE
    """
    
    def __init__(self, config: NegationConfig):
        self.config = config
        self.logger = get_logger()
    
    def _upscale_for_analysis(self, image: np.ndarray) -> np.ndarray:
        """Upscale image using bicubic interpolation for better histogram analysis."""
        h, w = image.shape[:2]
        new_w = w * self.config.upscale_factor
        new_h = h * self.config.upscale_factor
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    def _get_histogram(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get grayscale histogram of the image.
        
        Returns:
            Tuple of (histogram, grayscale_image)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        return hist, gray
    
    def _find_two_peaks(self, hist: np.ndarray) -> Tuple[int, int, float, float]:
        """
        Find the two highest peaks in the histogram.
        
        Returns:
            Tuple of (background_peak_pos, character_peak_pos, background_peak_height, character_peak_height)
            where background peak is the highest (more pixels) and character peak is second highest
        """
        # Smooth histogram to reduce noise
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        hist_smooth = np.convolve(hist, kernel, mode='same')
        
        # Find all peaks
        peaks, properties = find_peaks(
            hist_smooth, 
            distance=self.config.peak_distance, 
            height=0
        )
        
        if len(peaks) < 2:
            # If less than 2 peaks found, use alternative approach
            # Find the global maximum first
            peak1_pos = np.argmax(hist_smooth)
            peak1_height = hist_smooth[peak1_pos]
            
            # Mask the area around the first peak and find second peak
            hist_masked = hist_smooth.copy()
            mask_start = max(0, peak1_pos - self.config.peak_distance)
            mask_end = min(256, peak1_pos + self.config.peak_distance)
            hist_masked[mask_start:mask_end] = 0
            
            peak2_pos = np.argmax(hist_masked)
            peak2_height = hist_smooth[peak2_pos]
            
            # peak1 is background (highest), peak2 is characters (second highest)
            return int(peak1_pos), int(peak2_pos), float(peak1_height), float(peak2_height)
        
        # Get peak heights
        peak_heights = hist_smooth[peaks]
        
        # Sort peaks by height (descending)
        sorted_indices = np.argsort(peak_heights)[::-1]
        
        # Get top 2 peaks
        # Peak 1 (background) is the highest, Peak 2 (characters) is second highest
        background_peak_pos = peaks[sorted_indices[0]]
        character_peak_pos = peaks[sorted_indices[1]]
        background_peak_height = peak_heights[sorted_indices[0]]
        character_peak_height = peak_heights[sorted_indices[1]]
        
        return (
            int(background_peak_pos), 
            int(character_peak_pos), 
            float(background_peak_height), 
            float(character_peak_height)
        )
    
    def _needs_inversion(self, image: np.ndarray) -> Tuple[bool, dict]:
        """
        Determine if image needs inversion based on histogram analysis.
        
        Logic:
        - Find two peaks: background (highest) and characters (second highest)
        - Compare their positions:
          - If background_peak_pos < character_peak_pos → background is darker → INVERT
          - If background_peak_pos > character_peak_pos → background is lighter → NO CHANGE
        
        Returns:
            Tuple of (needs_inversion, peak_info_dict)
        """
        hist, gray = self._get_histogram(image)
        bg_peak_pos, char_peak_pos, bg_peak_height, char_peak_height = self._find_two_peaks(hist)
        
        # Background is darker if its peak is to the left of character peak
        background_is_dark = bg_peak_pos < char_peak_pos
        
        peak_info = {
            'background_peak_position': bg_peak_pos,
            'character_peak_position': char_peak_pos,
            'background_peak_height': bg_peak_height,
            'character_peak_height': char_peak_height,
            'background_is_dark': background_is_dark
        }
        
        # If background peak is on the left of character peak (darker background)
        # then we need to invert
        needs_inversion = background_is_dark
        
        return needs_inversion, peak_info
    
    def _invert_image(self, image: np.ndarray) -> np.ndarray:
        """Invert image colors."""
        return cv2.bitwise_not(image)
    
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
        # Load image and convert to numpy array
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
            img_array = np.array(pil_img)
        elif isinstance(image, Image.Image):
            pil_img = image.copy()
            img_array = np.array(pil_img)
        else:
            img_array = image.copy()
            pil_img = Image.fromarray(img_array)
        
        original_size = img_array.shape[:2]
        
        # Upscale for histogram analysis
        upscaled = self._upscale_for_analysis(img_array)
        
        self.logger.debug(
            f"Upscaled image from {original_size} to {upscaled.shape[:2]} for histogram analysis"
        )
        
        # Determine if inversion is needed (analyze upscaled image)
        needs_inversion, peak_info = self._needs_inversion(upscaled)
        
        self.logger.debug(
            f"Peak analysis - Background peak: {peak_info['background_peak_position']} "
            f"(height: {peak_info['background_peak_height']:.0f}), "
            f"Character peak: {peak_info['character_peak_position']} "
            f"(height: {peak_info['character_peak_height']:.0f})"
        )
        
        # Apply inversion to ORIGINAL image if needed
        if needs_inversion:
            result_array = self._invert_image(img_array)
            self.logger.debug(
                f"Applied inversion: background peak ({peak_info['background_peak_position']}) "
                f"< character peak ({peak_info['character_peak_position']}) → dark background"
            )
        else:
            result_array = img_array
            self.logger.debug(
                f"No inversion needed: background peak ({peak_info['background_peak_position']}) "
                f"> character peak ({peak_info['character_peak_position']}) → light background"
            )
        
        return Image.fromarray(result_array)
    
    def process_batch(
        self, 
        images: List[Union[Image.Image, np.ndarray]]
    ) -> List[Image.Image]:
        """Process multiple images."""
        return [self.process(img) for img in images]
    
    def process_with_info(
        self, 
        image: Union[Image.Image, np.ndarray, str, Path]
    ) -> Tuple[Image.Image, bool, dict]:
        """
        Process image and return additional info.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (processed_image, was_inverted, peak_info)
        """
        # Load image and convert to numpy array
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
            img_array = np.array(pil_img)
        elif isinstance(image, Image.Image):
            pil_img = image.copy()
            img_array = np.array(pil_img)
        else:
            img_array = image.copy()
        
        # Upscale for histogram analysis
        upscaled = self._upscale_for_analysis(img_array)
        
        # Determine if inversion is needed
        needs_inversion, peak_info = self._needs_inversion(upscaled)
        
        # Apply inversion to ORIGINAL image if needed
        if needs_inversion:
            result_array = self._invert_image(img_array)
        else:
            result_array = img_array
        
        return Image.fromarray(result_array), needs_inversion, peak_info