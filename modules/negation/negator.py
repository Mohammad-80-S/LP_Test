from typing import Union, Tuple, List, Optional
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt

from configs import NegationConfig
from utils.logger import get_logger


class HistogramNegator:
    """
    Histogram-based image negation for license plates.
    
    Analyzes the histogram of an upscaled image to determine if the background
    is darker than the characters. If so, negates the image.
    
    Logic:
    - Find two highest peaks in smoothed histogram
    - The highest peak represents the background (more pixels)
    - The second highest peak represents the characters/text
    - If background peak position < character peak position → dark background → INVERT
    - If background peak position > character peak position → light background → NO CHANGE
    """
    
    def __init__(self, config: NegationConfig):
        self.config = config
        self.logger = get_logger()
        self._visualization_counter = 0
    
    def _upscale_for_analysis(self, image: np.ndarray) -> np.ndarray:
        """Upscale image using bicubic interpolation for better histogram analysis."""
        h, w = image.shape[:2]
        new_w = w * self.config.scale_factor
        new_h = h * self.config.scale_factor
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    def _get_histogram(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get grayscale histogram of the image.
        
        Returns:
            Tuple of (histogram, grayscale_image)
        """
        if len(image.shape) == 3:
            # Convert RGB to grayscale (input is RGB from PIL)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        return hist, gray
    
    def _get_smoothed_histogram(
        self, 
        hist: np.ndarray, 
        window_size: int = None, 
        std_dev: float = None
    ) -> np.ndarray:
        """
        Smooth the histogram using a Weighted Moving Average (Gaussian Window).
        This ensures points farther from the center have less weight.
        
        Args:
            hist: Raw histogram
            window_size: Size of Gaussian window (default from config)
            std_dev: Standard deviation for Gaussian (default from config)
            
        Returns:
            Smoothed histogram
        """
        if window_size is None:
            window_size = self.config.gaussian_window_size
        if std_dev is None:
            std_dev = self.config.gaussian_std_dev
        
        # Create a Gaussian kernel
        kernel = gaussian(window_size, std=std_dev)
        
        # Normalize kernel so the sum is 1
        kernel /= np.sum(kernel)
        
        # Apply the moving average via convolution
        hist_smooth = np.convolve(hist, kernel, mode='same')
        
        return hist_smooth
    
    def _find_peaks_and_analyze(self, image: np.ndarray) -> Tuple[bool, dict]:
        """
        Finds the two highest peaks and determines if inversion is needed.
        
        Args:
            image: Input image (numpy array, RGB format)
            
        Returns:
            Tuple of (should_invert, details_dict)
        """
        hist, gray = self._get_histogram(image)
        
        # Smooth the histogram
        hist_smooth = self._get_smoothed_histogram(hist)
        
        # Find peaks with minimum distance
        peaks, properties = find_peaks(
            hist_smooth, 
            distance=self.config.peak_distance, 
            height=0
        )
        
        # Handle edge case: if less than 2 peaks found
        if len(peaks) < 2:
            max_idx = np.argmax(hist_smooth)
            # Fallback: simple threshold logic if peaks fail
            should_invert = max_idx < 128
            
            self.logger.debug(f"Less than 2 peaks found, using fallback. Max at {max_idx}")
            
            return should_invert, {
                'hist': hist,
                'hist_smooth': hist_smooth,
                'p1_idx': int(max_idx),
                'p2_idx': None,
                'p1_h': float(hist_smooth[max_idx]),
                'p2_h': 0.0,
                'msg': 'Less than 2 peaks found'
            }
        
        # Sort peaks by height (Highest first)
        peak_heights = hist_smooth[peaks]
        sorted_indices = np.argsort(peak_heights)[::-1]  # Descending order
        
        # Get top 2 peaks
        peak1_idx = peaks[sorted_indices[0]]  # Highest Peak (Background)
        peak2_idx = peaks[sorted_indices[1]]  # Second Highest Peak (Text/Details)
        
        peak1_height = peak_heights[sorted_indices[0]]
        peak2_height = peak_heights[sorted_indices[1]]
        
        # Compare Highest Peak (P1) with Second Peak (P2)
        # If P1 (Background) is to the RIGHT of P2 (Text) -> Normal (Light BG)
        # If P1 (Background) is to the LEFT of P2 (Text)  -> Invert (Dark BG)
        should_invert = peak1_idx < peak2_idx
        
        self.logger.debug(
            f"Peak analysis - P1 (Background): {peak1_idx} (h={peak1_height:.0f}), "
            f"P2 (Text): {peak2_idx} (h={peak2_height:.0f}), "
            f"Should invert: {should_invert}"
        )
        
        return should_invert, {
            'hist': hist,
            'hist_smooth': hist_smooth,
            'p1_idx': int(peak1_idx),
            'p1_h': float(peak1_height),
            'p2_idx': int(peak2_idx),
            'p2_h': float(peak2_height),
            'msg': 'Analyzed two peaks'
        }
    
    def _invert_image(self, image: np.ndarray) -> np.ndarray:
        """Invert image colors."""
        return cv2.bitwise_not(image)
    
    def _save_visualization(
        self, 
        original_image: np.ndarray,
        processed_image: np.ndarray,
        was_inverted: bool,
        details: dict,
        save_path: Path
    ) -> None:
        """
        Save visualization of histogram analysis and result.
        
        Args:
            original_image: Original input image (RGB format)
            processed_image: Processed output image (RGB format)
            was_inverted: Whether inversion was applied
            details: Analysis details dictionary
            save_path: Path to save the visualization
        """
        plt.figure(figsize=(15, 5))
        
        # 1. Original Image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title("Original Input")
        plt.axis('off')
        
        # 2. Histogram Analysis
        plt.subplot(1, 3, 2)
        
        # Plot raw histogram lightly
        plt.bar(range(256), details['hist'], color='gray', alpha=0.3, label='Raw Hist')
        
        # Plot Smoothed histogram
        plt.plot(details['hist_smooth'], color='black', linewidth=2, 
                label=f'Smoothed (Window={self.config.gaussian_window_size})')
        
        if details['p2_idx'] is not None:
            # Mark Peak 1 (Highest - Background)
            plt.plot(details['p1_idx'], details['p1_h'], "x", 
                    color="red", markersize=12, markeredgewidth=3)
            plt.text(details['p1_idx'], details['p1_h'], " P1 (Bg)", 
                    color="red", va="bottom", ha="left", fontsize=10)
            
            # Mark Peak 2 (Second Highest - Text)
            plt.plot(details['p2_idx'], details['p2_h'], "x", 
                    color="blue", markersize=12, markeredgewidth=3)
            plt.text(details['p2_idx'], details['p2_h'], " P2 (Txt)", 
                    color="blue", va="bottom", ha="left", fontsize=10)
            
            if details['p1_idx'] > details['p2_idx']:
                status = f"P1({details['p1_idx']}) > P2({details['p2_idx']}) → Light BG"
            else:
                status = f"P1({details['p1_idx']}) < P2({details['p2_idx']}) → Dark BG"
        else:
            status = f"Single Peak at {details['p1_idx']}"
        
        plt.title(f"Analysis: {status}")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.legend(loc='upper right')
        plt.xlim([0, 255])
        
        # 3. Final Result
        plt.subplot(1, 3, 3)
        plt.imshow(processed_image)
        res_title = "INVERTED" if was_inverted else "NO CHANGE"
        col = 'red' if was_inverted else 'green'
        plt.title(f"Result: {res_title}", color=col, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.debug(f"Saved visualization to {save_path}")
    
    def process(
        self, 
        image: Union[Image.Image, np.ndarray, str, Path],
        save_dir: Optional[Path] = None,
        image_name: Optional[str] = None
    ) -> Image.Image:
        """
        Process image and negate if necessary.
        
        The image is upscaled for histogram analysis, but the output
        is the original size image (negated if necessary).
        
        Args:
            image: Input image
            save_dir: Directory to save visualization (optional)
            image_name: Name for the saved visualization file (optional)
            
        Returns:
            Original size image, negated if background was dark
        """
        # Load image and convert to numpy array (RGB format)
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
            img_array = np.array(pil_img)
        elif isinstance(image, Image.Image):
            pil_img = image.copy()
            img_array = np.array(pil_img)
        else:
            img_array = image.copy()
        
        original_size = img_array.shape[:2]  # (height, width)
        
        # Upscale for histogram analysis
        upscaled = self._upscale_for_analysis(img_array)
        
        self.logger.debug(
            f"Upscaled image from {original_size} to {upscaled.shape[:2]} "
            f"(scale={self.config.scale_factor}x) for histogram analysis"
        )
        
        # Analyze the upscaled image
        should_invert, details = self._find_peaks_and_analyze(upscaled)
        
        # Apply inversion to ORIGINAL image if needed
        if should_invert:
            result_array = self._invert_image(img_array)
            self.logger.debug(
                f"Applied inversion: P1({details['p1_idx']}) < P2({details['p2_idx']}) → dark background"
            )
        else:
            result_array = img_array.copy()
            self.logger.debug(
                f"No inversion needed: P1({details['p1_idx']}) > P2({details.get('p2_idx', 'N/A')}) → light background"
            )
        
        # Save visualization if enabled and save_dir provided
        if self.config.save_visualization and save_dir is not None:
            self._visualization_counter += 1
            if image_name is None:
                image_name = f"negation_{self._visualization_counter:04d}"
            
            viz_path = save_dir / f"{image_name}_analysis.png"
            self._save_visualization(
                original_image=img_array,
                processed_image=result_array,
                was_inverted=should_invert,
                details=details,
                save_path=viz_path
            )
        
        return Image.fromarray(result_array)
    
    def process_batch(
        self, 
        images: List[Union[Image.Image, np.ndarray]],
        save_dir: Optional[Path] = None
    ) -> List[Image.Image]:
        """Process multiple images."""
        results = []
        for idx, img in enumerate(images):
            result = self.process(
                img, 
                save_dir=save_dir, 
                image_name=f"plate_{idx:04d}"
            )
            results.append(result)
        return results
    
    def process_with_info(
        self, 
        image: Union[Image.Image, np.ndarray, str, Path],
        save_dir: Optional[Path] = None,
        image_name: Optional[str] = None
    ) -> Tuple[Image.Image, bool, dict]:
        """
        Process image and return additional info.
        
        Args:
            image: Input image
            save_dir: Directory to save visualization (optional)
            image_name: Name for the saved visualization file (optional)
            
        Returns:
            Tuple of (processed_image, was_inverted, details)
        """
        # Load image and convert to numpy array (RGB format)
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
        
        # Analyze the upscaled image
        should_invert, details = self._find_peaks_and_analyze(upscaled)
        
        # Apply inversion to ORIGINAL image if needed
        if should_invert:
            result_array = self._invert_image(img_array)
        else:
            result_array = img_array.copy()
        
        # Save visualization if enabled
        if self.config.save_visualization and save_dir is not None:
            self._visualization_counter += 1
            if image_name is None:
                image_name = f"negation_{self._visualization_counter:04d}"
            
            viz_path = save_dir / f"{image_name}_analysis.png"
            self._save_visualization(
                original_image=img_array,
                processed_image=result_array,
                was_inverted=should_invert,
                details=details,
                save_path=viz_path
            )
        
        return Image.fromarray(result_array), should_invert, details