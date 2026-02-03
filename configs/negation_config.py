from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class NegationConfig(BaseConfig):
    """Configuration for histogram-based negation."""
    enabled: bool = True
    scale_factor: int = 8  # Factor to upscale for histogram analysis
    gaussian_window_size: int = 15  # Window size for Gaussian smoothing
    gaussian_std_dev: float = 3.0  # Standard deviation for Gaussian smoothing
    peak_distance: int = 40  # Minimum distance between peaks
    save_visualization: bool = True  # Save histogram visualization for debugging
    debug: bool = False
    output_subdir: str = "03b_negation"