from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class NegationConfig(BaseConfig):
    """Configuration for histogram-based negation."""
    enabled: bool = True
    upscale_factor: int = 8  # Factor to upscale for histogram analysis
    peak_distance: int = 30  # Minimum distance between peaks in histogram
    debug: bool = False
    output_subdir: str = "03b_negation"