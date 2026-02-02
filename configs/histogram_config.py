from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class HistogramConfig(BaseConfig):
    """Configuration for histogram equalization (CLAHE)."""
    enabled: bool = True
    clip_limit: float = 2.0
    tile_grid_size: tuple = (8, 8)
    output_subdir: str = "03_histogram_equalization"