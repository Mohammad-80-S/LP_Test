from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class PlateDetectionConfig(BaseConfig):
    """Configuration for WPOD-NET plate detection."""
    enabled: bool = True
    model_path: str = "weights/wpodnet.pth"
    scale: float = 1
    dim_min: int = 512
    dim_max: int = 768
    confidence_threshold: float = 0.9
    
    # Output size options
    use_fixed_size: bool = False  # If False, use variable size based on detected corners
    fixed_width: int = 240        # Only used if use_fixed_size=True
    fixed_height: int = 80        # Only used if use_fixed_size=True
    
    # For variable size output
    min_plate_width: int = 20     # Minimum output width
    min_plate_height: int = 10    # Minimum output height
    max_plate_width: int = 400    # Maximum output width
    max_plate_height: int = 150   # Maximum output height
    
    output_subdir: str = "02_plate_detection"