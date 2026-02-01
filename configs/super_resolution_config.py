from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class SuperResolutionConfig(BaseConfig):
    """Configuration for Super Resolution model."""
    # Model architecture
    model_path: str = "weights/super_resolution.pth"
    num_blocks: int = 6
    in_channels: int = 3
    growth_channels: int = 64
    scale_factor: int = 8
    
    # Training dimensions - the size the model was trained on
    training_width: int = 38
    training_height: int = 13
    
    # Size threshold - apply SR only to small images
    min_width: int = 50   # If image width < this, apply SR
    min_height: int = 20  # If image height < this, apply SR
    apply_threshold: bool = True
    
    # Preprocessing
    resize_to_training_size: bool = True  # Resize input to training dimensions before SR
    
    # Inference settings
    patch_size: int = 8
    stride_roi: int = 4
    stride_bg: int = 8
    roi_margin: int = 2
    detect_scale: int = 4
    
    # Training settings
    batch_size: int = 128
    num_epochs: int = 600
    learning_rate: float = 1e-4
    alpha: float = 0.5
    checkpoint_dir: str = "checkpoints/super_resolution"
    
    output_subdir: str = "04_super_resolution"