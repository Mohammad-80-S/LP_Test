from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class CarDetectionConfig(BaseConfig):
    """Configuration for YOLOv8 car detection."""
    enabled: bool = True
    model_path: str = "yolov8s.pt"
    confidence_threshold: float = 0.5
    target_class_id: int = 2  # Car class in COCO dataset
    save_cropped_cars: bool = True
    output_subdir: str = "01_car_detection"
    
    # Image extensions to process
    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")