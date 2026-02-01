from .base_config import BaseConfig, PipelineConfig
from .car_detection_config import CarDetectionConfig
from .plate_detection_config import PlateDetectionConfig
from .histogram_config import HistogramConfig
from .super_resolution_config import SuperResolutionConfig
from .ocr_config import OCRConfig

__all__ = [
    "BaseConfig",
    "PipelineConfig",
    "CarDetectionConfig",
    "PlateDetectionConfig",
    "HistogramConfig",
    "SuperResolutionConfig",
    "OCRConfig",
]