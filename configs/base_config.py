from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class BaseConfig:
    """Base configuration class with common settings."""
    debug: bool = False
    device: str = "cuda"  # "cuda" or "cpu"
    verbose: bool = False
    

@dataclass
class PipelineConfig(BaseConfig):
    """Main pipeline configuration."""
    # Debug settings
    debug: bool = False
    debug_output_dir: str = "Test_Debug"
    save_intermediate: bool = True
    
    # Pipeline stages to run
    start_stage: str = "car_detection"  # Options: car_detection, plate_detection, histogram, super_resolution, ocr
    end_stage: str = "ocr"
    
    # Input/Output settings
    input_path: str = ""  # Can be image path or folder path
    output_dir: str = "output"
    
    # Device settings
    device: str = "cuda"
    
    # Stage-specific configs (will be populated)
    car_detection: Optional["CarDetectionConfig"] = None
    plate_detection: Optional["PlateDetectionConfig"] = None
    histogram: Optional["HistogramConfig"] = None
    super_resolution: Optional["SuperResolutionConfig"] = None
    ocr: Optional["OCRConfig"] = None
    
    # Supported stages in order
    STAGES: List[str] = field(default_factory=lambda: [
        "car_detection",
        "plate_detection", 
        "histogram",
        "super_resolution",
        "ocr"
    ])
    
    def __post_init__(self):
        from .car_detection_config import CarDetectionConfig
        from .plate_detection_config import PlateDetectionConfig
        from .histogram_config import HistogramConfig
        from .super_resolution_config import SuperResolutionConfig
        from .ocr_config import OCRConfig
        
        if self.car_detection is None:
            self.car_detection = CarDetectionConfig()
        if self.plate_detection is None:
            self.plate_detection = PlateDetectionConfig()
        if self.histogram is None:
            self.histogram = HistogramConfig()
        if self.super_resolution is None:
            self.super_resolution = SuperResolutionConfig()
        if self.ocr is None:
            self.ocr = OCRConfig()
            
    def get_active_stages(self) -> List[str]:
        """Get list of stages to run based on start and end stage."""
        try:
            start_idx = self.STAGES.index(self.start_stage)
            end_idx = self.STAGES.index(self.end_stage)
            return self.STAGES[start_idx:end_idx + 1]
        except ValueError as e:
            raise ValueError(f"Invalid stage name. Available stages: {self.STAGES}") from e