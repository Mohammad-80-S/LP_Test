from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class BaseConfig:
    """Base configuration class with common settings."""
    enabled: bool = True  # Enable/disable this stage
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
    negation: Optional["NegationConfig"] = None  # Negation is integrated with SR
    super_resolution: Optional["SuperResolutionConfig"] = None
    ocr: Optional["OCRConfig"] = None
    
    # Supported stages in order (negation is not a separate stage, it's part of SR)
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
        from .negation_config import NegationConfig
        from .super_resolution_config import SuperResolutionConfig
        from .ocr_config import OCRConfig
        
        if self.car_detection is None:
            self.car_detection = CarDetectionConfig()
        if self.plate_detection is None:
            self.plate_detection = PlateDetectionConfig()
        if self.histogram is None:
            self.histogram = HistogramConfig()
        if self.negation is None:
            self.negation = NegationConfig()
        if self.super_resolution is None:
            self.super_resolution = SuperResolutionConfig()
        if self.ocr is None:
            self.ocr = OCRConfig()
            
    def get_active_stages(self) -> List[str]:
        """Get list of stages to run based on start/end stage and enabled flags."""
        try:
            start_idx = self.STAGES.index(self.start_stage)
            end_idx = self.STAGES.index(self.end_stage)
            stages_in_range = self.STAGES[start_idx:end_idx + 1]
            
            # Filter by enabled flag
            active_stages = []
            stage_configs = {
                "car_detection": self.car_detection,
                "plate_detection": self.plate_detection,
                "histogram": self.histogram,
                "super_resolution": self.super_resolution,
                "ocr": self.ocr,
            }
            
            for stage in stages_in_range:
                config = stage_configs.get(stage)
                if config is None or config.enabled:
                    active_stages.append(stage)
                    
            return active_stages
        except ValueError as e:
            raise ValueError(f"Invalid stage name. Available stages: {self.STAGES}") from e