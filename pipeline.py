from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from PIL import Image
import numpy as np

from configs import PipelineConfig
from utils.logger import Logger, get_logger
from utils.image_utils import ImageUtils
from modules import (
    CarDetector,
    PlateDetector,
    HistogramEqualizer,
    HistogramNegator,
    SuperResolutionInference,
    OCRRecognizer,
)


@dataclass
class PipelineResult:
    """Result container for pipeline output."""
    image_path: str
    cars_detected: int = 0
    plates_detected: int = 0
    plate_sizes: List[Tuple[int, int]] = None  # Original plate sizes before SR
    recognized_texts: List[str] = None
    details: List[Dict[str, Any]] = None
    intermediate_images: Dict[str, List[Image.Image]] = None
    
    def __post_init__(self):
        if self.plate_sizes is None:
            self.plate_sizes = []
        if self.recognized_texts is None:
            self.recognized_texts = []
        if self.details is None:
            self.details = []
        if self.intermediate_images is None:
            self.intermediate_images = {}


class LPRPipeline:
    """License Plate Recognition Pipeline."""
    
    STAGE_ORDER = [
        "car_detection",
        "plate_detection",
        "histogram",
        "negation",
        "super_resolution",
        "ocr"
    ]
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Initialize logger
        Logger.reset()
        log_file = Path(config.debug_output_dir) / "pipeline.log" if config.debug else None
        self.logger = Logger(debug=config.debug, log_file=str(log_file) if log_file else None)
        
        # Debug output directory
        if config.debug and config.save_intermediate:
            self.debug_dir = Path(config.debug_output_dir)
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.debug_dir = None
        
        # Get active stages
        self.active_stages = config.get_active_stages()
        self.logger.info(f"Active stages: {self.active_stages}")
        
        # Initialize modules based on active stages
        self.modules = {}
        self._initialize_modules()
    
    def _initialize_modules(self):
        """Initialize required modules based on active stages."""
        if "car_detection" in self.active_stages:
            self.modules["car_detection"] = CarDetector(self.config.car_detection)
        
        if "plate_detection" in self.active_stages:
            self.modules["plate_detection"] = PlateDetector(self.config.plate_detection)
        
        if "histogram" in self.active_stages:
            self.modules["histogram"] = HistogramEqualizer(self.config.histogram)
        
        if "negation" in self.active_stages:
            self.modules["negation"] = HistogramNegator(self.config.negation)
        
        if "super_resolution" in self.active_stages:
            self.modules["super_resolution"] = SuperResolutionInference(self.config.super_resolution)
        
        if "ocr" in self.active_stages:
            self.modules["ocr"] = OCRRecognizer(self.config.ocr)
    
    def process(
        self, 
        input_path: Union[str, Path, Image.Image, List[Image.Image]]
    ) -> List[PipelineResult]:
        """
        Process input through the pipeline.
        
        Args:
            input_path: Path to image/folder or PIL Image(s)
            
        Returns:
            List of PipelineResult objects
        """
        results = []
        
        # Handle different input types
        if isinstance(input_path, Image.Image):
            images = [("input_image", input_path)]
        elif isinstance(input_path, list):
            images = [(f"image_{i}", img) for i, img in enumerate(input_path)]
        else:
            image_paths = ImageUtils.get_image_paths(input_path)
            images = [(str(p), ImageUtils.load_image(p)) for p in image_paths]
        
        for img_name, img in images:
            self.logger.info(f"Processing: {img_name}")
            result = self._process_single(img_name, img)
            results.append(result)
        
        return results
    
    def _process_single(
        self, 
        image_name: str, 
        image: Image.Image
    ) -> PipelineResult:
        """Process a single image through the pipeline."""
        result = PipelineResult(image_path=image_name)
        
        # Track intermediate images for debug
        intermediate = {}
        
        # Starting images
        current_images = [image]
        
        # Car Detection
        if "car_detection" in self.active_stages:
            self.logger.stage_start("Car Detection")
            detector = self.modules["car_detection"]
            
            all_cars = []
            for img in current_images:
                boxes, cars = detector.detect(img)
                all_cars.extend(cars)
                
                if self.debug_dir:
                    annotated = detector.draw_boxes(img, boxes)
                    intermediate.setdefault("car_detection", []).append(annotated)
            
            result.cars_detected = len(all_cars)
            current_images = all_cars if all_cars else current_images
            
            self.logger.stage_end("Car Detection")
            self.logger.info(f"  Detected {len(all_cars)} cars")
        
        # Plate Detection
        if "plate_detection" in self.active_stages:
            self.logger.stage_start("Plate Detection")
            detector = self.modules["plate_detection"]
            
            plates = []
            plate_sizes = []
            for img in current_images:
                pred, warped = detector.detect(img)
                if warped is not None:
                    plates.append(warped)
                    plate_sizes.append(warped.size)  # Store original plate size
                    if self.debug_dir:
                        annotated = detector.annotate(img, pred)
                        intermediate.setdefault("plate_detection_annotated", []).append(annotated)
                        intermediate.setdefault("plate_detection_warped", []).append(warped)
            
            result.plates_detected = len(plates)
            result.plate_sizes = plate_sizes
            current_images = plates if plates else current_images
            
            self.logger.stage_end("Plate Detection")
            self.logger.info(f"  Detected {len(plates)} plates with sizes: {plate_sizes}")
        
        # Histogram Equalization
        if "histogram" in self.active_stages:
            self.logger.stage_start("Histogram Equalization")
            equalizer = self.modules["histogram"]
            
            equalized = []
            for img in current_images:
                eq = equalizer.equalize(img)
                equalized.append(eq)
                if self.debug_dir:
                    intermediate.setdefault("histogram", []).append(eq)
            
            current_images = equalized
            self.logger.stage_end("Histogram Equalization")
        
        # Negation (after histogram equalization)
        if "negation" in self.active_stages:
            self.logger.stage_start("Histogram-based Negation")
            negator = self.modules["negation"]
            
            processed = []
            for img in current_images:
                neg = negator.process(img)
                processed.append(neg)
                if self.debug_dir:
                    intermediate.setdefault("negation", []).append(neg)
            
            current_images = processed
            self.logger.stage_end("Histogram-based Negation")
        
        # Super Resolution
        if "super_resolution" in self.active_stages:
            self.logger.stage_start("Super Resolution")
            sr_module = self.modules["super_resolution"]
            
            enhanced = []
            for idx, img in enumerate(current_images):
                original_size = img.size
                
                # Check if SR should be applied based on size threshold
                if sr_module.should_apply(img):
                    # Apply SR (automatically resizes to training size first)
                    sr_img = sr_module.enhance_simple(img)
                    enhanced.append(sr_img)
                    
                    self.logger.debug(
                        f"  Plate {idx+1}: {original_size} -> "
                        f"({sr_module.config.training_width}x{sr_module.config.training_height}) -> "
                        f"{sr_img.size}"
                    )
                    
                    if self.debug_dir:
                        intermediate.setdefault("super_resolution", []).append(sr_img)
                else:
                    # Image is large enough, skip SR
                    enhanced.append(img)
                    self.logger.debug(f"  Plate {idx+1}: Skipped SR (size {original_size} above threshold)")
            
            current_images = enhanced
            self.logger.stage_end("Super Resolution")
        
        # OCR
        if "ocr" in self.active_stages:
            self.logger.stage_start("OCR")
            ocr = self.modules["ocr"]
            
            texts = []
            all_details = []
            for img in current_images:
                text, details = ocr.recognize(img)
                texts.append(text)
                all_details.append(details)
            
            result.recognized_texts = texts
            result.details = all_details
            
            self.logger.stage_end("OCR")
            for i, text in enumerate(texts):
                self.logger.info(f"  Plate {i+1}: {text}")
        
        # Save debug images
        if self.debug_dir and intermediate:
            result.intermediate_images = intermediate
            self._save_debug_images(image_name, intermediate)
        
        return result
    
    def _save_debug_images(
        self, 
        image_name: str, 
        intermediate: Dict[str, List[Image.Image]]
    ):
        """Save intermediate images for debugging."""
        base_name = Path(image_name).stem
        
        for stage_idx, (stage_name, images) in enumerate(intermediate.items()):
            stage_dir = self.debug_dir / f"{stage_idx+1:02d}_{stage_name}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            
            for img_idx, img in enumerate(images):
                save_path = stage_dir / f"{base_name}_{img_idx}.png"
                img.save(save_path)
                self.logger.debug(f"Saved: {save_path}")