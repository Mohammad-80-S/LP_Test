from typing import Union, List, Tuple, Optional
from pathlib import Path
import numpy as np
from PIL import Image

from configs import OCRConfig
from utils.logger import get_logger


class OCRRecognizer:
    """YOLOv8-based OCR for license plate recognition."""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = get_logger()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 OCR model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.config.model_path)
            self.logger.debug(f"Loaded OCR model: {self.config.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load OCR model: {e}")
            raise
    
    def recognize(
        self, 
        image: Union[Image.Image, np.ndarray, str, Path]
    ) -> Tuple[str, List[dict]]:
        """
        Recognize text in license plate image.
        
        Args:
            image: Input plate image
            
        Returns:
            Tuple of (recognized_text, detection_details)
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            img_path = str(image)
        elif isinstance(image, Image.Image):
            # Save temporarily for YOLO
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                image.save(f.name)
                img_path = f.name
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                Image.fromarray(image).save(f.name)
                img_path = f.name
        
        # Run detection
        results = self.model(img_path, verbose=False)
        result = results[0]
        
        # Extract and sort characters
        text, details = self._process_detections(result)
        
        self.logger.debug(f"Recognized: {text}")
        return text, details
    
    def _process_detections(self, result) -> Tuple[str, List[dict]]:
        """Process YOLO detections to get sorted text."""
        detections = result.boxes.data.cpu().numpy()
        
        if len(detections) == 0:
            return "", []
        
        characters = []
        details = []
        
        for det in detections:
            xmin, ymin, xmax, ymax, conf, cls_idx = det
            class_name = self.model.names[int(cls_idx)]
            
            characters.append((xmin, class_name))
            details.append({
                'class': class_name,
                'confidence': float(conf),
                'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]
            })
        
        # Sort by x position (left to right)
        characters.sort(key=lambda x: x[0])
        text = "".join([char[1] for char in characters])
        
        return text, details
    
    def batch_recognize(
        self, 
        images: List[Union[Image.Image, str, Path]]
    ) -> List[Tuple[str, List[dict]]]:
        """Recognize text in multiple images."""
        results = []
        for img in images:
            results.append(self.recognize(img))
        return results