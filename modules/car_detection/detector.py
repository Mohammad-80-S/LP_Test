from typing import List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from configs import CarDetectionConfig
from utils.logger import get_logger


class CarDetector:
    """YOLOv8-based car detection module."""
    
    def __init__(self, config: CarDetectionConfig):
        self.config = config
        self.logger = get_logger()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.config.model_path)
            self.logger.debug(f"Loaded car detection model: {self.config.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load car detection model: {e}")
            raise
    
    def detect(
        self, 
        image: Union[Image.Image, np.ndarray, str, Path]
    ) -> Tuple[List[Tuple[int, int, int, int]], List[Image.Image]]:
        """
        Detect cars in the image.
        
        Args:
            image: Input image (PIL Image, numpy array, or path)
            
        Returns:
            Tuple of (bounding_boxes, cropped_car_images)
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image
        
        img_array = np.array(pil_img)
        
        # Run detection
        results = self.model(img_array, verbose=False)
        boxes = self._filter_boxes_by_class(
            results[0].boxes, 
            self.config.target_class_id
        )
        
        self.logger.debug(f"Detected {len(boxes)} vehicles")
        
        if len(boxes) == 0:
            self.logger.warning("No cars detected in the image")
            return [], []
        
        # Crop car images
        cropped_cars = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped = pil_img.crop((x1, y1, x2, y2))
            cropped_cars.append(cropped)
        
        return boxes, cropped_cars
    
    def _filter_boxes_by_class(self, boxes, cls_id: int) -> List[Tuple[float, ...]]:
        """Filter detection boxes by class ID."""
        mask = boxes.cls == cls_id
        filtered_xyxy = boxes.xyxy[mask]
        return filtered_xyxy.cpu().tolist()
    
    def draw_boxes(
        self, 
        image: Image.Image, 
        boxes: List[Tuple[int, int, int, int]],
        color: Tuple[int, int, int] = (0, 0, 255),
        width: int = 4
    ) -> Image.Image:
        """Draw bounding boxes on image."""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        
        return img_copy