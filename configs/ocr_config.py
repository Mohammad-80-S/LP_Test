from dataclasses import dataclass
from typing import Dict
from .base_config import BaseConfig


@dataclass
class OCRConfig(BaseConfig):
    """Configuration for YOLOv8 OCR."""
    model_path: str = "weights/ocr_model.pt"
    confidence_threshold: float = 0.5
    output_subdir: str = "05_ocr"
    
    # Class mapping for Persian license plates
    class_mapping: Dict[str, str] = None
    
    def __post_init__(self):
        if self.class_mapping is None:
            self.class_mapping = {
                "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
                "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
                "الف": "A", "ب": "B", "د": "D", "ع": "Ein",
                "ق": "Gh", "ه‍": "H", "ج": "J", "ل": "L",
                "م": "M", "ن": "N", "ص": "Sad", "س": "Sin",
                "ت": "T", "ط": "Ta", "و": "V", "ی": "Y",
                "ژ (معلولین و جانبازان)": "Zh",
            }