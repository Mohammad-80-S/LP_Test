#!/usr/bin/env python3
"""
License Plate Recognition Pipeline - Main Entry Point

Usage:
    # Process from car detection to OCR
    python run.py --input path/to/image.jpg
    
    # Process folder
    python run.py --input path/to/folder --output results/
    
    # Start from specific stage
    python run.py --input plate_image.jpg --start-stage histogram --end-stage ocr
    
    # Enable debug mode
    python run.py --input image.jpg --debug
    
    # Disable specific stages
    python run.py --input image.jpg --disable-negation --disable-histogram
"""

import argparse
from pathlib import Path
from PIL import Image

from configs import (
    PipelineConfig,
    CarDetectionConfig,
    PlateDetectionConfig,
    HistogramConfig,
    NegationConfig,
    SuperResolutionConfig,
    OCRConfig,
)
from pipeline import LPRPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="License Plate Recognition Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --input image.jpg
  python run.py --input folder/ --output results/
  python run.py --input plate.jpg --start-stage histogram --end-stage ocr
  python run.py --input image.jpg --debug
  python run.py --input image.jpg --disable-negation
  python run.py --input image.jpg --disable-histogram --disable-negation
        """
    )
    
    # Input/Output
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input image path or folder path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    
    # Pipeline stages
    parser.add_argument(
        "--start-stage",
        type=str,
        default="car_detection",
        choices=["car_detection", "plate_detection", "histogram", "super_resolution", "ocr"],
        help="Starting stage of the pipeline (default: car_detection)"
    )
    parser.add_argument(
        "--end-stage",
        type=str,
        default="ocr",
        choices=["car_detection", "plate_detection", "histogram", "super_resolution", "ocr"],
        help="Ending stage of the pipeline (default: ocr)"
    )
    
    # Stage enable/disable options
    parser.add_argument(
        "--disable-car-detection",
        action="store_true",
        help="Disable car detection stage"
    )
    parser.add_argument(
        "--disable-plate-detection",
        action="store_true",
        help="Disable plate detection stage"
    )
    parser.add_argument(
        "--disable-histogram",
        action="store_true",
        help="Disable histogram equalization stage"
    )
    parser.add_argument(
        "--disable-negation",
        action="store_true",
        help="Disable histogram-based negation (applied before SR)"
    )
    parser.add_argument(
        "--disable-super-resolution",
        action="store_true",
        help="Disable super resolution stage"
    )
    parser.add_argument(
        "--disable-ocr",
        action="store_true",
        help="Disable OCR stage"
    )
    
    # Negation options
    parser.add_argument(
        "--negation-scale",
        type=int,
        default=8,
        help="Scale factor for upscaling before negation analysis (default: 8)"
    )
    parser.add_argument(
        "--no-negation-visualization",
        action="store_true",
        help="Disable saving negation visualization images"
    )
    
    # OCR options
    parser.add_argument(
        "--max-characters",
        type=int,
        default=8,
        help="Maximum number of characters to recognize (default: 8)"
    )
    parser.add_argument(
        "--no-character-limit",
        action="store_true",
        help="Disable character limiting in OCR"
    )
    
    # Debug options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose output and intermediate image saving"
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default="Test_Debug",
        help="Directory for debug outputs (default: Test_Debug)"
    )
    
    # Model paths
    parser.add_argument(
        "--car-model",
        type=str,
        default="yolov8s.pt",
        help="Path to car detection model"
    )
    parser.add_argument(
        "--plate-model",
        type=str,
        default="weights/wpodnet.pth",
        help="Path to plate detection model"
    )
    parser.add_argument(
        "--sr-model",
        type=str,
        default="weights/super_resolution.pth",
        help="Path to super resolution model"
    )
    parser.add_argument(
        "--ocr-model",
        type=str,
        default="weights/ocr_model.pt",
        help="Path to OCR model"
    )
    
    # Super Resolution options
    parser.add_argument(
        "--sr-threshold",
        action="store_true",
        default=True,
        help="Apply SR only to small images (default: True)"
    )
    parser.add_argument(
        "--no-sr-threshold",
        action="store_false",
        dest="sr_threshold",
        help="Apply SR to all images regardless of size"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    
    return parser.parse_args()


def create_config(args) -> PipelineConfig:
    """Create pipeline configuration from arguments."""
    
    # Create sub-configs with enabled flags
    car_config = CarDetectionConfig(
        enabled=not args.disable_car_detection,
        model_path=args.car_model,
        device=args.device,
        debug=args.debug,
    )
    
    plate_config = PlateDetectionConfig(
        enabled=not args.disable_plate_detection,
        model_path=args.plate_model,
        device=args.device,
        debug=args.debug,
    )
    
    histogram_config = HistogramConfig(
        enabled=not args.disable_histogram,
        debug=args.debug,
    )
    
    negation_config = NegationConfig(
        enabled=not args.disable_negation,
        scale_factor=args.negation_scale,
        save_visualization=not args.no_negation_visualization,
        debug=args.debug,
    )
    
    sr_config = SuperResolutionConfig(
        enabled=not args.disable_super_resolution,
        model_path=args.sr_model,
        device=args.device,
        debug=args.debug,
        apply_threshold=args.sr_threshold,
    )
    
    ocr_config = OCRConfig(
        enabled=not args.disable_ocr,
        model_path=args.ocr_model,
        device=args.device,
        debug=args.debug,
        max_characters=args.max_characters,
        limit_characters=not args.no_character_limit,
    )
    
    # Main config
    config = PipelineConfig(
        debug=args.debug,
        debug_output_dir=args.debug_dir,
        save_intermediate=args.debug,
        start_stage=args.start_stage,
        end_stage=args.end_stage,
        input_path=args.input,
        output_dir=args.output,
        device=args.device,
        car_detection=car_config,
        plate_detection=plate_config,
        histogram=histogram_config,
        negation=negation_config,
        super_resolution=sr_config,
        ocr=ocr_config,
    )
    
    return config


def main():
    args = parse_args()
    
    # Create configuration
    config = create_config(args)
    
    # Create pipeline
    pipeline = LPRPipeline(config)
    
    # Process input
    results = pipeline.process(args.input)
    
    # Print results
    print("\n" + "="*60)
    print("PIPELINE RESULTS")
    print("="*60)
    
    for result in results:
        print(f"\nImage: {result.image_path}")
        print(f"  Cars detected: {result.cars_detected}")
        print(f"  Plates detected: {result.plates_detected}")
        
        if result.negation_applied:
            for i, was_negated in enumerate(result.negation_applied):
                status = "Yes" if was_negated else "No"
                print(f"  Plate {i+1} negation applied: {status}")
        
        if result.recognized_texts:
            print(f"  Recognized plates:")
            for i, text in enumerate(result.recognized_texts):
                print(f"    {i+1}. {text}")
    
    print("\n" + "="*60)
    print(f"Total images processed: {len(results)}")
    total_plates = sum(len(r.recognized_texts) for r in results)
    print(f"Total plates recognized: {total_plates}")
    print("="*60)


if __name__ == "__main__":
    main()