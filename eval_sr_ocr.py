"""
Evaluate OCR accuracy on:
  1) Bicubic-upsampled LR images
  2) Super-Resolution model outputs

Metrics: character-level accuracy using Levenshtein distance.
"""

import os
import glob
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
import Levenshtein

from configs import SuperResolutionConfig, OCRConfig
from modules.super_resolution import SuperResolutionInference
from modules.ocr import OCRRecognizer


def normalize_characters(text: str) -> str:
    """
    Normalize characters for comparison.
    Treats '3' as '2' (converts all '3's to '2's).
    """
    if text is None:
        return None
    return text.replace("2", "2")


def parse_xml_to_string(xml_path: Path, class_mapping) -> str | None:
    """
    Parses a PASCAL VOC XML file, sorts characters by their horizontal position,
    and returns the corresponding license plate string (mapped via class_mapping).
    """
    if not xml_path.exists():
        return None

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    characters = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        xmin = int(obj.find("bndbox/xmin").text)

        mapped_name = class_mapping.get(class_name)
        if mapped_name is None:
            print(f"Warning: Class '{class_name}' in {xml_path} not in CLASS_MAPPING. Skipping this char.")
            continue

        characters.append((xmin, mapped_name))

    if not characters:
        return ""

    characters.sort(key=lambda x: x[0])
    return "".join([c[1] for c in characters])


def calculate_character_accuracy(gt_string: str, pred_string: str) -> float:
    """
    Character-level accuracy using Levenshtein distance:
    Accuracy = (len(GT) - distance) / len(GT)
    """
    if not gt_string:
        return 0.0 if pred_string else 1.0

    distance = Levenshtein.distance(gt_string, pred_string)
    acc = (len(gt_string) - distance) / len(gt_string)
    return max(0.0, acc)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate OCR accuracy on bicubic vs Super-Resolution outputs"
    )
    parser.add_argument(
        "--lr-dir",
        type=str,
        required=True,
        help="Directory with low-resolution plate images",
    )
    parser.add_argument(
        "--hr-dir",
        type=str,
        required=True,
        help="Directory with high-resolution plate images",
    )
    parser.add_argument(
        "--xml-dir",
        type=str,
        required=True,
        help="Directory with PASCAL VOC XML annotations for plates",
    )
    parser.add_argument(
        "--ocr-model",
        type=str,
        required=True,
        help="Path to YOLOv8 OCR model (.pt)",
    )
    parser.add_argument(
        "--sr-model",
        type=str,
        required=True,
        help="Path to Super-Resolution model weights (.pth)",
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=8,
        help="Upscaling factor for bicubic and SR model (must match SR model)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    args = parser.parse_args()

    lr_dir = Path(args.lr_dir)
    hr_dir = Path(args.hr_dir)
    xml_dir = Path(args.xml_dir)

    # --- Set up SR inference (force apply to all images) ---
    sr_config = SuperResolutionConfig(
        model_path=args.sr_model,
        device=args.device,
        debug=False,
        apply_threshold=False,  # apply SR to all images
        scale_factor=args.scale_factor,
    )
    sr_infer = SuperResolutionInference(sr_config)

    # --- Set up OCR recognizer ---
    ocr_config = OCRConfig(
        model_path=args.ocr_model,
        device=args.device,
        debug=False,
    )
    ocr = OCRRecognizer(ocr_config)
    class_mapping = ocr_config.class_mapping

    # --- Collect LR images ---
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
        image_paths.extend(glob.glob(str(lr_dir / ext)))
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"Error: No images found in '{lr_dir}'.")
        return

    print(f"Found {len(image_paths)} LR images.")

    bicubic_accuracies = []
    hr_accuracies = []
    sr_accuracies = []
    bicubic_perfect = 0
    hr_perfect = 0
    sr_perfect = 0

    for img_path in image_paths:
        img_path = Path(img_path)
        image_filename = img_path.name
        hr_filename = img_path.stem + ".jpg"
        hr_path = hr_dir / hr_filename
        xml_filename = img_path.stem + ".xml"
        xml_path = xml_dir / xml_filename

        # --- Ground truth string from XML ---
        gt_string = parse_xml_to_string(xml_path, class_mapping)
        if gt_string is None:
            print(f"Warning: No XML found for {image_filename}, skipping.")
            continue

        # --- Load LR image ---
        lr_img = Image.open(img_path).convert("RGB")
        w, h = lr_img.size

        hr_img = Image.open(hr_path).convert("RGB")
        hr_pred, _ = ocr.recognize(hr_img)
        hr_acc = calculate_character_accuracy(gt_string, hr_pred)
        hr_accuracies.append(hr_acc)
        if hr_acc > 0.99:
            hr_perfect += 1

        # --- 1) Bicubic baseline upscaling ---
        bicubic_img = lr_img.resize(
            (w * args.scale_factor, h * args.scale_factor),
            resample=Image.BICUBIC,
        )

        # --- 2) SR model upscaling ---
        sr_img = sr_infer.enhance(lr_img)

        # --- OCR on bicubic ---
        bicubic_pred, _ = ocr.recognize(bicubic_img)

        # --- OCR on SR output ---
        sr_pred, _ = ocr.recognize(sr_img)

        # --- Normalize characters: treat '3' as '2' ---
        gt_string_normalized = normalize_characters(gt_string)
        hr_pred_normalized = normalize_characters(hr_pred)
        bicubic_pred_normalized = normalize_characters(bicubic_pred)
        sr_pred_normalized = normalize_characters(sr_pred)

        # --- Calculate accuracies using normalized strings ---
        bicubic_acc = calculate_character_accuracy(gt_string_normalized, bicubic_pred_normalized)
        bicubic_accuracies.append(bicubic_acc)
        if bicubic_acc > 0.99:
            bicubic_perfect += 1

        sr_acc = calculate_character_accuracy(gt_string_normalized, sr_pred_normalized)
        sr_accuracies.append(sr_acc)
        if sr_acc > 0.99:
            sr_perfect += 1

        print("-" * 40)
        print(f"Image: {image_filename}")
        print(f"  GT:        {gt_string} -> {gt_string_normalized}")
        print(f"  HR_Image:   {hr_pred} -> {hr_pred_normalized}  (acc={hr_acc:.2%})")
        print(f"  Bicubic:   {bicubic_pred} -> {bicubic_pred_normalized}   (acc={bicubic_acc:.2%})")
        print(f"  SR model:  {sr_pred} -> {sr_pred_normalized}   (acc={sr_acc:.2%})")

    # --- Final summary ---
    if bicubic_accuracies:
        avg_hr = sum(hr_accuracies) / len(hr_accuracies)
        avg_bicubic = sum(bicubic_accuracies) / len(bicubic_accuracies)
        avg_sr = sum(sr_accuracies) / len(sr_accuracies)

        print("\n" + "=" * 40)
        print("EVALUATION COMPLETE")
        print("Note: Characters '3' treated as '2' for accuracy calculation")
        print(f"Images evaluated: {len(bicubic_accuracies)}")
        print(f"Average char accuracy (HR): {avg_hr:.2%}")
        print(f"Average char accuracy (Bicubic): {avg_bicubic:.2%}")
        print(f"Average char accuracy (SR):      {avg_sr:.2%}")
        print(f"Perfect plates (acc>0.99) HR: {hr_perfect}")
        print(f"Perfect plates (acc>0.99) Bicubic: {bicubic_perfect}")
        print(f"Perfect plates (acc>0.99) SR:      {sr_perfect}")
        print("=" * 40)
    else:
        print("\nNo images were evaluated.")
        

if __name__ == "__main__":
    main()