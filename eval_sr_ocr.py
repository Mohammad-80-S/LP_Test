"""
Evaluate OCR accuracy on:
  1) Bicubic-upsampled LR images
  2) Super-Resolution model outputs

Metrics: character-level accuracy using Levenshtein distance.
Includes confusion matrix generation for OCR character predictions.
"""

import os
import glob
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict
import re

from PIL import Image
import Levenshtein
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from configs import SuperResolutionConfig, OCRConfig
from modules.super_resolution import SuperResolutionInference
from modules.ocr import OCRRecognizer


# Define multi-character tokens that should be treated as single units
MULTI_CHAR_TOKENS = ["Ein", "Gh", "Sad", "Sin", "Ta", "Zh"]


def tokenize_plate_string(text: str) -> list:
    """
    Tokenize a plate string into individual tokens.
    Multi-character class names (Ein, Gh, Sad, Sin, Ta, Zh) are treated as single tokens.
    
    Args:
        text: The plate string (e.g., "12Ein45Gh7")
    
    Returns:
        List of tokens (e.g., ["1", "2", "Ein", "4", "5", "Gh", "7"])
    """
    if not text:
        return []
    
    tokens = []
    i = 0
    
    while i < len(text):
        matched = False
        # Check for multi-character tokens (longest match first)
        for token in sorted(MULTI_CHAR_TOKENS, key=len, reverse=True):
            if text[i:].startswith(token):
                tokens.append(token)
                i += len(token)
                matched = True
                break
        
        if not matched:
            # Single character token
            tokens.append(text[i])
            i += 1
    
    return tokens


def normalize_characters(text: str) -> str:
    """
    Normalize characters for comparison.
    Treats '3' as '2' (converts all '3's to '2's).
    """
    if text is None:
        return None
    return text.replace("2", "2")


def normalize_tokens(tokens: list) -> list:
    """
    Normalize tokens for comparison.
    """
    if not tokens:
        return []
    return [normalize_characters(t) for t in tokens]


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


def parse_xml_to_tokens(xml_path: Path, class_mapping) -> list | None:
    """
    Parses a PASCAL VOC XML file, sorts characters by their horizontal position,
    and returns the corresponding license plate as a list of tokens.
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
        return []

    characters.sort(key=lambda x: x[0])
    return [c[1] for c in characters]


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


def calculate_token_accuracy(gt_tokens: list, pred_tokens: list) -> float:
    """
    Token-level accuracy using Levenshtein distance on token lists.
    """
    if not gt_tokens:
        return 0.0 if pred_tokens else 1.0
    
    # Convert token lists to strings for Levenshtein (using separator)
    gt_str = "\x00".join(gt_tokens)
    pred_str = "\x00".join(pred_tokens)
    
    distance = Levenshtein.distance(gt_str, pred_str)
    # Approximate token-level distance
    gt_len = len(gt_tokens)
    
    # Count token differences more accurately
    token_distance = 0
    alignments = align_tokens_for_confusion(gt_tokens, pred_tokens)
    for gt_tok, pred_tok in alignments:
        if gt_tok != pred_tok:
            token_distance += 1
    
    acc = (gt_len - token_distance) / gt_len if gt_len > 0 else 1.0
    return max(0.0, acc)


def align_tokens_for_confusion(gt_tokens: list, pred_tokens: list) -> list:
    """
    Align ground truth and predicted token lists.
    Uses dynamic programming for alignment.
    Returns list of (gt_token, pred_token) tuples.
    """
    alignments = []
    
    if not gt_tokens and not pred_tokens:
        return alignments
    
    if not gt_tokens:
        for p in pred_tokens:
            alignments.append(("", p))
        return alignments
    
    if not pred_tokens:
        for g in gt_tokens:
            alignments.append((g, ""))
        return alignments
    
    m, n = len(gt_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt_tokens[i-1] == pred_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    # Backtrack to find alignment
    i, j = m, n
    aligned_pairs = []
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and gt_tokens[i-1] == pred_tokens[j-1]:
            aligned_pairs.append((gt_tokens[i-1], pred_tokens[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            # Substitution
            aligned_pairs.append((gt_tokens[i-1], pred_tokens[j-1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            # Insertion in prediction
            aligned_pairs.append(("", pred_tokens[j-1]))
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            # Deletion (missed in prediction)
            aligned_pairs.append((gt_tokens[i-1], ""))
            i -= 1
        else:
            # Fallback
            if j > 0:
                aligned_pairs.append(("", pred_tokens[j-1]))
                j -= 1
            elif i > 0:
                aligned_pairs.append((gt_tokens[i-1], ""))
                i -= 1
    
    aligned_pairs.reverse()
    return aligned_pairs


class ConfusionMatrixBuilder:
    """Build and visualize confusion matrix for OCR tokens (characters/classes)."""
    
    def __init__(self, name: str = "OCR"):
        self.name = name
        self.confusion_counts = defaultdict(lambda: defaultdict(int))
        self.all_tokens = set()
    
    def add_prediction(self, gt_tokens: list, pred_tokens: list):
        """Add a prediction pair to the confusion matrix using token lists."""
        if gt_tokens is None:
            gt_tokens = []
        if pred_tokens is None:
            pred_tokens = []
        
        alignments = align_tokens_for_confusion(gt_tokens, pred_tokens)
        
        for gt_token, pred_token in alignments:
            if gt_token:  # Only count if there's a ground truth token
                pred_label = pred_token if pred_token else "<MISS>"
                self.confusion_counts[gt_token][pred_label] += 1
                self.all_tokens.add(gt_token)
                if pred_token:
                    self.all_tokens.add(pred_token)
    
    def _sort_tokens(self, tokens):
        """Sort tokens: digits first (0-9), then single letters (A-Z), then multi-char tokens."""
        def sort_key(x):
            if x.isdigit():
                return (0, int(x))  # Digits first, sorted numerically
            elif len(x) == 1 and x.isalpha():
                return (1, x)  # Single letters second
            else:
                return (2, x)  # Multi-char tokens last
        return sorted(tokens, key=sort_key)
    
    def build_matrix(self):
        """Build the confusion matrix as numpy array."""
        tokens = self._sort_tokens(self.all_tokens)
        tokens_with_miss = tokens + ["<MISS>"]
        
        n_gt = len(tokens)
        n_pred = len(tokens_with_miss)
        
        matrix = np.zeros((n_gt, n_pred), dtype=int)
        
        for i, gt_token in enumerate(tokens):
            for j, pred_token in enumerate(tokens_with_miss):
                matrix[i, j] = self.confusion_counts[gt_token][pred_token]
        
        return matrix, tokens, tokens_with_miss
    
    def plot_and_save(self, save_path: str, figsize: tuple = None):
        """Plot confusion matrix and save as image."""
        matrix, gt_labels, pred_labels = self.build_matrix()
        
        if len(gt_labels) == 0:
            print(f"Warning: No data to plot for {self.name} confusion matrix")
            return
        
        # Determine figure size based on matrix size
        if figsize is None:
            n_labels = max(len(gt_labels), len(pred_labels))
            figsize = (max(14, n_labels * 0.6), max(10, len(gt_labels) * 0.5))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=pred_labels,
            yticklabels=gt_labels,
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        
        ax.set_xlabel('Predicted Token', fontsize=12)
        ax.set_ylabel('Ground Truth Token', fontsize=12)
        ax.set_title(f'Confusion Matrix - {self.name}', fontsize=14)
        
        # Rotate x labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved: {save_path}")
    
    def print_summary(self):
        """Print summary statistics from confusion matrix."""
        matrix, gt_labels, pred_labels = self.build_matrix()
        
        if len(gt_labels) == 0:
            print(f"No data for {self.name}")
            return
        
        print(f"\n{self.name} - Per-token accuracy:")
        print("-" * 50)
        
        total_correct = 0
        total_count = 0
        
        for i, gt_token in enumerate(gt_labels):
            row_sum = matrix[i, :].sum()
            if row_sum > 0:
                if gt_token in pred_labels:
                    correct_idx = pred_labels.index(gt_token)
                    correct = matrix[i, correct_idx]
                else:
                    correct = 0
                accuracy = correct / row_sum * 100
                total_correct += correct
                total_count += row_sum
                print(f"  '{gt_token}': {correct}/{row_sum} ({accuracy:.1f}%)")
        
        if total_count > 0:
            overall_acc = total_correct / total_count * 100
            print(f"\n  Overall: {total_correct}/{total_count} ({overall_acc:.1f}%)")


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
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save confusion matrix images (default: eval_results)",
    )
    args = parser.parse_args()

    lr_dir = Path(args.lr_dir)
    hr_dir = Path(args.hr_dir)
    xml_dir = Path(args.xml_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Set up SR inference (force apply to all images) ---
    sr_config = SuperResolutionConfig(
        model_path=args.sr_model,
        device=args.device,
        debug=False,
        apply_threshold=False,
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

    # Initialize confusion matrix builders
    cm_hr = ConfusionMatrixBuilder(name="HR (High Resolution)")
    cm_bicubic = ConfusionMatrixBuilder(name="Bicubic Upsampling")
    cm_sr = ConfusionMatrixBuilder(name="Super Resolution")

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

        # --- Ground truth string and tokens from XML ---
        gt_string = parse_xml_to_string(xml_path, class_mapping)
        gt_tokens = parse_xml_to_tokens(xml_path, class_mapping)
        
        if gt_string is None:
            print(f"Warning: No XML found for {image_filename}, skipping.")
            continue

        # --- Load LR image ---
        lr_img = Image.open(img_path).convert("RGB")
        w, h = lr_img.size

        # --- HR image OCR ---
        hr_img = Image.open(hr_path).convert("RGB")
        hr_pred, _ = ocr.recognize(hr_img)
        hr_pred_tokens = tokenize_plate_string(hr_pred) if hr_pred else []
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
        bicubic_pred_tokens = tokenize_plate_string(bicubic_pred) if bicubic_pred else []

        # --- OCR on SR output ---
        sr_pred, _ = ocr.recognize(sr_img)
        sr_pred_tokens = tokenize_plate_string(sr_pred) if sr_pred else []

        # --- Normalize strings for accuracy calculation ---
        gt_string_normalized = normalize_characters(gt_string)
        hr_pred_normalized = normalize_characters(hr_pred)
        bicubic_pred_normalized = normalize_characters(bicubic_pred)
        sr_pred_normalized = normalize_characters(sr_pred)

        # --- Add to confusion matrices (using tokens) ---
        cm_hr.add_prediction(gt_tokens, hr_pred_tokens)
        cm_bicubic.add_prediction(gt_tokens, bicubic_pred_tokens)
        cm_sr.add_prediction(gt_tokens, sr_pred_tokens)

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
        print(f"  GT:         {gt_string} -> tokens: {gt_tokens}")
        print(f"  HR_Image:   {hr_pred} -> tokens: {hr_pred_tokens}  (acc={hr_acc:.2%})")
        print(f"  Bicubic:    {bicubic_pred} -> tokens: {bicubic_pred_tokens}  (acc={bicubic_acc:.2%})")
        print(f"  SR model:   {sr_pred} -> tokens: {sr_pred_tokens}  (acc={sr_acc:.2%})")

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
        
        # --- Generate and save confusion matrices ---
        print("\n" + "=" * 40)
        print("GENERATING CONFUSION MATRICES")
        print("=" * 40)
        
        # Save confusion matrix images
        cm_hr.plot_and_save(str(output_dir / "confusion_matrix_hr.png"))
        cm_bicubic.plot_and_save(str(output_dir / "confusion_matrix_bicubic.png"))
        cm_sr.plot_and_save(str(output_dir / "confusion_matrix_sr.png"))
        
        # Print per-token summaries
        cm_hr.print_summary()
        cm_bicubic.print_summary()
        cm_sr.print_summary()
        
        # Display confusion matrices
        print("\n" + "=" * 40)
        print("DISPLAYING CONFUSION MATRICES")
        print("=" * 40)
        
        # Create a combined figure for display
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        
        for idx, (cm, ax, title) in enumerate([
            (cm_hr, axes[0], "HR (High Resolution)"),
            (cm_bicubic, axes[1], "Bicubic Upsampling"),
            (cm_sr, axes[2], "Super Resolution")
        ]):
            matrix, gt_labels, pred_labels = cm.build_matrix()
            if len(gt_labels) > 0:
                sns.heatmap(
                    matrix,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=pred_labels,
                    yticklabels=gt_labels,
                    ax=ax,
                    cbar_kws={'label': 'Count'}
                )
                ax.set_xlabel('Predicted Token', fontsize=10)
                ax.set_ylabel('Ground Truth Token', fontsize=10)
                ax.set_title(f'Confusion Matrix - {title}', fontsize=12)
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
                ax.set_title(f'Confusion Matrix - {title}', fontsize=12)
        
        plt.tight_layout()
        combined_path = output_dir / "confusion_matrix_combined.png"
        plt.savefig(str(combined_path), dpi=150, bbox_inches='tight')
        print(f"Combined confusion matrix saved: {combined_path}")
        
        # Show the plot
        plt.show()
        
    else:
        print("\nNo images were evaluated.")
        

if __name__ == "__main__":
    main()