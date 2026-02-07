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
from matplotlib.colors import LinearSegmentedColormap

from configs import SuperResolutionConfig, OCRConfig
from modules.super_resolution import SuperResolutionInference
from modules.ocr import OCRRecognizer


# Define multi-character tokens that should be treated as single units
MULTI_CHAR_TOKENS = ["Ein", "Gh", "Sad", "Sin", "Ta", "Zh"]

# Background token for confusion matrix
BG_TOKEN = "<BG>"


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


def align_tokens_for_confusion(gt_tokens: list, pred_tokens: list, return_operations: bool = False):
    """
    Align ground truth and predicted token lists using Needleman-Wunsch algorithm.
    This properly handles insertions, deletions, and substitutions.
    
    Returns list of (gt_token, pred_token, operation) tuples where operation is:
    - 'match': tokens are identical
    - 'substitution': tokens differ
    - 'deletion': gt token has no corresponding pred token (gt=char, pred=background)
    - 'insertion': pred token has no corresponding gt token (gt=background, pred=char)
    """
    if not gt_tokens and not pred_tokens:
        return []
    
    if not gt_tokens:
        if return_operations:
            return [(BG_TOKEN, p, "insertion") for p in pred_tokens]
        return [(BG_TOKEN, p) for p in pred_tokens]
    
    if not pred_tokens:
        if return_operations:
            return [(g, BG_TOKEN, "deletion") for g in gt_tokens]
        return [(g, BG_TOKEN) for g in gt_tokens]
    
    m, n = len(gt_tokens), len(pred_tokens)
    
    # Gap penalty
    gap = -1
    # Match score
    match_score = 2
    # Mismatch penalty
    mismatch = -1
    
    # Initialize score matrix
    score = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize traceback matrix
    # 0: diagonal, 1: up (deletion), 2: left (insertion)
    traceback = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize first row and column
    for i in range(m + 1):
        score[i][0] = i * gap
        traceback[i][0] = 1  # up
    for j in range(n + 1):
        score[0][j] = j * gap
        traceback[0][j] = 2  # left
    
    # Fill the matrices
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt_tokens[i-1] == pred_tokens[j-1]:
                diag = score[i-1][j-1] + match_score
            else:
                diag = score[i-1][j-1] + mismatch
            
            up = score[i-1][j] + gap
            left = score[i][j-1] + gap
            
            max_score = max(diag, up, left)
            score[i][j] = max_score
            
            if max_score == diag:
                traceback[i][j] = 0
            elif max_score == up:
                traceback[i][j] = 1
            else:
                traceback[i][j] = 2
    
    # Traceback to find alignment
    aligned_pairs = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and traceback[i][j] == 0:
            # Diagonal: match or substitution
            gt_tok = gt_tokens[i-1]
            pred_tok = pred_tokens[j-1]
            if gt_tok == pred_tok:
                op = "match"
            else:
                op = "substitution"
            if return_operations:
                aligned_pairs.append((gt_tok, pred_tok, op))
            else:
                aligned_pairs.append((gt_tok, pred_tok))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or traceback[i][j] == 1):
            # Up: deletion (GT token not in prediction -> predicted as background)
            if return_operations:
                aligned_pairs.append((gt_tokens[i-1], BG_TOKEN, "deletion"))
            else:
                aligned_pairs.append((gt_tokens[i-1], BG_TOKEN))
            i -= 1
        else:
            # Left: insertion (extra token in prediction -> background recognized as char)
            if return_operations:
                aligned_pairs.append((BG_TOKEN, pred_tokens[j-1], "insertion"))
            else:
                aligned_pairs.append((BG_TOKEN, pred_tokens[j-1]))
            j -= 1
    
    aligned_pairs.reverse()
    return aligned_pairs


def print_alignment_details(gt_tokens: list, pred_tokens: list, label: str = ""):
    """
    Print detailed alignment information for debugging.
    """
    alignments = align_tokens_for_confusion(gt_tokens, pred_tokens, return_operations=True)
    
    print(f"\n  Alignment details for {label}:")
    print(f"  {'GT Token':<12} {'Pred Token':<12} {'Operation':<15}")
    print(f"  {'-'*12} {'-'*12} {'-'*15}")
    
    for item in alignments:
        gt_tok, pred_tok, op = item
        gt_display = gt_tok if gt_tok != BG_TOKEN else "(background)"
        pred_display = pred_tok if pred_tok != BG_TOKEN else "(background)"
        
        # Add symbol for clarity
        if op == "match":
            symbol = "✓"
        elif op == "substitution":
            symbol = "✗ (sub)"
        elif op == "deletion":
            symbol = "✗ (del) - missed character"
        else:  # insertion
            symbol = "✗ (ins) - false positive"
        
        print(f"  {gt_display:<12} {pred_display:<12} {symbol}")
    
    return alignments


def create_white_green_cmap():
    """Create a custom colormap from white to green."""
    colors = ['#FFFFFF', '#E8F5E9', '#C8E6C9', '#A5D6A7', '#81C784', 
              '#66BB6A', '#4CAF50', '#43A047', '#388E3C', '#2E7D32', '#1B5E20']
    return LinearSegmentedColormap.from_list('WhiteGreen', colors, N=256)


class ConfusionMatrixBuilder:
    """Build and visualize confusion matrix for OCR tokens (characters/classes)."""
    
    def __init__(self, name: str = "OCR"):
        self.name = name
        self.confusion_counts = defaultdict(lambda: defaultdict(int))
        self.all_gt_tokens = set()
        self.all_pred_tokens = set()
        self.total_insertions = 0
        self.total_deletions = 0
    
    def add_prediction(self, gt_tokens: list, pred_tokens: list):
        """Add a prediction pair to the confusion matrix using token lists."""
        if gt_tokens is None:
            gt_tokens = []
        if pred_tokens is None:
            pred_tokens = []
        
        alignments = align_tokens_for_confusion(gt_tokens, pred_tokens, return_operations=True)
        
        for item in alignments:
            gt_token, pred_token, operation = item
            
            if operation == "match":
                # Correct prediction
                self.confusion_counts[gt_token][gt_token] += 1
                self.all_gt_tokens.add(gt_token)
                self.all_pred_tokens.add(gt_token)
                
            elif operation == "substitution":
                # Wrong prediction (one char confused with another)
                self.confusion_counts[gt_token][pred_token] += 1
                self.all_gt_tokens.add(gt_token)
                self.all_pred_tokens.add(pred_token)
                
            elif operation == "deletion":
                # GT token was missed -> predicted as background
                # Row: actual character, Column: <BG>
                self.confusion_counts[gt_token][BG_TOKEN] += 1
                self.all_gt_tokens.add(gt_token)
                self.all_pred_tokens.add(BG_TOKEN)
                self.total_deletions += 1
                
            elif operation == "insertion":
                # Background was predicted as a character (false positive)
                # Row: <BG>, Column: predicted character
                self.confusion_counts[BG_TOKEN][pred_token] += 1
                self.all_gt_tokens.add(BG_TOKEN)
                self.all_pred_tokens.add(pred_token)
                self.total_insertions += 1
    
    def _sort_tokens(self, tokens):
        """Sort tokens: digits first (0-9), then single letters (A-Z), then multi-char tokens, then <BG>."""
        def sort_key(x):
            if x == BG_TOKEN:
                return (4, x)  # <BG> at the end
            elif x.isdigit():
                return (0, int(x))  # Digits first, sorted numerically
            elif len(x) == 1 and x.isalpha():
                return (1, x)  # Single letters second
            else:
                return (2, x)  # Multi-char tokens third
        return sorted(tokens, key=sort_key)
    
    def build_matrix(self):
        """Build the confusion matrix as numpy array."""
        # Get all unique tokens and sort them
        all_tokens = self.all_gt_tokens | self.all_pred_tokens
        sorted_tokens = self._sort_tokens(all_tokens)
        
        n = len(sorted_tokens)
        matrix = np.zeros((n, n), dtype=int)
        
        for i, gt_token in enumerate(sorted_tokens):
            for j, pred_token in enumerate(sorted_tokens):
                matrix[i, j] = self.confusion_counts[gt_token][pred_token]
        
        return matrix, sorted_tokens, sorted_tokens
    
    def build_normalized_matrix(self):
        """Build the normalized confusion matrix (row-wise normalization)."""
        matrix, gt_tokens, pred_tokens = self.build_matrix()
        
        # Normalize each row (each ground truth class)
        row_sums = matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        normalized_matrix = matrix.astype(float) / row_sums
        
        return normalized_matrix, gt_tokens, pred_tokens
    
    def plot_and_save(self, save_path: str, figsize: tuple = None, normalized: bool = False):
        """Plot confusion matrix and save as image."""
        if normalized:
            matrix, gt_labels, pred_labels = self.build_normalized_matrix()
            fmt = '.2f'
            title_suffix = "(Normalized)"
            cbar_label = 'Proportion'
            vmin, vmax = 0, 1
        else:
            matrix, gt_labels, pred_labels = self.build_matrix()
            fmt = 'd'
            title_suffix = "(Count)"
            cbar_label = 'Count'
            vmin, vmax = None, None
        
        if len(gt_labels) == 0:
            print(f"Warning: No data to plot for {self.name} confusion matrix")
            return
        
        # Determine figure size based on matrix size
        if figsize is None:
            n_labels = len(gt_labels)
            figsize = (max(14, n_labels * 0.6), max(10, n_labels * 0.5))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use custom white-green colormap
        cmap = create_white_green_cmap()
        
        # Create heatmap
        sns.heatmap(
            matrix,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=pred_labels,
            yticklabels=gt_labels,
            ax=ax,
            cbar_kws={'label': cbar_label},
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            linecolor='lightgray'
        )
        
        ax.set_xlabel('Predicted Token', fontsize=12)
        ax.set_ylabel('Ground Truth Token', fontsize=12)
        ax.set_title(f'Confusion Matrix - {self.name} {title_suffix}', fontsize=14)
        
        # Rotate x labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        
        # Highlight <BG> row and column with different background
        if BG_TOKEN in gt_labels:
            bg_idx = gt_labels.index(BG_TOKEN)
            # Add a subtle highlight to the BG row and column
            ax.axhline(y=bg_idx, color='orange', linewidth=2)
            ax.axhline(y=bg_idx + 1, color='orange', linewidth=2)
            ax.axvline(x=bg_idx, color='orange', linewidth=2)
            ax.axvline(x=bg_idx + 1, color='orange', linewidth=2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved: {save_path}")
    
    def print_summary(self):
        """Print summary statistics from confusion matrix."""
        matrix, labels, _ = self.build_matrix()
        
        if len(labels) == 0:
            print(f"No data for {self.name}")
            return
        
        print(f"\n{'='*60}")
        print(f"{self.name} - Per-token accuracy:")
        print(f"{'='*60}")
        
        total_correct = 0
        total_count = 0
        
        for i, gt_token in enumerate(labels):
            row_sum = matrix[i, :].sum()
            if row_sum > 0:
                # Correct predictions are on the diagonal
                correct = matrix[i, i]
                accuracy = correct / row_sum * 100
                
                if gt_token != BG_TOKEN:
                    total_correct += correct
                    total_count += row_sum
                
                # Show common confusions
                confusions = []
                for j, pred_token in enumerate(labels):
                    if i != j and matrix[i, j] > 0:
                        confusions.append(f"{pred_token}:{matrix[i, j]}")
                
                conf_str = f" | Confused with: {', '.join(confusions)}" if confusions else ""
                
                if gt_token == BG_TOKEN:
                    print(f"  '{gt_token}' (Background -> False Positives): {row_sum} total{conf_str}")
                else:
                    print(f"  '{gt_token}': {correct}/{row_sum} ({accuracy:.1f}%){conf_str}")
        
        if total_count > 0:
            overall_acc = total_correct / total_count * 100
            print(f"\n  Overall (excluding background): {total_correct}/{total_count} ({overall_acc:.1f}%)")
        
        # Print insertion/deletion summary
        if BG_TOKEN in labels:
            bg_idx = labels.index(BG_TOKEN)
            
            # Deletions: characters predicted as background (column <BG>, excluding row <BG>)
            deletions = sum(matrix[i, bg_idx] for i in range(len(labels)) if i != bg_idx)
            
            # Insertions: background predicted as characters (row <BG>, excluding column <BG>)
            insertions = sum(matrix[bg_idx, j] for j in range(len(labels)) if j != bg_idx)
            
            print(f"\n  Background Statistics:")
            print(f"    Deletions (char -> background): {deletions}")
            print(f"    Insertions (background -> char): {insertions}")


def plot_combined_matrices(cms: list, output_dir: Path, normalized: bool = False):
    """
    Plot combined confusion matrices for all methods in a single figure.
    
    Args:
        cms: List of tuples (ConfusionMatrixBuilder, title)
        output_dir: Directory to save the output
        normalized: If True, plot normalized matrices
    """
    n_matrices = len(cms)
    fig, axes = plt.subplots(1, n_matrices, figsize=(10 * n_matrices, 10))
    
    if n_matrices == 1:
        axes = [axes]
    
    # Use custom white-green colormap
    cmap = create_white_green_cmap()
    
    for idx, (cm, title) in enumerate(cms):
        ax = axes[idx]
        
        if normalized:
            matrix, gt_labels, pred_labels = cm.build_normalized_matrix()
            fmt = '.2f'
            title_suffix = "(Normalized)"
            cbar_label = 'Proportion'
            vmin, vmax = 0, 1
        else:
            matrix, gt_labels, pred_labels = cm.build_matrix()
            fmt = 'd'
            title_suffix = "(Count)"
            cbar_label = 'Count'
            vmin, vmax = None, None
        
        if len(gt_labels) > 0:
            sns.heatmap(
                matrix,
                annot=True,
                fmt=fmt,
                cmap=cmap,
                xticklabels=pred_labels,
                yticklabels=gt_labels,
                ax=ax,
                cbar_kws={'label': cbar_label},
                vmin=vmin,
                vmax=vmax,
                linewidths=0.5,
                linecolor='lightgray'
            )
            ax.set_xlabel('Predicted Token', fontsize=10)
            ax.set_ylabel('Ground Truth Token', fontsize=10)
            ax.set_title(f'{title}\n{title_suffix}', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            
            # Highlight <BG> row and column
            if BG_TOKEN in gt_labels:
                bg_idx = gt_labels.index(BG_TOKEN)
                ax.axhline(y=bg_idx, color='orange', linewidth=2)
                ax.axhline(y=bg_idx + 1, color='orange', linewidth=2)
                ax.axvline(x=bg_idx, color='orange', linewidth=2)
                ax.axvline(x=bg_idx + 1, color='orange', linewidth=2)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title(f'{title}\n{title_suffix}', fontsize=12)
    
    plt.tight_layout()
    
    suffix = "normalized" if normalized else "count"
    save_path = output_dir / f"confusion_matrix_combined_{suffix}.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    print(f"Combined confusion matrix ({suffix}) saved: {save_path}")
    
    return fig


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
    parser.add_argument(
        "--show-alignment",
        action="store_true",
        help="Show detailed alignment for each image (verbose output)",
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

        print("-" * 60)
        print(f"Image: {image_filename}")
        print(f"  GT:         {gt_string} -> tokens: {gt_tokens}")
        print(f"  HR_Image:   {hr_pred} -> tokens: {hr_pred_tokens}  (acc={hr_acc:.2%})")
        print(f"  Bicubic:    {bicubic_pred} -> tokens: {bicubic_pred_tokens}  (acc={bicubic_acc:.2%})")
        print(f"  SR model:   {sr_pred} -> tokens: {sr_pred_tokens}  (acc={sr_acc:.2%})")
        
        # Show detailed alignment if requested or if there are errors
        if args.show_alignment or (sr_acc < 0.99 and sr_acc > 0):
            print_alignment_details(gt_tokens, sr_pred_tokens, "SR model")

    # --- Final summary ---
    if bicubic_accuracies:
        avg_hr = sum(hr_accuracies) / len(hr_accuracies)
        avg_bicubic = sum(bicubic_accuracies) / len(bicubic_accuracies)
        avg_sr = sum(sr_accuracies) / len(sr_accuracies)

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print("Note: Characters '3' treated as '2' for accuracy calculation")
        print(f"Images evaluated: {len(bicubic_accuracies)}")
        print(f"Average char accuracy (HR): {avg_hr:.2%}")
        print(f"Average char accuracy (Bicubic): {avg_bicubic:.2%}")
        print(f"Average char accuracy (SR):      {avg_sr:.2%}")
        print(f"Perfect plates (acc>0.99) HR: {hr_perfect}")
        print(f"Perfect plates (acc>0.99) Bicubic: {bicubic_perfect}")
        print(f"Perfect plates (acc>0.99) SR:      {sr_perfect}")
        print("=" * 60)
        
        # --- Generate and save confusion matrices ---
        print("\n" + "=" * 60)
        print("GENERATING CONFUSION MATRICES")
        print("=" * 60)
        
        # Save individual confusion matrix images (both count and normalized)
        # HR matrices
        cm_hr.plot_and_save(
            str(output_dir / "confusion_matrix_hr_count.png"), 
            normalized=False
        )
        cm_hr.plot_and_save(
            str(output_dir / "confusion_matrix_hr_normalized.png"), 
            normalized=True
        )
        
        # Bicubic matrices
        cm_bicubic.plot_and_save(
            str(output_dir / "confusion_matrix_bicubic_count.png"), 
            normalized=False
        )
        cm_bicubic.plot_and_save(
            str(output_dir / "confusion_matrix_bicubic_normalized.png"), 
            normalized=True
        )
        
        # SR matrices
        cm_sr.plot_and_save(
            str(output_dir / "confusion_matrix_sr_count.png"), 
            normalized=False
        )
        cm_sr.plot_and_save(
            str(output_dir / "confusion_matrix_sr_normalized.png"), 
            normalized=True
        )
        
        # Print per-token summaries
        cm_hr.print_summary()
        cm_bicubic.print_summary()
        cm_sr.print_summary()
        
        # --- Create combined figures ---
        print("\n" + "=" * 60)
        print("GENERATING COMBINED CONFUSION MATRICES")
        print("=" * 60)
        
        cms_list = [
            (cm_hr, "HR (High Resolution)"),
            (cm_bicubic, "Bicubic Upsampling"),
            (cm_sr, "Super Resolution")
        ]
        
        # Combined count matrix
        fig_count = plot_combined_matrices(cms_list, output_dir, normalized=False)
        
        # Combined normalized matrix
        fig_normalized = plot_combined_matrices(cms_list, output_dir, normalized=True)
        
        # Display the plots
        print("\n" + "=" * 60)
        print("DISPLAYING CONFUSION MATRICES")
        print("=" * 60)
        
        plt.show()
        
    else:
        print("\nNo images were evaluated.")
        

if __name__ == "__main__":
    main()