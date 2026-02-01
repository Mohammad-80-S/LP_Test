"""
Given a folder containing images and matching XML files
  - HR/  : high-resolution images
  - LR/  : generated low-resolution images
  - XML/ : file names matching the image base name
Only images that have an XML with the same base name are processed.
"""

import argparse
import shutil
from pathlib import Path
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize dataset into HR, LR, XML folders (optional LR generation)."
    )
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        required=True,
        help="Root folder containing images and XML files"
    )
    parser.add_argument(
        "--generate-lr",
        action="store_true",
        help="If set, generate LR images from HR via bicubic downsampling"
    )
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=4,
        help="Downscale factor for LR when --generate-lr is used (default: 4)"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them (default: move)"
    )
    args = parser.parse_args()

    root = Path(args.data_dir)
    if not root.is_dir():
        raise RuntimeError(f"Data directory does not exist: {root}")

    hr_dir = root / "HR"
    lr_dir = root / "LR"
    xml_dir = root / "XML"

    hr_dir.mkdir(exist_ok=True)
    lr_dir.mkdir(exist_ok=True)
    xml_dir.mkdir(exist_ok=True)

    move_or_copy = shutil.copy2 if args.copy else shutil.move

    processed = 0

    for item in root.iterdir():
        if not item.is_file():
            continue

        if item.suffix.lower() not in IMG_EXTS:
            continue

        img_path = item
        base = img_path.stem
        xml_path = root / f"{base}.xml"

        if not xml_path.exists():
            print(f"Warning: No XML found for image '{img_path.name}', skipping.")
            continue

        dest_hr_img = hr_dir / img_path.name
        move_or_copy(str(img_path), str(dest_hr_img))

        dest_xml = xml_dir / xml_path.name
        move_or_copy(str(xml_path), str(dest_xml))

        if args.generate_lr:
            hr_img = Image.open(dest_hr_img).convert("RGB")
            w, h = hr_img.size
            new_w = max(1, w // args.scale_factor)
            new_h = max(1, h // args.scale_factor)
            lr_img = hr_img.resize((new_w, new_h), resample=Image.BICUBIC)
            dest_lr_img = lr_dir / img_path.name
            lr_img.save(dest_lr_img)
        else:
            # If you want LR to contain a copy of HR instead, uncomment:
            # shutil.copy2(dest_hr_img, lr_dir / img_path.name)
            pass

        processed += 1
        print(f"Processed: {base}")

    print(f"\nDone. Total image+XML pairs processed: {processed}")
    print(f"HR dir:  {hr_dir}")
    print(f"LR dir:  {lr_dir}")
    print(f"XML dir: {xml_dir}")


if __name__ == "__main__":
    main()