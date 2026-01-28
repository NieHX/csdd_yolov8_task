import argparse
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from config import SEG_DATASET, SEG_RAW_DATASET

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
CLASS_MAPPING = {1: 0, 2: 1, 3: 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CSDD_seg masks to YOLOv8 segmentation labels.")
    parser.add_argument("--raw-root", default=str(SEG_RAW_DATASET))
    parser.add_argument("--out-root", default=str(SEG_DATASET))
    parser.add_argument("--force", action="store_true", help="Overwrite output folder if it exists.")
    return parser.parse_args()


def find_mask(stem: str, mask_dir: Path) -> Optional[Path]:
    candidates = []
    for ext in IMAGE_EXTS:
        candidates.append(mask_dir / f"{stem}{ext}")
        candidates.append(mask_dir / f"{stem}_mask{ext}")
        candidates.append(mask_dir / f"{stem}_label{ext}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def contours_from_binary(binary: np.ndarray):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def convert_split(split: str, raw_root: Path, out_root: Path) -> int:
    img_src_dir = raw_root / "img" / split
    mask_src_dir = raw_root / "ground_truth" / split

    if not img_src_dir.exists() or not mask_src_dir.exists():
        raise FileNotFoundError(f"Missing data for split: {split}")

    img_dst_dir = out_root / "images" / split
    label_dst_dir = out_root / "labels" / split

    img_dst_dir.mkdir(parents=True, exist_ok=True)
    label_dst_dir.mkdir(parents=True, exist_ok=True)

    image_files = [p for p in img_src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    converted = 0

    for img_path in tqdm(sorted(image_files), desc=f"{split}"):
        shutil.copy2(img_path, img_dst_dir / img_path.name)
        mask_path = find_mask(img_path.stem, mask_src_dir)
        if mask_path is None:
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        height, width = mask.shape
        lines = []
        for value in np.unique(mask):
            if value == 0:
                continue
            class_id = CLASS_MAPPING.get(int(value))
            if class_id is None:
                continue

            binary = np.where(mask == value, 255, 0).astype(np.uint8)
            for contour in contours_from_binary(binary):
                if cv2.contourArea(contour) < 10:
                    continue
                contour = contour.reshape(-1, 2)
                coords = []
                for x, y in contour:
                    coords.append(f"{x / width:.6f}")
                    coords.append(f"{y / height:.6f}")
                if len(coords) >= 6:
                    lines.append(f"{class_id} " + " ".join(coords))

        if lines:
            (label_dst_dir / f"{img_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            converted += 1

    return converted


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    if out_root.exists() and args.force:
        shutil.rmtree(out_root)

    total = 0
    for split in ("train", "val", "test"):
        total += convert_split(split, raw_root, out_root)

    print(f"Converted labels: {total}")


if __name__ == "__main__":
    main()
