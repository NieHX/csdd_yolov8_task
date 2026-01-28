import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from config import CLASSES, SEG_RAW_DATASET, SEG_YAML, ensure_configs
from utils import print_seg_metrics

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
MASK_VALUE_TO_CLASS = {1: 0, 2: 1, 3: 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 segmentation on CSDD.")
    parser.add_argument("--weights", default="yolov8l-seg.pt", help="Trained weights path.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--save-json", action="store_true", help="Save COCO JSON (can be large).")
    parser.add_argument("--plots", action="store_true", help="Save PR curves and plots.")
    parser.add_argument("--miou", action="store_true", help="Compute pixel mIoU using raw masks.")
    parser.add_argument("--miou-only", action="store_true", help="Skip val metrics and only compute mIoU.")
    parser.add_argument("--miou-device", default=None, help="Device for mIoU inference (default: same as --device).")
    parser.add_argument("--miou-imgsz", type=int, default=None, help="Override image size for mIoU.")
    parser.add_argument("--miou-half", action="store_true", help="Use FP16 for mIoU (CUDA only).")
    parser.add_argument("--miou-batch", type=int, default=1, help="Batch size for mIoU inference.")
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


def compute_miou(
    model: YOLO,
    img_dir: Path,
    mask_dir: Path,
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
    half: bool,
    batch: int,
):
    image_paths = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not image_paths:
        raise FileNotFoundError("No images found for mIoU evaluation.")

    intersections = np.zeros(len(CLASSES), dtype=np.float64)
    unions = np.zeros(len(CLASSES), dtype=np.float64)

    batch = max(int(batch), 1)
    pbar = tqdm(total=len(image_paths), desc="mIoU")
    for start in range(0, len(image_paths), batch):
        chunk = image_paths[start : start + batch]
        results = model.predict(
            source=[str(p) for p in chunk],
            imgsz=imgsz,
            device=device,
            conf=conf,
            iou=iou,
            half=half,
            batch=batch,
            verbose=False,
        )
        for result in results:
            img_path = Path(result.path)
            mask_path = find_mask(img_path.stem, mask_dir)
            if mask_path is None:
                pbar.update(1)
                continue

            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                pbar.update(1)
                continue
            if gt_mask.ndim == 3:
                if gt_mask.shape[2] == 4:
                    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGRA2GRAY)
                elif gt_mask.shape[2] == 3:
                    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
                else:
                    gt_mask = gt_mask[:, :, 0]

            height, width = gt_mask.shape
            pred_masks = np.zeros((len(CLASSES), height, width), dtype=bool)

            if result.masks is not None and result.boxes is not None:
                masks = result.masks.data
                if masks is not None and len(masks) > 0:
                    masks_np = masks.cpu().numpy()
                    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                    for mask, cls_id in zip(masks_np, cls_ids):
                        if cls_id < 0 or cls_id >= len(CLASSES):
                            continue
                        if mask.shape != (height, width):
                            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                        pred_masks[cls_id] |= mask > 0.5

            for value, cls_id in MASK_VALUE_TO_CLASS.items():
                gt = gt_mask == value
                pred = pred_masks[cls_id]
                inter = np.logical_and(gt, pred).sum()
                union = np.logical_or(gt, pred).sum()
                intersections[cls_id] += inter
                unions[cls_id] += union
            pbar.update(1)
    pbar.close()

    per_class_iou = []
    for cls_id, (inter, union) in enumerate(zip(intersections, unions)):
        if union == 0:
            iou_val = float("nan")
        else:
            iou_val = float(inter / union)
        per_class_iou.append(iou_val)

    miou = float(np.nanmean(per_class_iou)) if per_class_iou else float("nan")
    return per_class_iou, miou


def main() -> None:
    args = parse_args()
    if args.miou_only:
        args.miou = True
    ensure_configs()
    model = YOLO(args.weights)
    if not args.miou_only:
        metrics = model.val(
            data=str(SEG_YAML),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            plots=args.plots,
            save_json=args.save_json,
        )
        print_seg_metrics(metrics, CLASSES)
        del metrics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if args.miou:
        miou_device = args.miou_device or args.device
        miou_imgsz = args.miou_imgsz or args.imgsz
        miou_half = args.miou_half and str(miou_device).lower() not in ("cpu", "mps")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        miou_model = model
        if str(miou_device).lower() == "cpu":
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            miou_model = YOLO(args.weights)

        img_dir = SEG_RAW_DATASET / "img" / args.split
        mask_dir = SEG_RAW_DATASET / "ground_truth" / args.split
        if not img_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError("Raw CSDD_seg images or masks not found for mIoU.")
        per_class_iou, miou = compute_miou(
            miou_model,
            img_dir,
            mask_dir,
            miou_imgsz,
            miou_device,
            args.conf,
            args.iou,
            miou_half,
            args.miou_batch,
        )
        print("mIoU (pixel)")
        for name, iou_val in zip(CLASSES, per_class_iou):
            label = f"{iou_val:.4f}" if not np.isnan(iou_val) else "nan"
            print(f"{name} IoU: {label}")
        print(f"mIoU: {miou:.4f}" if not np.isnan(miou) else "mIoU: nan")


if __name__ == "__main__":
    main()
