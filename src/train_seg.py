import argparse
from ultralytics import YOLO

from config import SEG_YAML, ensure_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on CSDD segmentation.")
    parser.add_argument("--model", default="yolov8l-seg.pt", help="Base model or weights.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1, help="Lower this if you hit CUDA OOM.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", action="store_true", help="Cache images to RAM.")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    parser.add_argument("--no-val", action="store_true", help="Skip validation to save memory/time.")
    parser.add_argument("--project", default="runs/csdd_seg")
    parser.add_argument("--name", default="yolov8_seg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_configs()
    model = YOLO(args.model)
    model.train(
        data=str(SEG_YAML),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        workers=args.workers,
        cache=args.cache,
        amp=not args.no_amp,
        val=not args.no_val,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
