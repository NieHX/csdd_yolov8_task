import argparse
from ultralytics import YOLO

from config import CLASSES, DET_YAML, ensure_configs
from utils import print_det_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 detection on CSDD.")
    parser.add_argument("--weights", default="yolov8l.pt", help="Trained weights path.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_configs()
    model = YOLO(args.weights)
    metrics = model.val(
        data=str(DET_YAML),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        plots=True,
        save_json=True,
    )
    print_det_metrics(metrics, CLASSES)


if __name__ == "__main__":
    main()
