import argparse
from pathlib import Path

from ultralytics import YOLO

from utils import list_images, sample_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 prediction and save visualizations.")
    parser.add_argument("--task", default="det", choices=["det", "seg"])
    parser.add_argument("--weights", required=True, help="Trained weights path.")
    parser.add_argument("--source", required=True, help="Image file or directory.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/csdd_predict")
    parser.add_argument("--name", default="predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = list_images(Path(args.source))
    if not images:
        raise FileNotFoundError("No images found for prediction.")

    images = sample_images(images, args.limit, args.seed)
    model = YOLO(args.weights)
    model.predict(
        source=[str(p) for p in images],
        save=True,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        show_labels=True,
        show_conf=True,
        retina_masks=True,
    )


if __name__ == "__main__":
    main()
