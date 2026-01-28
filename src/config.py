from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATASET_ROOT = Path(r"your dataset path")
DET_DATASET = DATASET_ROOT / "CSDD_det"
SEG_DATASET = DATASET_ROOT / "CSDD_seg_yolo"
SEG_RAW_DATASET = DATASET_ROOT / "CSDD_seg"

CLASSES = ["Scratch", "Spot", "Rust"]

CONFIG_DIR = PROJECT_DIR / "configs"
DET_YAML = CONFIG_DIR / "csdd_det.yaml"
SEG_YAML = CONFIG_DIR / "csdd_seg.yaml"


def _yaml_text(dataset_path: Path, train: str, val: str, test: str) -> str:
    return (
        f"path: {dataset_path.as_posix()}\n"
        f"train: {train}\n"
        f"val: {val}\n"
        f"test: {test}\n"
        "names:\n"
        "  0: Scratch\n"
        "  1: Spot\n"
        "  2: Rust\n"
    )


def ensure_configs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not DET_YAML.exists():
        DET_YAML.write_text(
            _yaml_text(DET_DATASET, "images/train2017", "images/val2017", "images/test2017"),
            encoding="utf-8",
        )
    if not SEG_YAML.exists():
        SEG_YAML.write_text(
            _yaml_text(SEG_DATASET, "images/train", "images/val", "images/test"),
            encoding="utf-8",
        )
