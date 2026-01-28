# csdd_Det-and-Seg-mission
This repository implements **Defect Detection** and **Instance Segmentation** on the CSDD (Collective Surface Defect Datasets) using Ultralytics YOLOv8.It covers three specific defect types: Scratch, Spot, and Rust. The project includes scripts for data preparation, model training, performance evaluation (including pixel-level mIoU), and visualization.
## Project Structure
Based on the provided file tree, the project is organized as follows:
```text
User_Workspace/              <-- Root workspace folder
│
├── csdd_yolov8_task/        <-- [Code Repository] Main project folder
│   ├── configs/             # Configuration directory
│   │   ├── csdd_det.yaml    # [Auto-generated] Detection config
│   │   └── csdd_seg.yaml    # [Auto-generated] Segmentation config
│   ├── runs/                # [Output] Training logs, weights, and plots
│   │   ├── csdd_det/        # Detection experiment results
│   │   ├── csdd_seg/        # Segmentation experiment results
│   │   ├── detect/          # Inference outputs (Detection)
│   │   └── segment/         # Inference outputs (Segmentation)
│   ├── src/                 # [Source Code] Core Python scripts
│   │   ├── __pycache__/
│   │   ├── config.py        # Path configuration (Controls dataset linking)
│   │   ├── eval_det.py      # Detection evaluation script
│   │   ├── eval_seg.py      # Segmentation evaluation (Mask mAP + Pixel mIoU)
│   │   ├── predict_vis.py   # Inference & Visualization script
│   │   ├── prepare_seg_yolo.py # Data processing (Mask -> YOLO format)
│   │   ├── train_det.py     # Detection training script
│   │   ├── train_seg.py     # Segmentation training script
│   │   └── utils.py         # Helper functions
│   ├── yolov8l.pt           # Pre-trained weights (Detection)
│   ├── yolov8l-seg.pt       # Pre-trained weights (Segmentation)
│   ├── yolo11n.pt           # Additional weights
│   ├── requirements.txt     # Python dependencies
│   └── README.md            # Project documentation
│
└── datasets/                # [Data Directory] Sibling folder to code
    ├── CSDD_det/            # Detection dataset (Standard YOLO format)
    │   ├── images/          # Images: train2017, val2017, test2017
    │   └── labels/          # Labels: train2017, val2017, test2017
    ├── CSDD_seg/            # Raw Segmentation dataset (Original Source)
    │   ├── img/             # Raw images (train, val, test)
    │   └── ground_truth/    # Raw binary masks
    └── CSDD_seg_yolo/       # [Generated] Converted Segmentation dataset
        ├── images/          # Copied from CSDD_seg
        └── labels/          # Converted polygon labels (.txt)
```
## Installation
### 1. Clone the repository:
```
git clone https://github.com/NieHX/csdd_yolov8_task.git
```
### 2. Install dependencies:
```
pip install -r requirements.txt
```
## Configuration
### Crucial Step: You must update ```src/config.py``` to point to the sibling ```datasets``` folder.
#### 1. Open ```src/config.py```.
#### 2. Update the ```DATASET_ROOT``` logic to look one level up from the project directory.
```
# src/config.py
from pathlib import Path

# Automatically finds 'datasets' in the parent directory
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_DIR.parent / "datasets"
```
The scripts will automatically check ```configs/``` and generate/update the YAML files pointing to this path.
## Usage
Run all commands from the **project root** (```csdd_yolov8_task/```).
### 1. Data Preparation
Convert raw segmentation masks (in ```CSDD_seg```) to YOLO polygon format.
```
# Optional: Use --force to overwrite existing labels
python src/prepare_seg_yolo.py --force
```
### 2. Defect Detection (Object Detection)
**Train**：（You can adjust the batch size according to your own conditions）
```
python src/train_det.py --epochs 100 --imgsz 1024 --batch 4 --device 0
```
**Evaluate**: Calculates mAP and plots PR curves.
```
# Replace <path_to_best.pt> with your actual weight path, e.g., runs/csdd_det/.../best.pt
python src/eval_det.py --weights <path_to_best.pt> --split test
```
### 3. Defect Segmentation (Instance Segmentation)
**Train**：（You can adjust the batch size according to your own conditions）
```
python src/train_seg.py --epochs 100 --imgsz 1024 --batch 4 --device 0
```
**Evaluate**: Calculates standard **Mask mAP** and **Pixel mIoU**. Note: ```imgsz``` is fixed to 1024. If you run out of VRAM, use ```--miou-half``` or ```--miou-device cpu```.
```
# Option 1: Full Evaluation (Validation + mIoU) using Mixed Precision (Half)
python src/eval_seg.py --weights <path_to_best.pt> --split test --miou --batch 1 --miou-half

# Option 2: Full Evaluation on CPU (Save GPU memory)
python src/eval_seg.py --weights <path_to_best.pt> --split test --miou --batch 1 --miou-device cpu --miou-batch 1

# Option 3: Compute Pixel mIoU ONLY (Skip standard val, fastest for checking IoU)
python src/eval_seg.py --weights <path_to_best.pt> --split test --miou-only --miou-half --miou-batch 1
```
### 4. Visualization
Run inference on images and save visual results with bounding boxes or masks.
```
# Visualize Detection
python src/predict_vis.py --task det --weights <path_to_best.pt> --source <image_or_dir> --limit 6

# Visualize Segmentation
python src/predict_vis.py --task seg --weights <path_to_best.pt> --source <image_or_dir> --limit 6
```
## Output
All training and inference results are saved in the ```runs/``` directory by default:
Training：```runs/csdd_det/```or```runs/csdd_seg/```  
Evaluation：```runs/detect/```
