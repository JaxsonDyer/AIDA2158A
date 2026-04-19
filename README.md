# Strawberry Harvesting Deep Learning Pipeline
**AIDA 2158A-F — Neural Networks and Deep Learning — Red Deer Polytechnic**
**Instructor:** Dr. M. Tufail

A fully automated deep learning pipeline for autonomous strawberry harvesting. The system goes from raw RGB images to precise robotic cut-line placement on peduncles — identifying which strawberries are ripe, finding their stems, and computing the exact angle and position for a gripper to snip.

---

## Pipeline Overview

```
Raw Images → YOLO Detection → Target Selection → ROI Crops + Auto-Masks → U-Net Segmentation → Cut Line Extraction
```

| Step | Module | What It Does |
|:-----|:-------|:-------------|
| 1 | **YOLOv11-seg Training** | Trains a YOLOv11-seg model to detect strawberry instances and peduncles in full-size images. |
| 1b | **ROI Crop & Auto-Masking** | Selects the ripest, largest strawberry per image using an HSV redness heuristic, crops a zoomed-in Region of Interest with 25% padding, and extracts binary peduncle masks from existing YOLO labels — eliminating manual annotation. |
| 3 | **U-Net Training** | Trains a custom U-Net on the ROI crops and masks to segment crown/stem/peduncle pixels at high resolution. Uses BCE + Dice loss for robust thin-structure segmentation. |
| 4 | **Cut Line Extraction** | Runs YOLO on original full-size images to detect all ripe strawberries and their peduncles, matches each strawberry to its nearest peduncle, skeletonizes the peduncle mask, finds the cut point at 1/3 from the stem tip, and draws a perpendicular cut line for robotic gripper alignment. |

---

## Setup & Execution

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# 1. Install all required libraries
~/.local/bin/uv add ultralytics torch torchvision opencv-python numpy pyyaml \
    matplotlib scikit-learn Pillow tqdm scipy scikit-image

# 2. Run the full pipeline start-to-finish
~/.local/bin/uv run python main.py --all
```

### Run Individual Modules

```bash
~/.local/bin/uv run python main.py --module 1      # Merge dataset + train YOLOv11-seg
~/.local/bin/uv run python main.py --module 1b     # Generate ROI crops AND masks
~/.local/bin/uv run python main.py --module 3      # Train U-Net
~/.local/bin/uv run python main.py --module 4      # Extract peduncle cut lines
```

> **Hardware Support:** The pipeline automatically detects and utilizes **Apple Silicon (MPS)**, **NVIDIA (CUDA)**, or **CPU**.

---

## Module Details

### Module 1 — YOLOv11-seg Training

- Splits into 80% train / 15% val / 5% test with seed 42.
- Trains `yolo11n-seg.pt` for 10 epochs at 640×640 with batch size 8.
- Two classes: `peduncle` (class 0) and `strawberry` (class 1).
- Outputs training curves (`training_curves.png`) and best weights (`best.pt`).

### Module 1b — ROI Crop & Auto-Masking

- Loads the trained YOLO model and runs inference on all dataset images.
- For each image, selects the **reddest** strawberry using an HSV redness heuristic (breaks ties by area).
- Crops a Region of Interest around the target strawberry with 25% padding (min 20px).
- Cross-references the crop coordinates with the original YOLO label files to extract peduncle polygons into the crop space, generating binary masks automatically — **no manual annotation required**.
- Outputs 100 matched pairs of ROI crops + masks.

### Module 3 — U-Net Training

- **Architecture:** Classic U-Net with 4-level encoder (64→128→256→512), bottleneck (1024), and symmetric decoder with skip connections. Each block uses double Conv+BN+ReLU.
- **Loss:** BCE + Dice (handles class imbalance for thin peduncle structures).
- **Training:** 50 epochs, AdamW optimizer with cosine annealing LR schedule, 256×256 input resolution.
- **Augmentations:** Random horizontal and vertical flips during training.
- **Validation:** 15% held out, best model selected by IoU.
- Outputs `unet_model.pth`, loss/IoU curves, and side-by-side prediction grids.

### Module 4 — Peduncle Cut Line Extraction

This module operates directly on the **original full-size images** (not the ROI crops), drawing cut lines in context:

1. **YOLO Detection:** Detects all strawberry and peduncle instances in the image.
2. **Ripeness Filtering:** Filters strawberries by HSV redness score (minimum 0.10) to identify ripe targets.
3. **Strawberry–Peduncle Matching:** Each ripe strawberry is matched to its nearest peduncle by centroid distance (max 300px), with priority given to redder strawberries.
4. **Skeletonization:** Each peduncle mask is thinned to a 1-pixel centerline via `skimage.morphology.skeletonize`, then ordered into a contiguous path using nearest-neighbor chaining.
5. **Cut Point Selection:** The skeleton is oriented so the end **farthest from the strawberry** is the "top" (stem tip). The cut point is placed at **1/3 of the arc length** from the top — well on the peduncle, away from the fruit body.
6. **Local Tangent Estimation:** PCA is applied to a window of neighboring skeleton points around the cut point to compute the local peduncle direction.
7. **Perpendicular Cut Line:** A red line is drawn perpendicular to the peduncle at the cut point, with a yellow circle marking the exact location and the angle displayed in degrees from vertical.
8. **Fallback:** If skeletonization fails (very small or noisy masks), a weighted PCA fallback is used that biases toward the peduncle tip.

**Outputs per image:**
- Full-size overlay with green peduncle highlight, red cut line, yellow cut-point circle, and angle label
- `angles.csv` with per-image cut angles (semicolon-separated for multiple strawberries)
- `cut_angle_histogram.png` showing the distribution of cut angles across all images

---

## Project Structure

```
.
├── main.py                    # Unified CLI controller
├── module1_yolo_train.py      # Dataset merge + YOLOv11-seg training
├── module1_roi_crop.py        # Target selection, ROI cropping, auto-masking
├── module3_unet_train.py      # Custom U-Net architecture + training loop
├── module4_stem_angle.py      # Peduncle cut line extraction
├── project.md                 # Original project specification
├── pyproject.toml             # Dependencies (uv)
├── yolo11n-seg.pt             # YOLO pretrained weights
├── yolo11s-seg.pt             # YOLO pretrained weights
├── unet_model.pth             # Trained U-Net weights
├── strawberry_db/             # Raw annotated data (herve / kelsey / mark)
├── dataset/                   # Merged + split dataset (train/val/test)
│   └── data.yaml
├── roi_crops/                 # 100 zoomed-in ROI images
├── roi_masks/                 # 100 binary peduncle masks
└── runs/
    ├── segment/yolo11s_strawberry/  # YOLO weights, curves, val plots
    ├── unet/                        # U-Net loss/IoU curves, predictions
    └── cut_lines/                   # Cut line overlays, angles.csv, histogram
```

---

## Output Directories

| Directory | Content |
| :--- | :--- |
| `runs/segment/yolo11s_strawberry/` | YOLO weights (`best.pt`, `last.pt`), training curves, confusion matrix, validation predictions. |
| `roi_crops/` | 100 high-resolution zoomed-in images around the ripest strawberry. |
| `roi_masks/` | 100 binary masks (white = peduncle, black = background) auto-extracted from YOLO labels. |
| `runs/unet/` | U-Net model, BCE+Dice loss curves, validation IoU curve, side-by-side prediction grid. |
| `runs/cut_lines/` | Full-size overlay images with cut lines, `angles.csv`, and `cut_angle_histogram.png`. |

---

## Key Design Decisions

| Decision | Rationale |
|:---------|:----------|
| **Auto-masking instead of manual annotation** | The YOLO labels already contain peduncle polygons. Re-projecting them into crop space produces high-quality binary masks without the labor of the Digital Sreeni annotation tool. |
| **HSV redness heuristic for target selection** | Simple and effective — ripe strawberries are red, and thresholding in HSV space is robust to lighting variation compared to RGB. |
| **BCE + Dice loss for U-Net** | Peduncle pixels are a small fraction of the image. Dice loss handles class imbalance; BCE provides stable gradients for thin structures. |
| **Skeleton-based cut point instead of centroid** | The peduncle centroid can fall on the strawberry body. Skeletonization + directional ordering ensures the cut point is on the actual stem. |
| **1/3 from stem tip for cut point** | Places the cut well away from both the strawberry body and the branching point, giving the gripper a clean, graspable segment. |
| **Full-size image output for Module 4** | Cut lines are drawn on the original images so results are immediately interpretable in context — not on tiny crops. |
| **Strawberry→peduncle matching by proximity** | The nearest peduncle to a ripe strawberry is almost always its own stem, making a simple distance-based matcher reliable. |

---

## Hyperparameters

| Parameter | Value | Module |
|:----------|:------|:-------|
| YOLO model | yolo11n-seg | 1 |
| YOLO epochs | 10 | 1 |
| YOLO image size | 640 | 1 |
| YOLO batch size | 8 | 1 |
| Train/Val/Test split | 80/15/5% | 1 |
| Random seed | 42 | 1 |
| ROI padding | 25% (min 20px) | 1b |
| Max ROI crops | 100 | 1b |
| U-Net features | 64, 128, 256, 512 | 3 |
| U-Net input size | 256×256 | 3 |
| U-Net epochs | 50 | 3 |
| U-Net batch size | 8 | 3 |
| U-Net learning rate | 1e-4 | 3 |
| U-Net optimizer | AdamW (weight decay 1e-4) | 3 |
| U-Net scheduler | CosineAnnealingLR | 3 |
| Loss function | BCE + Dice | 3 |
| U-Net augmentation | Random H-flip + V-flip | 3 |
| Cut point position | 1/3 from stem tip | 4 |
| Matching max distance | 300px centroid distance | 4 |
| Redness threshold | 0.10 | 4 |