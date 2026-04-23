# Colab Training — README

## What to upload to Colab

1. **`colab_train.ipynb`** — the notebook (open in Colab)
2. **`unet_train.py`** — U-Net training script (called from the notebook)
3. **`dataset.zip`** — zip of your local `dataset/` folder
4. **`roi_data.zip`** — zip of your local `roi_crops/` + `roi_masks/` folders (for U-Net)

## Creating the zip files

From your project root:

```bash
# YOLO dataset
cd /Users/jaxsondyer/Documents/AIDA_2158A_F
zip -r colab_train/dataset.zip dataset/

# ROI data (U-Net)
zip -r colab_train/roi_data.zip roi_crops/ roi_masks/
```

## Changing the model / epochs

Edit the CONFIG block at the top of `colab_train.ipynb`:

```python
YOLO_MODEL  = "yolo11s-seg.pt"   # n=speed, s=balanced, m=accuracy, l=best accuracy
YOLO_EPOCHS = 50                  # more epochs = better convergence
YOLO_BATCH  = 16                  # T4 GPU: 16 works at imgsz=640

UNET_EPOCHS = 100                 # was 50 locally
```

### YOLO model sizes (smallest → largest)

| Model | Params | Speed | Accuracy |
|-------|--------|-------|----------|
| `yolo11n-seg.pt` | ~1M | Fastest | Lowest |
| `yolo11s-seg.pt` | ~5M | Fast | Good |
| `yolo11m-seg.pt` | ~12M | Medium | Better |
| `yolo11l-seg.pt` | ~20M | Slow | Best |

## After training

1. The notebook auto-downloads **`yolo_best.pt`** and **`unet_model.pth`**
2. Drop them into your local project root:
   - `yolo_best.pt` → rename or use as your YOLO weights
   - `unet_model.pth` → replaces the existing one

## Using the trained YOLO model locally

In your local code, point to the new weights:

```python
from ultralytics import YOLO
model = YOLO("yolo_best.pt")  # instead of "yolo11n-seg.pt"
```
