"""
Module 1b & 2 – ROI Crop & Auto-Mask Generation
AIDA 2158A-F: Strawberry Harvesting Pipeline

This script automates TWO steps:
1. Module 1b: Crop the largest ripe strawberry ROI from each image.
2. Module 2: Automatically extract the "peduncle" (stem) masks from the
   existing YOLO dataset for these crops.

This eliminates the need for manual annotation in Digital Sreeni.

Output: 
  - roi_crops/  -> JPEG/PNG images
  - roi_masks/  -> Binary masks (White=peduncle, Black=background)
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT      = Path(__file__).parent
DATASET   = ROOT / "dataset"
ROI_DIR   = ROOT / "roi_crops"
MASK_DIR  = ROOT / "roi_masks"
BEST_PT   = ROOT / "runs" / "segment" / "yolo11s_strawberry" / "weights" / "best.pt"

# Padding applied around the strawberry mask bounding box (fraction of bbox size)
PAD_FRAC  = 0.25
# How many ROI crops to write out
MAX_CROPS = 100


# ─────────────────────────────────────────────────────────────────────────────
# Parsing YOLO Labels
# ─────────────────────────────────────────────────────────────────────────────
def get_label_path(img_path: Path) -> Path:
    """Find the corresponding label .txt file for an image."""
    lbl_path = img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name
    return lbl_path


def parse_yolo_polygons(lbl_path: Path, img_w: int, img_h: int):
    """
    Parses a YOLO segmentation .txt file.
    Returns list of dicts: {'cls': int, 'poly': np.ndarray(None, 2)}
    """
    if not lbl_path.exists():
        return []
    
    results = []
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3: continue
            cls_id = int(parts[0])
            # Coords are normalized x, y, x, y ...
            coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
            coords[:, 0] *= img_w
            coords[:, 1] *= img_h
            results.append({'cls': cls_id, 'poly': coords.astype(np.int32)})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Ripeness heuristic: score how "red" a masked region is
# ─────────────────────────────────────────────────────────────────────────────
def redness_score(image_bgr: np.ndarray, mask: np.ndarray) -> float:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    red_lo = cv2.inRange(hsv, (0,   60, 60), (10,  255, 255))
    red_hi = cv2.inRange(hsv, (160, 60, 60), (180, 255, 255))
    red    = cv2.bitwise_or(red_lo, red_hi)
    mask_bool = mask.astype(bool)
    if mask_bool.sum() == 0: return 0.0
    return float(red[mask_bool].sum()) / mask_bool.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Selection & Generation
# ─────────────────────────────────────────────────────────────────────────────
def select_target(result, image_bgr: np.ndarray):
    if result.masks is None: return None, None
    masks  = result.masks.data.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    h, w = image_bgr.shape[:2]
    best_mask = None
    best_area = 0
    best_red  = -1.0
    for i, cls_id in enumerate(cls_ids):
        if cls_id != 1: continue # only strawberry
        raw = masks[i]
        bin_mask = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
        bin_mask = (bin_mask > 0.5).astype(np.uint8)
        area = int(bin_mask.sum())
        red = redness_score(image_bgr, bin_mask)
        if red > best_red or (red == best_red and area > best_area):
            best_mask, best_area, best_red = bin_mask, area, red
    return best_mask, best_area


def all_images(dataset_dir: Path) -> list[Path]:
    imgs = []
    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split / "images"
        if split_dir.exists():
            imgs.extend(sorted([*split_dir.glob("*.png"), *split_dir.glob("*.jpg")]))
    return imgs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    from ultralytics import YOLO
    print("\n══════════════════════════════════════════")
    print("  Module 1b & 2: ROI & Auto-Masking")
    print("══════════════════════════════════════════\n")

    if not BEST_PT.exists():
        print(f"❌  YOLO weights not found at: {BEST_PT}. Run Module 1 first.")
        sys.exit(1)

    ROI_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(BEST_PT))

    images = all_images(DATASET)
    print(f"  Found {len(images)} images in dataset")
    
    saved = 0
    for img_path in tqdm(images, desc="Generating ROIs and Masks"):
        if saved >= MAX_CROPS: break
        
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]

        results = model(img, verbose=False)
        mask_strawberry, area = select_target(results[0], img)
        if mask_strawberry is None: continue

        # Calculate Crop Box
        ys, xs = np.where(mask_strawberry > 0)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        pad_x, pad_y = max(int((x2-x1)*PAD_FRAC), 20), max(int((y2-y1)*PAD_FRAC), 20)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

        # 1. Save ROI Crop
        roi_img = img[y1:y2, x1:x2]
        cv2.imwrite(str(ROI_DIR / img_path.name), roi_img)

        # 2. Generate and Save Stem Mask (Module 2 automation)
        lbl_path = get_label_path(img_path)
        polygons = parse_yolo_polygons(lbl_path, w, h)
        
        # Create a blank mask for the crop
        roi_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
        
        for poly_data in polygons:
            if poly_data['cls'] == 0: # 0 = peduncle/stem
                # Shift coordinates to crop space
                poly_local = poly_data['poly'].copy()
                poly_local[:, 0] -= x1
                poly_local[:, 1] -= y1
                # Draw on ROI mask
                cv2.fillPoly(roi_mask, [poly_local], 255)

        cv2.imwrite(str(MASK_DIR / img_path.name), roi_mask)
        saved += 1

    print(f"\n✅  Done! Saved {saved} crops and masks.")
    print(f"    Images: {ROI_DIR}")
    print(f"    Masks : {MASK_DIR}")


if __name__ == "__main__":
    main()
