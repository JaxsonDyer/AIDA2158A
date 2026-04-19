"""
Module 1 – YOLOv11-seg Training
AIDA 2158A-F: Strawberry Harvesting Pipeline

Steps:
  1. Merge the three sub-datasets (herve / kelsey / mark) into dataset/
  2. Auto-split into train / val / test (80 / 15 / 5)
  3. Write a unified data.yaml
  4. Train yolo11s-seg.pt on the merged dataset
  5. Save training curves to runs/segment/
"""

import os
import shutil
import random
from pathlib import Path

import yaml
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DB_ROOT    = ROOT / "strawberry_db"
DATASET    = ROOT / "dataset"
SPLITS     = {"train": 0.80, "val": 0.15, "test": 0.05}

# Canonical class ordering (index 0 = peduncle, index 1 = strawberry)
CLASS_NAMES = ["peduncle", "strawberry"]

RANDOM_SEED = 42
EPOCHS      = 10
IMG_SIZE    = 640
BATCH_SIZE  = 8   # lower if OOM; increase if you have a big GPU


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Collect all images + labels from every contributor folder
# ─────────────────────────────────────────────────────────────────────────────
def collect_samples(db_root: Path) -> list[tuple[Path, Path]]:
    """Return list of (image_path, label_path) pairs from all sub-datasets."""
    samples = []
    for contributor in sorted(db_root.iterdir()):
        if not contributor.is_dir():
            continue
        img_dir = contributor / "train" / "images"
        lbl_dir = contributor / "train" / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            print(f"  ⚠  Skipping {contributor.name}: missing images or labels folder")
            continue
        for img_path in sorted([*img_dir.glob("*.png"), *img_dir.glob("*.jpg")]):
            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            if lbl_path.exists():
                samples.append((img_path, lbl_path))
            else:
                print(f"  ⚠  No label for {img_path.name}, skipping")
    print(f"  ✓  Collected {len(samples)} annotated samples from {db_root}")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Split samples and copy files into dataset/
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(samples: list[tuple[Path, Path]], dataset_dir: Path):
    """Copy images & labels into train / val / test splits."""
    random.seed(RANDOM_SEED)
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * SPLITS["train"])
    n_val   = int(n * SPLITS["val"])
    # rest goes to test

    split_map = {
        "train": samples[:n_train],
        "val":   samples[n_train : n_train + n_val],
        "test":  samples[n_train + n_val :],
    }

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    for split, split_samples in split_map.items():
        img_out = dataset_dir / split / "images"
        lbl_out = dataset_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_src, lbl_src in split_samples:
            shutil.copy2(img_src, img_out / img_src.name)
            shutil.copy2(lbl_src, lbl_out / lbl_src.name)

        print(f"  ✓  {split:5s}: {len(split_samples):3d} samples → {img_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Write data.yaml
# ─────────────────────────────────────────────────────────────────────────────
def write_data_yaml(dataset_dir: Path, class_names: list[str]) -> Path:
    yaml_path = dataset_dir / "data.yaml"
    config = {
        "path":  str(dataset_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    len(class_names),
        "names": class_names,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  ✓  data.yaml written to {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Train YOLOv11-seg
# ─────────────────────────────────────────────────────────────────────────────
def train_yolo(yaml_path: Path):
    import torch
    if torch.cuda.is_available():
        device = "0"          # CUDA GPU 0
    elif torch.backends.mps.is_available():
        device = "mps"        # Apple Silicon GPU
    else:
        device = "cpu"
    print(f"  ✓  Using device: {device}")

    model = YOLO("yolo11n-seg.pt")  # downloads weights on first run
    results = model.train(
        data    = str(yaml_path),
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH_SIZE,
        device  = device,
        project = str(ROOT / "runs" / "segment"),
        name    = "yolo11s_strawberry",
        exist_ok= True,
        verbose = True,
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Plot & save training curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(run_dir: Path):
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        print(f"  ⚠  results.csv not found in {run_dir}")
        return

    import csv
    rows = []
    with open(results_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): float(v) for k, v in row.items() if v.strip()})

    epochs = [r["epoch"] for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("YOLOv11-seg Training – Strawberry Dataset", fontsize=14, fontweight="bold")

    metrics = [
        ("train/seg_loss",   "Train Seg Loss",    axes[0, 0]),
        ("val/seg_loss",     "Val Seg Loss",      axes[0, 1]),
        ("train/box_loss",   "Train Box Loss",    axes[0, 2]),
        ("metrics/mAP50(B)", "mAP@0.5 (box)",    axes[1, 0]),
        ("metrics/mAP50(M)", "mAP@0.5 (mask)",   axes[1, 1]),
        ("metrics/precision(B)", "Precision",     axes[1, 2]),
    ]

    for key, title, ax in metrics:
        if key in rows[0]:
            ax.plot(epochs, [r[key] for r in rows], linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    out_path = run_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓  Training curves saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n══════════════════════════════════════════")
    print("  Module 1: YOLOv11-seg Training")
    print("══════════════════════════════════════════\n")

    print("[ Step 1 ] Collecting samples ...")
    samples = collect_samples(DB_ROOT)

    print("\n[ Step 2 ] Building merged dataset ...")
    build_dataset(samples, DATASET)

    print("\n[ Step 3 ] Writing data.yaml ...")
    yaml_path = write_data_yaml(DATASET, CLASS_NAMES)

    print("\n[ Step 4 ] Training YOLOv11s-seg ...")
    train_yolo(yaml_path)

    # Find the latest run directory
    run_dir = ROOT / "runs" / "segment" / "yolo11s_strawberry"
    print("\n[ Step 5 ] Saving training curves ...")
    plot_results(run_dir)

    print("\n✅  Module 1 complete!")
    print(f"    Weights: {run_dir / 'weights' / 'best.pt'}")
    print(f"    Plots  : {run_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
