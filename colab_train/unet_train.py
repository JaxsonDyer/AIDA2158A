"""
U-Net Training on ROI Crown/Stem/Peduncle Masks
Copy of module3_unet_train.py — standalone for Colab

Architecture: Classic U-Net (encoder–bottleneck–decoder + skip connections)
Loss:         BCE + Dice (robust for small, thin structures)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─── Paths (Colab) ───────────────────────────────────────────────────────
WORK_DIR   = Path("/content")
ROI_DIR    = WORK_DIR / "roi_crops"
MASK_DIR   = WORK_DIR / "roi_masks"
MODEL_PATH = WORK_DIR / "unet_model.pth"
RUN_DIR    = WORK_DIR / "runs" / "unet"

# ─── Hyperparameters ────────────────────────────────────────────────────
IMG_SIZE   = 256
BATCH_SIZE = 8
EPOCHS     = 100      # increased from 50 for Colab
LR         = 1e-4
VAL_FRAC   = 0.15
SEED       = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────
class ROIDataset(Dataset):
    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)

    def __init__(self, image_paths, mask_paths, augment=False):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.augment     = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img  = cv2.imread(str(self.image_paths[idx]))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        img  = cv2.resize(img,  (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        img_t  = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)

        for c, (m, s) in enumerate(zip(self.MEAN, self.STD)):
            img_t[c] = (img_t[c] - m) / s

        if self.augment:
            img_t, mask_t = self._augment(img_t, mask_t)

        return img_t, mask_t

    @staticmethod
    def _augment(img, mask):
        if torch.rand(1) > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)
        if torch.rand(1) > 0.5:
            img  = TF.vflip(img)
            mask = TF.vflip(mask)
        return img, mask


# ─────────────────────────────────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────────────────────────────────
def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2, 2)

        ch = in_channels
        for f in features:
            self.downs.append(double_conv(ch, f))
            ch = f

        self.bottleneck = double_conv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(double_conv(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections.reverse()

        for i in range(0, len(self.ups), 2):
            x    = self.ups[i](x)
            skip = skip_connections[i // 2]
            if x.shape != skip.shape:
                x = TF.resize(x, skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return self.final(x)


# ─────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce    = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_val  = self.bce(logits, targets)
        probs    = torch.sigmoid(logits)
        inter    = (probs * targets).sum(dim=(1, 2, 3))
        dice_val = 1.0 - (2.0 * inter + self.smooth) / (
                        probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + self.smooth)
        return bce_val + dice_val.mean()


def iou_score(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets).clamp(0, 1).sum(dim=(1, 2, 3))
    return ((inter + 1e-6) / (union + 1e-6)).mean().item()


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────
def match_pairs(roi_dir, mask_dir):
    roi_files  = {p.stem: p for p in [*roi_dir.glob("*.png"), *roi_dir.glob("*.jpg")]}
    mask_files = {p.stem: p for p in [*mask_dir.glob("*.png"), *mask_dir.glob("*.jpg")]}
    common     = set(roi_files.keys()) & set(mask_files.keys())
    if not common:
        return [], []
    stems = sorted(common)
    return [roi_files[s] for s in stems], [mask_files[s] for s in stems]


def plot_history(history, run_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("U-Net Training – Crown/Stem/Peduncle Segmentation", fontsize=13, fontweight="bold")
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"],   label="Val")
    ax1.set_title("BCE + Dice Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["val_iou"], color="green")
    ax2.set_title("Validation IoU"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("IoU"); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "unet_training_curves.png", dpi=150)
    plt.close()


def save_sample_predictions(model, val_loader, run_dir, n=4):
    model.eval()
    imgs, masks, preds = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x.to(DEVICE))
            p = (torch.sigmoid(logits) > 0.5).cpu().float()
            imgs.extend(x[:n].unbind(0)); masks.extend(y[:n].unbind(0)); preds.extend(p[:n].unbind(0))
            if len(imgs) >= n:
                break

    MEAN = torch.tensor(ROIDataset.MEAN).view(3,1,1)
    STD  = torch.tensor(ROIDataset.STD ).view(3,1,1)
    fig, axes = plt.subplots(3, min(n, len(imgs)), figsize=(4*min(n,len(imgs)), 10))
    for col in range(min(n, len(imgs))):
        img_show = (imgs[col] * STD + MEAN).permute(1,2,0).clamp(0,1).numpy()
        axes[0, col].imshow(img_show); axes[0, col].axis("off")
        axes[1, col].imshow(masks[col].squeeze(), cmap="gray"); axes[1, col].axis("off")
        axes[2, col].imshow(preds[col].squeeze(), cmap="gray"); axes[2, col].axis("off")
    plt.tight_layout()
    plt.savefig(run_dir / "unet_sample_predictions.png", dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────
def train():
    print(f"\n═══ U-Net Training ═══  Device: {DEVICE}")

    imgs, masks = match_pairs(ROI_DIR, MASK_DIR)
    if not imgs:
        print("❌ No matching image/mask pairs"); sys.exit(1)
    print(f"  Found {len(imgs)} matched pairs")

    n_val   = max(1, int(len(imgs) * VAL_FRAC))
    n_train = len(imgs) - n_val

    g = torch.Generator().manual_seed(SEED)
    full_ds = ROIDataset(imgs, masks, augment=False)
    train_ds_raw, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    train_imgs  = [full_ds.image_paths[i] for i in train_ds_raw.indices]
    train_masks = [full_ds.mask_paths[i]  for i in train_ds_raw.indices]
    val_imgs    = [full_ds.image_paths[i] for i in val_ds.indices]
    val_masks   = [full_ds.mask_paths[i]  for i in val_ds.indices]

    train_ds = ROIDataset(train_imgs, train_masks, augment=True)
    val_ds   = ROIDataset(val_imgs,   val_masks,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    model     = UNet().to(DEVICE)
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "val_iou": []}
    best_iou = 0.0; best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch:3d}/{EPOCHS} [train]", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        scheduler.step()
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0; val_iou = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                val_loss += criterion(logits, y).item() * x.size(0)
                val_iou  += iou_score(logits, y) * x.size(0)
        val_loss /= len(val_ds)
        val_iou  /= len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        if val_iou > best_iou:
            best_iou = val_iou; best_epoch = epoch
            torch.save(model.state_dict(), MODEL_PATH)

        print(f"  Epoch {epoch:3d}/{EPOCHS} │ train={train_loss:.4f} │ val={val_loss:.4f} │ iou={val_iou:.4f}"
              + (" ← best" if epoch == best_epoch else ""))

    print(f"\n✅ Done! Best val IoU: {best_iou:.4f} (epoch {best_epoch})")
    print(f"   Model → {MODEL_PATH}")
    plot_history(history, RUN_DIR)
    save_sample_predictions(model, val_loader, RUN_DIR)


if __name__ == "__main__":
    train()
