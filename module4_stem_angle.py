"""
Module 4 – Peduncle Cut Line Extraction
AIDA 2158A-F: Strawberry Harvesting Pipeline

Uses YOLO to detect both strawberries and peduncles in the original
full-size image, matches each ripe strawberry to its nearest peduncle,
then draws a perpendicular cut line on that peduncle for robotic
gripper alignment.

Output:
  runs/cut_lines/            – full-size overlay images with cut line annotation
  runs/cut_lines/angles.csv  – per-image cut angle values
"""

import sys
import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT       = Path(__file__).parent
DATASET    = ROOT / "dataset"
BEST_PT    = ROOT / "runs" / "segment" / "yolo11s_strawberry" / "weights" / "best.pt"
OUT_DIR    = ROOT / "runs" / "cut_lines"


# ─────────────────────────────────────────────────────────────────────────────
# Load YOLO
# ─────────────────────────────────────────────────────────────────────────────
def load_yolo():
    from ultralytics import YOLO
    return YOLO(str(BEST_PT))


# ─────────────────────────────────────────────────────────────────────────────
# Ripeness heuristic (same as module1_roi_crop.py)
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
# Parse YOLO results into per-instance masks
# ─────────────────────────────────────────────────────────────────────────────
def extract_instances(result, img_h: int, img_w: int):
    """
    Returns two lists:
      strawberries: [{'mask': H×W uint8, 'centroid': (cx, cy), 'redness': float}, ...]
      peduncles:    [{'mask': H×W uint8, 'centroid': (cx, cy)}, ...]
    """
    if result.masks is None:
        return [], []

    masks   = result.masks.data.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)

    strawberries = []
    peduncles    = []

    for i, cls_id in enumerate(cls_ids):
        raw = masks[i]
        bin_mask = cv2.resize(raw, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        bin_mask = (bin_mask > 0.5).astype(np.uint8)

        ys, xs = np.where(bin_mask > 0)
        if len(ys) == 0:
            continue
        cx, cy = float(xs.mean()), float(ys.mean())

        if cls_id == 1:  # strawberry
            strawberries.append({
                'mask': bin_mask,
                'centroid': (cx, cy),
                'redness': redness_score(result.orig_img, bin_mask),
            })
        elif cls_id == 0:  # peduncle
            peduncles.append({
                'mask': bin_mask,
                'centroid': (cx, cy),
            })

    return strawberries, peduncles


# ─────────────────────────────────────────────────────────────────────────────
# Match each ripe strawberry to its nearest peduncle
# ─────────────────────────────────────────────────────────────────────────────
def match_peduncles(strawberries, peduncles, min_redness=0.10):
    """
    For each strawberry above the redness threshold, find the closest
    peduncle (by centroid distance) and return matched pairs.
    A peduncle can only be matched to one strawberry (the closest one).

    Returns: list of (strawberry_dict, peduncle_dict) pairs
    """
    if not strawberries or not peduncles:
        return []

    # Filter to ripe strawberries
    ripe = [s for s in strawberries if s['redness'] >= min_redness]
    if not ripe:
        return []

    # Build cost matrix: distance from each ripe strawberry to each peduncle
    matched = []
    used_peduncles = set()

    # Sort ripe strawberries by redness (reddest first) to give priority
    ripe.sort(key=lambda s: s['redness'], reverse=True)

    for s in ripe:
        sx, sy = s['centroid']
        best_dist = float('inf')
        best_idx  = -1
        for j, p in enumerate(peduncles):
            if j in used_peduncles:
                continue
            px, py = p['centroid']
            dist = np.sqrt((sx - px) ** 2 + (sy - py) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        if best_idx >= 0 and best_dist < 300:  # reasonable max distance
            matched.append((s, peduncles[best_idx]))
            used_peduncles.add(best_idx)

    return matched


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton ordering: trace skeleton points into an ordered path
# ─────────────────────────────────────────────────────────────────────────────
def order_skeleton_points(skel: np.ndarray) -> np.ndarray | None:
    """
    Order skeleton points into a contiguous path via nearest-neighbor chaining.
    Returns (N, 2) array of (x, y) coordinates, or None.
    """
    ys, xs = np.where(skel > 0)
    if len(ys) < 5:
        return None

    points = np.stack([xs, ys], axis=1).astype(np.float64)

    # Find endpoints (1 neighbor in 8-connectivity)
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
    neighbor_count[skel == 0] = 0
    endpoints_y, endpoints_x = np.where((neighbor_count == 1) & (skel > 0))

    # Start from the topmost endpoint (smallest y = highest in image)
    if len(endpoints_x) > 0:
        idx = np.argmin(endpoints_y)
        start = np.array([endpoints_x[idx], endpoints_y[idx]], dtype=np.float64)
    else:
        idx = np.argmin(xs + ys)
        start = points[idx]

    # Nearest-neighbor ordering
    ordered = [start.copy()]
    remaining = set(range(len(points)))
    start_idx = np.argmin(np.sum((points - start) ** 2, axis=1))
    remaining.remove(start_idx)

    current = start.copy()
    while remaining:
        rem_arr = points[list(remaining)]
        dists = np.sum((rem_arr - current) ** 2, axis=1)
        nearest_local = np.argmin(dists)
        if dists[nearest_local] > 5.0:
            break
        nearest_global = list(remaining)[nearest_local]
        current = points[nearest_global].copy()
        ordered.append(current)
        remaining.remove(nearest_global)

    return np.array(ordered)


# ─────────────────────────────────────────────────────────────────────────────
# Compute cut line from a peduncle mask
# ─────────────────────────────────────────────────────────────────────────────
def compute_cut_line(ped_mask: np.ndarray, strawberry_centroid: tuple) -> dict | None:
    """
    Given a peduncle mask and the centroid of its associated strawberry,
    find the cut point on the peduncle and the perpendicular cut line.

    The cut point is placed at ~1/3 from the TOP of the skeleton
    (the end farthest from the strawberry), ensuring we cut on the
    actual stem, not the strawberry body.

    Returns dict with cut line info or None.
    """
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        return _pca_fallback(ped_mask, strawberry_centroid)

    skel = skeletonize(ped_mask > 0).astype(np.uint8)
    if skel.sum() < 5:
        return _pca_fallback(ped_mask, strawberry_centroid)

    ordered = order_skeleton_points(skel)
    if ordered is None or len(ordered) < 5:
        return _pca_fallback(ped_mask, strawberry_centroid)

    # Determine which end of the skeleton is "top" (farthest from strawberry)
    sx, sy = strawberry_centroid
    first_pt = ordered[0]
    last_pt  = ordered[-1]
    dist_first = np.sqrt((first_pt[0] - sx)**2 + (first_pt[1] - sy)**2)
    dist_last  = np.sqrt((last_pt[0] - sx)**2  + (last_pt[1] - sy)**2)

    # The "top" end is the one farther from the strawberry
    if dist_first >= dist_last:
        # First point is the top (peduncle tip)
        pass  # ordered is already top → bottom
    else:
        # Last point is the top, reverse so ordered goes top → bottom
        ordered = ordered[::-1]

    # Compute cumulative arc length
    diffs = np.diff(ordered, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_lengths[-1]

    if total_length < 3:
        return _pca_fallback(ped_mask, strawberry_centroid)

    # Cut point at 1/3 from the TOP of the peduncle
    cut_length = total_length * 0.33
    idx = np.searchsorted(cum_lengths, cut_length) - 1
    idx = max(0, min(idx, len(ordered) - 2))
    frac = (cut_length - cum_lengths[idx]) / max(seg_lengths[idx], 1e-6)
    frac = np.clip(frac, 0, 1)
    cut_point = ordered[idx] + frac * (ordered[idx + 1] - ordered[idx])
    cx, cy = float(cut_point[0]), float(cut_point[1])

    # Local tangent via PCA on neighboring skeleton points
    window = max(5, len(ordered) // 8)
    lo = max(0, idx - window)
    hi = min(len(ordered), idx + window + 1)
    local_pts = ordered[lo:hi]

    if len(local_pts) < 2:
        return _pca_fallback(ped_mask, strawberry_centroid)

    local_centered = local_pts - local_pts.mean(axis=0)
    cov = np.cov(local_centered.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    principal = eig_vecs[:, np.argmax(eig_vals)]
    dx, dy = principal

    # Perpendicular direction (rotate 90° clockwise)
    perp_dx, perp_dy = dy, -dx

    # Cut line angle from vertical
    cut_angle_rad = np.arctan2(perp_dx, -perp_dy)
    cut_angle_deg = float(np.degrees(cut_angle_rad))

    # Cut line endpoints — scale length relative to image
    line_half_len = 25.0
    pt1 = (int(cx - perp_dx * line_half_len), int(cy - perp_dy * line_half_len))
    pt2 = (int(cx + perp_dx * line_half_len), int(cy + perp_dy * line_half_len))

    return {
        'cut_point': (int(cx), int(cy)),
        'cut_angle_deg': cut_angle_deg,
        'cut_pt1': pt1,
        'cut_pt2': pt2,
    }


def _pca_fallback(ped_mask: np.ndarray, strawberry_centroid: tuple) -> dict | None:
    """PCA-based fallback when skeletonization fails."""
    ys, xs = np.where(ped_mask > 0)
    if len(ys) < 10:
        return None

    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    sx, sy = strawberry_centroid

    # Weight points by distance from strawberry (farther = higher weight)
    # to bias the centroid toward the peduncle tip
    dists = np.sqrt((pts[:, 0] - sx)**2 + (pts[:, 1] - sy)**2)
    weights = dists / (dists.max() + 1e-6)

    weighted_center = np.average(pts, weights=weights, axis=0)
    cx, cy = float(weighted_center[0]), float(weighted_center[1])

    pts_centered = pts - weighted_center
    cov = np.cov(pts_centered.T, aweights=weights)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    principal = eig_vecs[:, np.argmax(eig_vals)]
    dx, dy = principal

    perp_dx, perp_dy = dy, -dx
    cut_angle_rad = np.arctan2(perp_dx, -perp_dy)
    cut_angle_deg = float(np.degrees(cut_angle_rad))

    line_half_len = 25.0
    pt1 = (int(cx - perp_dx * line_half_len), int(cy - perp_dy * line_half_len))
    pt2 = (int(cx + perp_dx * line_half_len), int(cy + perp_dy * line_half_len))

    return {
        'cut_point': (int(cx), int(cy)),
        'cut_angle_deg': cut_angle_deg,
        'cut_pt1': pt1,
        'cut_pt2': pt2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation on the full-size image
# ─────────────────────────────────────────────────────────────────────────────
def draw_overlay(img_bgr: np.ndarray,
                 matched_pairs: list,
                 out_path: Path):
    overlay = img_bgr.copy()

    for straw, ped in matched_pairs:
        # Green overlay on peduncle
        green = np.zeros_like(overlay)
        green[:, :, 1] = 255
        mask_bool = ped['mask'] > 0
        overlay[mask_bool] = cv2.addWeighted(overlay, 0.4, green, 0.6, 0)[mask_bool]

        cut_info = compute_cut_line(ped['mask'], straw['centroid'])
        if cut_info is None:
            continue

        pt1 = cut_info['cut_pt1']
        pt2 = cut_info['cut_pt2']
        cp  = cut_info['cut_point']

        # Red cut line perpendicular to peduncle
        cv2.line(overlay, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

        # Yellow circle at cut point
        cv2.circle(overlay, cp, 8, (0, 255, 255), 2, cv2.LINE_AA)

        # Angle label near the cut point
        label = f"{cut_info['cut_angle_deg']:+.1f} deg"
        cv2.putText(overlay, label,
                    (cp[0] + 12, cp[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                    cv2.LINE_AA)

    # Top-left info
    n_cuts = sum(1 for s, p in matched_pairs if compute_cut_line(p['mask'], s['centroid']) is not None)
    cv2.putText(overlay, f"Cut lines: {n_cuts}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                cv2.LINE_AA)

    cv2.imwrite(str(out_path), overlay)


def save_angle_histogram(angles: list, out_dir: Path):
    valid = [a for a in angles if a is not None]
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(valid, bins=36, range=(-90, 90), color="#4CAF50", edgecolor="white")
    ax.set_title("Distribution of Cut Line Angles", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cut angle from vertical (degrees)")
    ax.set_ylabel("Count")
    ax.axvline(0, color="red", linestyle="--", label="Vertical")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = out_dir / "cut_angle_histogram.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  ✓  Angle histogram saved to {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Collect all dataset images
# ─────────────────────────────────────────────────────────────────────────────
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
    print("\n══════════════════════════════════════════")
    print("  Module 4: Peduncle Cut Line Extraction")
    print("══════════════════════════════════════════\n")

    if not BEST_PT.exists():
        print(f"❌  YOLO weights not found at:\n    {BEST_PT}")
        print("    Run module1_yolo_train.py first.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    yolo_model = load_yolo()

    images = all_images(DATASET)
    print(f"  Found {len(images)} original images in dataset")

    all_angles = []
    angles_rows = []

    for img_path in tqdm(images, desc="Extracting cut lines"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # YOLO → detect strawberries + peduncles
        results = yolo_model(img, verbose=False)
        strawberries, peduncles = extract_instances(results[0], h, w)

        # Match each ripe strawberry to its nearest peduncle
        matched = match_peduncles(strawberries, peduncles)

        if not matched:
            # No ripe strawberry with peduncle found — save original
            cv2.imwrite(str(OUT_DIR / img_path.name), img)
            angles_rows.append({"image": img_path.name, "cut_angle_deg": "N/A"})
            continue

        # Draw overlay and collect angles
        img_angles = []
        for straw, ped in matched:
            cut_info = compute_cut_line(ped['mask'], straw['centroid'])
            if cut_info is not None:
                img_angles.append(cut_info['cut_angle_deg'])
                all_angles.append(cut_info['cut_angle_deg'])
            else:
                all_angles.append(None)

        draw_overlay(img, matched, OUT_DIR / img_path.name)

        # CSV: one row per image, multiple angles joined with semicolons
        angle_strs = [f"{a:.2f}" if a is not None else "N/A" for a in img_angles]
        angles_rows.append({
            "image": img_path.name,
            "cut_angle_deg": ";".join(angle_strs) if angle_strs else "N/A",
        })

    # Save CSV
    csv_path = OUT_DIR / "angles.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "cut_angle_deg"])
        writer.writeheader()
        writer.writerows(angles_rows)
    print(f"  ✓  Angles written to {csv_path}")

    # Histogram
    save_angle_histogram(all_angles, OUT_DIR)

    valid = [a for a in all_angles if a is not None]
    if valid:
        print(f"\n  Total cut lines drawn : {len(valid)}")
        print(f"  Mean cut angle        : {np.mean(valid):+.1f}°")
        print(f"  Std  cut angle        : {np.std(valid):.1f}°")
        print(f"  Min / Max             : {np.min(valid):+.1f}° / {np.max(valid):+.1f}°")

    print(f"\n✅  Module 4 complete! Full-size overlay images + CSV in {OUT_DIR}")


if __name__ == "__main__":
    main()
