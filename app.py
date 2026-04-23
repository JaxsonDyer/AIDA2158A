"""
Streamlit Web UI – Strawberry Harvesting Pipeline Tester
Upload an image and see YOLO detections + cut line extraction in real time.

Run with:
  uv run streamlit run app.py
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).parent
BEST_PT = ROOT / "yolo_best.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Reuse pipeline logic from module4
# ─────────────────────────────────────────────────────────────────────────────
def redness_score(image_bgr: np.ndarray, mask: np.ndarray) -> float:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    red_lo = cv2.inRange(hsv, (0, 60, 60), (10, 255, 255))
    red_hi = cv2.inRange(hsv, (160, 60, 60), (180, 255, 255))
    red = cv2.bitwise_or(red_lo, red_hi)
    mask_bool = mask.astype(bool)
    if mask_bool.sum() == 0:
        return 0.0
    return float(red[mask_bool].sum()) / mask_bool.sum()


def extract_instances(result, img_h, img_w):
    if result.masks is None:
        return [], []
    masks = result.masks.data.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    strawberries, peduncles = [], []
    for i, cls_id in enumerate(cls_ids):
        raw = masks[i]
        bin_mask = cv2.resize(raw, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        bin_mask = (bin_mask > 0.5).astype(np.uint8)
        ys, xs = np.where(bin_mask > 0)
        if len(ys) == 0:
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        if cls_id == 1:
            strawberries.append({
                "mask": bin_mask,
                "centroid": (cx, cy),
                "redness": redness_score(result.orig_img, bin_mask),
                "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                "confidence": float(result.boxes.conf[i].cpu()),
            })
        elif cls_id == 0:
            peduncles.append({
                "mask": bin_mask,
                "centroid": (cx, cy),
                "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                "confidence": float(result.boxes.conf[i].cpu()),
            })
    return strawberries, peduncles


def match_peduncles(strawberries, peduncles, min_redness=0.10):
    if not strawberries or not peduncles:
        return []
    ripe = [s for s in strawberries if s["redness"] >= min_redness]
    if not ripe:
        return []
    matched = []
    used = set()
    ripe.sort(key=lambda s: s["redness"], reverse=True)
    for s in ripe:
        sx, sy = s["centroid"]
        best_dist, best_idx = float("inf"), -1
        for j, p in enumerate(peduncles):
            if j in used:
                continue
            px, py = p["centroid"]
            dist = np.sqrt((sx - px) ** 2 + (sy - py) ** 2)
            if dist < best_dist:
                best_dist, best_idx = dist, j
        if best_idx >= 0 and best_dist < 300:
            matched.append((s, peduncles[best_idx]))
            used.add(best_idx)
    return matched


def order_skeleton_points(skel):
    ys, xs = np.where(skel > 0)
    if len(ys) < 5:
        return None
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
    neighbor_count[skel == 0] = 0
    ep_y, ep_x = np.where((neighbor_count == 1) & (skel > 0))
    if len(ep_x) > 0:
        idx = np.argmin(ep_y)
        start = np.array([ep_x[idx], ep_y[idx]], dtype=np.float64)
    else:
        idx = np.argmin(xs + ys)
        start = points[idx]
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


def compute_cut_line(ped_mask, strawberry_centroid):
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
    sx, sy = strawberry_centroid
    d0 = np.sqrt((ordered[0][0] - sx) ** 2 + (ordered[0][1] - sy) ** 2)
    d1 = np.sqrt((ordered[-1][0] - sx) ** 2 + (ordered[-1][1] - sy) ** 2)
    if d1 > d0:
        ordered = ordered[::-1]
    diffs = np.diff(ordered, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum = np.concatenate([[0], np.cumsum(seg_lengths)])
    total = cum[-1]
    if total < 3:
        return _pca_fallback(ped_mask, strawberry_centroid)
    cut_len = total * 0.33
    idx = np.searchsorted(cum, cut_len) - 1
    idx = max(0, min(idx, len(ordered) - 2))
    frac = (cut_len - cum[idx]) / max(seg_lengths[idx], 1e-6)
    frac = np.clip(frac, 0, 1)
    cut_point = ordered[idx] + frac * (ordered[idx + 1] - ordered[idx])
    cx, cy = float(cut_point[0]), float(cut_point[1])
    window = max(5, len(ordered) // 8)
    lo, hi = max(0, idx - window), min(len(ordered), idx + window + 1)
    local_pts = ordered[lo:hi]
    if len(local_pts) < 2:
        return _pca_fallback(ped_mask, strawberry_centroid)
    centered = local_pts - local_pts.mean(axis=0)
    cov = np.cov(centered.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    principal = eig_vecs[:, np.argmax(eig_vals)]
    dx, dy = principal
    perp_dx, perp_dy = dy, -dx
    angle = float(np.degrees(np.arctan2(perp_dx, -perp_dy)))
    half = 25.0
    pt1 = (int(cx - perp_dx * half), int(cy - perp_dy * half))
    pt2 = (int(cx + perp_dx * half), int(cy + perp_dy * half))
    return {"cut_point": (int(cx), int(cy)), "cut_angle_deg": angle, "cut_pt1": pt1, "cut_pt2": pt2}


def _pca_fallback(ped_mask, strawberry_centroid):
    ys, xs = np.where(ped_mask > 0)
    if len(ys) < 10:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    sx, sy = strawberry_centroid
    dists = np.sqrt((pts[:, 0] - sx) ** 2 + (pts[:, 1] - sy) ** 2)
    weights = dists / (dists.max() + 1e-6)
    center = np.average(pts, weights=weights, axis=0)
    cx, cy = float(center[0]), float(center[1])
    centered = pts - center
    cov = np.cov(centered.T, aweights=weights)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    principal = eig_vecs[:, np.argmax(eig_vals)]
    dx, dy = principal
    perp_dx, perp_dy = dy, -dx
    angle = float(np.degrees(np.arctan2(perp_dx, -perp_dy)))
    half = 25.0
    pt1 = (int(cx - perp_dx * half), int(cy - perp_dy * half))
    pt2 = (int(cx + perp_dx * half), int(cy + perp_dy * half))
    return {"cut_point": (int(cx), int(cy)), "cut_angle_deg": angle, "cut_pt1": pt1, "cut_pt2": pt2}


def draw_overlay(img_bgr, matched_pairs):
    overlay = img_bgr.copy()
    results = []
    for straw, ped in matched_pairs:
        # Green peduncle highlight
        green = np.zeros_like(overlay)
        green[:, :, 1] = 255
        mask_bool = ped["mask"] > 0
        overlay[mask_bool] = cv2.addWeighted(overlay, 0.4, green, 0.6, 0)[mask_bool]
        # Strawberry outline
        straw_contours, _ = cv2.findContours(
            straw["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, straw_contours, -1, (0, 200, 255), 2)
        # Cut line
        cut = compute_cut_line(ped["mask"], straw["centroid"])
        if cut is None:
            results.append({"redness": straw["redness"], "angle": None, "cut_point": None})
            continue
        cv2.line(overlay, cut["cut_pt1"], cut["cut_pt2"], (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(overlay, cut["cut_point"], 8, (0, 255, 255), 2, cv2.LINE_AA)
        label = f"{cut['cut_angle_deg']:+.1f} deg"
        cv2.putText(
            overlay, label,
            (cut["cut_point"][0] + 12, cut["cut_point"][1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
        )
        results.append({
            "redness": straw["redness"],
            "angle": cut["cut_angle_deg"],
            "cut_point": cut["cut_point"],
        })
    # Info box
    n_cuts = sum(1 for r in results if r["angle"] is not None)
    cv2.putText(overlay, f"Cut lines: {n_cuts}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay, results


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Strawberry Pipeline", layout="wide")
st.title("🍓 Strawberry Harvesting Pipeline Tester")

# ── Model loading (cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not BEST_PT.exists():
        return None
    from ultralytics import YOLO
    return YOLO(str(BEST_PT))


model = load_model()
if model is None:
    st.error(
        f"YOLO weights not found at `{BEST_PT}`.\n\n"
        "Run `uv run python main.py --module 1` first to train the model."
    )
    st.stop()

# ── Sidebar settings ────────────────────────────────────────────────────────
st.sidebar.header("Settings")
redness_thresh = st.sidebar.slider(
    "Minimum redness threshold", 0.0, 0.5, 0.10, 0.01,
    help="HSV redness score below which strawberries are considered unripe.",
)
show_masks = st.sidebar.checkbox("Show individual masks", value=False)
confidence_thresh = st.sidebar.slider(
    "YOLO confidence threshold", 0.0, 1.0, 0.25, 0.05,
)

# ── File upload ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a strawberry image", type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    # ── Run pipeline ────────────────────────────────────────────────────
    with st.spinner("Running YOLO detection..."):
        results = model(img_bgr, verbose=False, conf=confidence_thresh)
        strawberries, peduncles = extract_instances(results[0], h, w)

    st.subheader("Detection Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Strawberries detected", len(strawberries))
    col2.metric("Peduncles detected", len(peduncles))
    col3.metric("Ripe strawberries (≥ threshold)",
                sum(1 for s in strawberries if s["redness"] >= redness_thresh))

    # ── Match & extract cut lines ───────────────────────────────────────
    matched = match_peduncles(strawberries, peduncles, min_redness=redness_thresh)

    if matched:
        overlay_img, cut_results = draw_overlay(img_bgr, matched)
        overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

        # Show side-by-side
        col_orig, col_overlay = st.columns(2)
        col_orig.image(img_rgb, caption="Original", use_container_width=True)
        col_overlay.image(overlay_rgb, caption="Cut Lines Overlay", use_container_width=True)

        # ── Per-strawberry details ──────────────────────────────────────
        st.subheader("Cut Line Details")
        for i, (straw, ped) in enumerate(matched):
            info = cut_results[i]
            with st.expander(
                f"Strawberry {i+1} — redness {straw['redness']:.2f}"
                + (f" — angle {info['angle']:+.1f}°" if info["angle"] is not None else " — no cut line")
            ):
                c1, c2 = st.columns(2)
                c1.write(f"**Centroid:** ({straw['centroid'][0]:.0f}, {straw['centroid'][1]:.0f})")
                c1.write(f"**Redness:** {straw['redness']:.3f}")
                c1.write(f"**Confidence:** {straw['confidence']:.2f}")
                c2.write(f"**Peduncle centroid:** ({ped['centroid'][0]:.0f}, {ped['centroid'][1]:.0f})")
                c2.write(f"**Peduncle confidence:** {ped['confidence']:.2f}")
                if info["angle"] is not None:
                    c2.write(f"**Cut angle:** {info['angle']:+.1f}° from vertical")
                    c2.write(f"**Cut point:** {info['cut_point']}")

                if show_masks:
                    mc1, mc2 = st.columns(2)
                    # Strawberry mask on crop
                    x1, y1, x2, y2 = straw["bbox"]
                    pad = 20
                    crop = img_bgr[max(0, y1 - pad):min(h, y2 + pad),
                                   max(0, x1 - pad):min(w, x2 + pad)]
                    crop_mask = straw["mask"][max(0, y1 - pad):min(h, y2 + pad),
                                              max(0, x1 - pad):min(w, x2 + pad)]
                    vis = crop.copy()
                    vis[crop_mask > 0] = (vis[crop_mask > 0] * 0.5 +
                                          np.array([0, 200, 255]) * 0.5).astype(np.uint8)
                    mc1.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Strawberry mask")
                    # Peduncle mask
                    ped_crop_mask = ped["mask"][max(0, y1 - pad):min(h, y2 + pad),
                                                max(0, x1 - pad):min(w, x2 + pad)]
                    vis2 = crop.copy()
                    vis2[ped_crop_mask > 0] = (vis2[ped_crop_mask > 0] * 0.5 +
                                               np.array([0, 255, 0]) * 0.5).astype(np.uint8)
                    mc2.image(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB), caption="Peduncle mask")
    else:
        st.warning("No ripe strawberry with matching peduncle found in this image.")
        st.image(img_rgb, caption="Original (no cut lines)", use_container_width=True)

else:
    st.info("Upload a strawberry image to get started.")
    # Show pipeline overview
    st.markdown("""
    ### Pipeline Overview

    | Step | Module | What It Does |
    |:-----|:-------|:-------------|
    | 1 | **YOLOv11-seg** | Detects strawberry instances + peduncle segments |
    | 1b | **ROI Crop & Auto-Mask** | Selects ripest strawberry, crops region, extracts peduncle mask |
    | 3 | **U-Net** | Segments peduncle pixels at high resolution |
    | 4 | **Cut Line Extraction** | Skeletonizes peduncle → finds cut point at 1/3 from tip → draws perpendicular cut line |

    Upload an image above to test the YOLO detection + cut line extraction on it.
    """)
