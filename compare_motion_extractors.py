#!/usr/bin/env python3
"""Compare motion-extraction methods for the AOT motion stream on a real GT sequence.

Runs every candidate motion extractor over a window of consecutive frames from a local
AOT flight that has a confirmed moving ground-truth object, then reports per-frame and
aggregate target-vs-clutter statistics and saves visual crops for side-by-side review.

Methods
-------
raw_lowres   Current pipeline: absdiff of frames resized to 640x480 (prepare_dataset.py),
             upsampled back to native res (the dataset loader does the same upsample).
raw_fullres  Plain absdiff at native resolution, no ego-motion compensation.
stab_k1      Homography-stabilized diff, interval k=1: grid keypoints + pyramidal LK
             + RANSAC homography (YOLOMG-style), warp prev onto curr, absdiff, then a
             small Gaussian on the |diff| (post-diff energy integration: averages down
             sensor noise, which dominates the stabilized residual, while a 2-6 px
             target blob keeps most of its energy). Morphological opening is NOT
             used - it erases targets this small outright; pre-diff blur is also worse
             (it attenuates a moving point target ~3x more than structural clutter).
stab_k4      Same but interval k=4 (0.4 s at 10 Hz) - accumulates slow target motion.
stab3        Stabilized three-frame diff: min(|I_t - warp(I_{t-1})|, |I_t - warp(I_{t+1})|).
             The min suppresses registration residue and k-interval ghosts.
stab_dual    Dual-interval: channels [stab_k1, stab_k4, max(k1, k4)]; metrics on the max.
stab_med5    Registered background subtraction: warp the previous 5 frames onto the
             current one, take the pixelwise median as a background model, then
             |curr - median|. Median rejects the (moving) target from the background
             and averages sensor noise down ~sqrt(5); structural background cancels.
stab_acc4    Motion-energy accumulation (track-before-detect flavored): pixelwise max
             of the stabilized diffs |curr - warp(t-j)| for j=1..4. A slow target
             paints a bright streak; noise max grows much slower than signal.
flow_res     Farneback optical flow at 1/2 res minus the homography-induced global flow;
             residual magnitude as the motion image.

Metrics (computed on the raw float map, before any visual normalization)
------------------------------------------------------------------------
signal   max motion response inside the (padded) GT box
bg99.9   99.9th percentile of the response outside the box (the "brightest clutter")
SNR      signal / bg99.9  - >1 means the target outshines effectively all clutter
n_hot    count of pixels outside the (dilated) box with response >= signal, i.e. how
         many clutter pixels a detector would rank above the target

Usage:
    python compare_motion_extractors.py
    python compare_motion_extractors.py --flight 001578c6 --start 900 --frames 16
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

EST_SCALE = 1 / 3  # resolution used for homography estimation (full res is wasteful)
FLOW_SCALE = 1 / 2  # resolution used for Farneback flow (a 6 px target is 3 px here; 1/4 loses it)
BORDER = 24  # px excluded from metrics at each edge (warp border artifacts)
BOX_PAD = 4  # px padding around GT box when measuring signal
EXCL_PAD = 14  # px dilation of GT box when counting clutter pixels
LOWRES_SIZE = (640, 480)  # current prepare_dataset.py motion resolution
POST_BLUR = (3, 3), 0.8  # Gaussian applied to |diff| (kernel, sigma): energy integration
CONTEXT_BACK = 5  # trailing frames needed by stab_med5 / stab_acc4


# ---------------------------------------------------------------------------- data


def load_window(flight_prefix: str, start: int, n_eval: int, k_long: int):
    """Load frames [start - k_long, start + n_eval] and their GT rows for one flight."""
    gt_csv = None
    for part in ("part1", "part3"):
        p = Path(part) / "ImageSets" / "groundtruth.csv"
        if p.exists():
            with open(p, newline="") as f:
                for row in csv.DictReader(f):
                    if row["flight_id"].startswith(flight_prefix):
                        gt_csv = p
                        break
        if gt_csv:
            break
    if not gt_csv:
        raise SystemExit(f"flight {flight_prefix} not found in any local part")

    rows = {}
    flight_id = None
    with open(gt_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["flight_id"].startswith(flight_prefix):
                flight_id = row["flight_id"]
                rows[int(row["frame"])] = row

    img_dir = gt_csv.parent.parent / "Images" / flight_id
    lo, hi = start - k_long, start + n_eval
    frames, boxes = {}, {}
    for fno in range(lo, hi + 1):
        r = rows.get(fno)
        if r is None:
            continue
        img = cv2.imread(str(img_dir / r["img_name"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        frames[fno] = img
        if r["gt_left"].strip():
            boxes[fno] = tuple(float(r[k]) for k in ("gt_left", "gt_top", "gt_right", "gt_bottom"))
    return frames, boxes


# ------------------------------------------------------------------- registration


def estimate_homography(prev: np.ndarray, curr: np.ndarray):
    """Global homography prev->curr via uniform-grid points + pyramidal LK + RANSAC.

    Estimated at EST_SCALE and rescaled to full resolution. Returns (H, inlier_ratio);
    H is None when tracking/RANSAC fails (caller should fall back to identity).
    """
    h, w = prev.shape
    sw, sh = int(w * EST_SCALE), int(h * EST_SCALE)
    p_small = cv2.resize(prev, (sw, sh))
    c_small = cv2.resize(curr, (sw, sh))

    step = 24  # grid pitch at estimation scale -> ~30x25 points
    xs = np.arange(step, sw - step, step, dtype=np.float32)
    ys = np.arange(step, sh - step, step, dtype=np.float32)
    pts = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 1, 2)

    nxt, ok, _ = cv2.calcOpticalFlowPyrLK(
        p_small, c_small, pts, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    ok = ok.ravel().astype(bool)
    if ok.sum() < 20:
        return None, 0.0
    H_small, inliers = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 3.0)
    if H_small is None:
        return None, 0.0
    inlier_ratio = float(inliers.sum()) / int(ok.sum())

    s = np.diag([1 / EST_SCALE, 1 / EST_SCALE, 1.0])
    return s @ H_small @ np.linalg.inv(s), inlier_ratio


def post_blur(diff: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(diff.astype(np.float32), POST_BLUR[0], POST_BLUR[1])


def warp_to(frames: dict, src_t: int, dst_t: int, H_cache: dict) -> np.ndarray:
    """Frame src_t homography-warped into frame dst_t's coordinates."""
    src, dst = frames[src_t], frames[dst_t]
    key = (src_t, dst_t)
    if key not in H_cache:
        H_cache[key] = estimate_homography(src, dst)
    H, _ = H_cache[key]
    if H is None:  # featureless scene: identity fallback (raw diff is usually fine there)
        return src
    h, w = dst.shape
    return cv2.warpPerspective(src, H, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def stabilized_diff(frames: dict, t: int, k: int, H_cache: dict) -> np.ndarray:
    """post_blur(|I_t - warp(I_{t-k})|) with the previous frame aligned onto frame t."""
    return post_blur(cv2.absdiff(frames[t], warp_to(frames, t - k, t, H_cache)))


# ---------------------------------------------------------------------- extractors


def make_extractors(frames: dict, H_cache: dict):
    """Return {name: fn(t) -> float32 motion map at native resolution}."""

    def raw_lowres(t):
        h, w = frames[t].shape
        a = cv2.resize(frames[t], LOWRES_SIZE)
        b = cv2.resize(frames[t - 1], LOWRES_SIZE)
        return cv2.resize(cv2.absdiff(a, b), (w, h)).astype(np.float32)

    def raw_fullres(t):
        return cv2.absdiff(frames[t], frames[t - 1]).astype(np.float32)

    def stab_k1(t):
        return stabilized_diff(frames, t, 1, H_cache).astype(np.float32)

    def stab_k4(t):
        return stabilized_diff(frames, t, 4, H_cache).astype(np.float32)

    def stab3(t):
        # forward diff aligns t+1 onto t (offline prep has t+1; online costs 1 frame latency)
        back = stabilized_diff(frames, t, 1, H_cache)
        fwd = post_blur(cv2.absdiff(frames[t], warp_to(frames, t + 1, t, H_cache)))
        return np.minimum(back, fwd)

    def stab_dual(t):
        d1 = stab_k1(t)
        d4 = stab_k4(t)
        return np.stack([d1, d4, np.maximum(d1, d4)], axis=-1)

    def stab_med5(t):
        stack = np.stack([warp_to(frames, t - j, t, H_cache) for j in range(1, CONTEXT_BACK + 1)])
        background = np.median(stack, axis=0).astype(np.uint8)
        return post_blur(cv2.absdiff(frames[t], background))

    def stab_acc4(t):
        acc = stabilized_diff(frames, t, 1, H_cache)
        for j in range(2, 5):
            acc = np.maximum(acc, stabilized_diff(frames, t, j, H_cache))
        return acc

    def flow_res(t):
        curr, prev = frames[t], frames[t - 1]
        h, w = curr.shape
        sw, sh = int(w * FLOW_SCALE), int(h * FLOW_SCALE)
        p = cv2.resize(prev, (sw, sh))
        c = cv2.resize(curr, (sw, sh))
        flow = cv2.calcOpticalFlowFarneback(p, c, None, 0.5, 3, 11, 3, 5, 1.2, 0)

        key = (t - 1, t)
        if key not in H_cache:
            H_cache[key] = estimate_homography(prev, curr)
        H, _ = H_cache[key]
        if H is not None:  # noqa: SIM108 - grid math below stays readable in a block
            # global flow induced by the homography, evaluated at flow resolution
            s = np.diag([FLOW_SCALE, FLOW_SCALE, 1.0])
            Hs = s @ H @ np.linalg.inv(s)
            gx, gy = np.meshgrid(np.arange(sw, dtype=np.float32), np.arange(sh, dtype=np.float32))
            ones = np.ones_like(gx)
            pts = np.stack([gx, gy, ones], axis=-1) @ Hs.T
            induced = pts[..., :2] / pts[..., 2:3] - np.stack([gx, gy], axis=-1)
            flow = flow - induced.astype(np.float32)
        mag = np.linalg.norm(flow, axis=-1) / FLOW_SCALE  # full-res pixel units
        return cv2.resize(mag, (w, h)).astype(np.float32)

    return {
        "raw_lowres": raw_lowres,
        "raw_fullres": raw_fullres,
        "stab_k1": stab_k1,
        "stab_k4": stab_k4,
        "stab3": stab3,
        "stab_dual": stab_dual,
        "stab_med5": stab_med5,
        "stab_acc4": stab_acc4,
        "flow_res": flow_res,
    }


# ------------------------------------------------------------------------- metrics


def score(motion: np.ndarray, box: tuple) -> dict:
    """Target-vs-clutter statistics for one motion map (single channel or max over channels)."""
    m = motion.max(axis=-1) if motion.ndim == 3 else motion
    h, w = m.shape
    l, t, r, b = box
    x0, y0 = max(int(l) - BOX_PAD, 0), max(int(t) - BOX_PAD, 0)
    x1, y1 = min(int(r) + BOX_PAD, w), min(int(b) + BOX_PAD, h)

    signal = float(m[y0:y1, x0:x1].max())

    bg = m[BORDER:-BORDER, BORDER:-BORDER].copy()
    ex0, ey0 = max(x0 - EXCL_PAD - BORDER, 0), max(y0 - EXCL_PAD - BORDER, 0)
    ex1, ey1 = max(x1 + EXCL_PAD - BORDER, 0), max(y1 + EXCL_PAD - BORDER, 0)
    bg[ey0:ey1, ex0:ex1] = -1.0  # exclude target neighbourhood from background stats
    valid = bg[bg >= 0]

    bg999 = float(np.percentile(valid, 99.9))
    return {
        "signal": signal,
        "bg99.9": bg999,
        "snr": signal / (bg999 + 1e-6),
        "n_hot": int((valid >= signal).sum()) if signal > 0 else int(valid.size),
    }


# ------------------------------------------------------------------------- visuals


def norm_u8(m: np.ndarray) -> np.ndarray:
    """Robust per-image contrast stretch to uint8 for visualization/storage."""
    hi = np.percentile(m, 99.8)
    return np.clip(m / max(hi, 1e-6) * 255.0, 0, 255).astype(np.uint8)


def save_visuals(out_dir: Path, name: str, t: int, motion: np.ndarray, box: tuple, crop_half: int = 64):
    l, tp, r, b = box
    cx, cy = int((l + r) / 2), int((tp + b) / 2)
    if motion.ndim == 3:
        vis = np.stack([norm_u8(motion[..., c]) for c in range(3)], axis=-1)
    else:
        vis = norm_u8(motion)
    h, w = vis.shape[:2]
    x0, x1 = np.clip([cx - crop_half, cx + crop_half], 0, w)
    y0, y1 = np.clip([cy - crop_half, cy + crop_half], 0, h)
    crop = cv2.resize(vis[y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_NEAREST)
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    # GT box outline in the crop (scale 2x from 128->256)
    sx = 256 / (x1 - x0)
    cv2.rectangle(crop, (int((l - x0) * sx) - 2, int((tp - y0) * sx) - 2),
                  (int((r - x0) * sx) + 2, int((b - y0) * sx) + 2), (0, 0, 255), 1)
    cv2.imwrite(str(out_dir / f"crop_f{t:04d}_{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])


def save_fullframe(out_dir: Path, name: str, t: int, motion: np.ndarray, box: tuple):
    vis = norm_u8(motion.max(axis=-1) if motion.ndim == 3 else motion)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    l, tp, r, b = box
    cv2.rectangle(vis, (int(l) - 24, int(tp) - 24), (int(r) + 24, int(b) + 24), (0, 0, 255), 6)
    vis = cv2.resize(vis, (734, 614))
    cv2.imwrite(str(out_dir / f"full_f{t:04d}_{name}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flight", default="001578c6", help="flight_id prefix")
    ap.add_argument("--start", type=int, default=900, help="first evaluated frame number")
    ap.add_argument("--frames", type=int, default=16, help="number of evaluated frames")
    ap.add_argument("--k-long", type=int, default=4, help="long interval for stab_k4/stab_dual")
    ap.add_argument("--out", default="runs/motion_compare", help="output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    context = max(args.k_long, CONTEXT_BACK)
    print(f"Loading flight {args.flight} frames {args.start - context}..{args.start + args.frames}")
    frames, boxes = load_window(args.flight, args.start, args.frames, context)
    eval_frames = [t for t in range(args.start, args.start + args.frames)
                   if t in boxes and all(t + o in frames for o in range(-context, 2))]
    print(f"  {len(frames)} frames loaded, {len(eval_frames)} evaluable (GT box + temporal context)")

    H_cache = {}
    extractors = make_extractors(frames, H_cache)
    full_frame_t = eval_frames[len(eval_frames) // 2]

    results = {name: [] for name in extractors}
    for t in eval_frames:
        for name, fn in extractors.items():
            motion = fn(t)
            s = score(motion, boxes[t])
            s["frame"] = t
            results[name].append(s)
            save_visuals(out_dir, name, t, motion, boxes[t])
            if t == full_frame_t:
                save_fullframe(out_dir, name, t, motion, boxes[t])

    # per-frame SNR table
    print(f"\nPer-frame SNR (signal / 99.9th-pct background)  flight={args.flight}")
    names = list(extractors)
    print(f"{'frame':>6} " + " ".join(f"{n:>12}" for n in names))
    for i, t in enumerate(eval_frames):
        print(f"{t:>6} " + " ".join(f"{results[n][i]['snr']:>12.2f}" for n in names))

    # aggregate summary
    print(f"\n{'method':<12}{'med SNR':>10}{'SNR>=1':>10}{'med n_hot':>12}{'med signal':>12}{'med bg99.9':>12}")
    summary = {}
    for n in names:
        snrs = [r["snr"] for r in results[n]]
        n_hots = [r["n_hot"] for r in results[n]]
        summary[n] = {
            "median_snr": float(np.median(snrs)),
            "frac_snr_ge_1": float(np.mean([s >= 1 for s in snrs])),
            "median_n_hot": float(np.median(n_hots)),
            "median_signal": float(np.median([r["signal"] for r in results[n]])),
            "median_bg999": float(np.median([r["bg99.9"] for r in results[n]])),
        }
        s = summary[n]
        print(f"{n:<12}{s['median_snr']:>10.2f}{s['frac_snr_ge_1']:>10.0%}"
              f"{s['median_n_hot']:>12.0f}{s['median_signal']:>12.1f}{s['median_bg999']:>12.1f}")

    inlier_ratios = [r for _, (H, r) in H_cache.items() if H is not None]
    print(f"\nHomography inlier ratio: median {np.median(inlier_ratios):.2f} "
          f"(n={len(inlier_ratios)}, failures={sum(1 for _, (H, _) in H_cache.items() if H is None)})")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"config": vars(args), "per_frame": results, "summary": summary}, f, indent=2)
    print(f"\nVisuals + metrics.json written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
