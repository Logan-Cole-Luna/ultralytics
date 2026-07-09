#!/usr/bin/env python3
"""Evaluate motion / base detectors (individually or as a WBF ensemble) on a test dataset.

Point CONFIG at any YOLO-format dataset laid out as::

    <DATA_DIR>/
      images/<SPLIT>/<name>.png      current-frame tiles
      motion/<SPLIT>/<name>.png      stab3 motion tiles  (required only for motion models)
      labels/<SPLIT>/<name>.txt      YOLO labels (one "cls xc yc w h" per line; empty = negative)

Everything is hard-coded in the CONFIG block below - edit it and run:

    .venv\\Scripts\\python.exe test_motion.py

Reports, per model and (if ENSEMBLE) the fused ensemble, on one shared protocol:
  mAP50 / mAP50-95   COCO 101-point AP, single class, from raw predictions
  precision / recall  at CONF_DET, greedy IoU >= IOU_MATCH matching
  recall by size      <=8 px / 8-16 px / >16 px  (native GT box size)
  FP tiles            negative tiles with any detection at >= CONF_DET

Ensemble = Weighted Box Fusion: per tile, detections from all models are clustered by mutual
IoU >= WBF_IOU, each cluster's box is confidence-weighted, and its score is combined by noisy-OR
(1 - prod(1 - conf)) so models agreeing on a box rank it up while a lone detection keeps its own
confidence (preserving complementary-model wins).
"""

from pathlib import Path

import cv2
import numpy as np
import torch

# ============================== CONFIG (edit these) ==============================
DATA_DIR = Path("tiled_aot_dataset")  # dataset root (see layout in module docstring)
SPLIT = "val"                          # which split subfolder to evaluate

# (label, weights .pt, is_motion): is_motion models also receive the paired motion tile
MODELS = [
    ("base", "runs/detect/stab3_ablation/base/weights/best.pt", False),
    ("hybrid", "runs/detect/stab3_ablation/hybrid/weights/best.pt", True),
]

ENSEMBLE = True    # True: also evaluate the WBF fusion of all MODELS; False: only individual models

WBF_IOU = 0.55     # IoU to cluster detections across models in the ensemble
CONF_KEEP = 0.01   # keep predictions above this (fed to mAP; below this never counts)
CONF_DET = 0.25    # operating-point threshold for precision / recall / FP-tile reporting
IOU_MATCH = 0.10   # loose IoU for a prediction to count as matching a (tiny) GT box
# ================================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE_BINS = [(0, 8, "<=8px"), (8, 16, "8-16px"), (16, 1e9, ">16px")]


# ------------------------------------------------------------------------- models


def load_model(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ckpt["model"].float().to(DEVICE).eval()
    return model


def decode(out) -> np.ndarray:
    """Normalize any YOLO/YOLO26 forward output to an (n, 5) [x0 y0 x1 y1 conf] array (conf-kept)."""
    preds = out[0] if isinstance(out, (tuple, list)) else out
    if isinstance(preds, dict):  # YOLO26 end2end returns {"one2one": ...}
        preds = preds.get("one2one", next(iter(preds.values())))
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
    preds = preds[0]
    preds = preds[preds[:, 4] > CONF_KEEP]
    return preds[:, :5].detach().cpu().numpy()


def predict_dataset(model, tiles, is_motion) -> dict:
    """{tile_stem: (n, 5) predictions} over every tile."""
    cache = {}
    img_dir, mot_dir = DATA_DIR / "images" / SPLIT, DATA_DIR / "motion" / SPLIT
    with torch.no_grad():
        for stem in tiles:
            img = cv2.imread(str(img_dir / f"{stem}.png"))
            t = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(DEVICE)
            kw = {}
            if is_motion:
                mp = mot_dir / f"{stem}.png"
                mot = cv2.imread(str(mp)) if mp.exists() else np.zeros_like(img)
                kw["motion"] = torch.from_numpy(mot).permute(2, 0, 1)[None].float().div(255).to(DEVICE)
            cache[stem] = decode(model(t, **kw))
    return cache


# ------------------------------------------------------------------- box utilities


def iou_matrix(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    if not len(preds) or not len(gts):
        return np.zeros((len(preds), len(gts)))
    px0, py0, px1, py1 = (preds[:, i:i + 1] for i in range(4))
    gx0, gy0, gx1, gy1 = (gts[None, :, i] for i in range(4))
    ix = np.clip(np.minimum(px1, gx1) - np.maximum(px0, gx0), 0, None)
    iy = np.clip(np.minimum(py1, gy1) - np.maximum(py0, gy0), 0, None)
    inter = ix * iy
    union = (px1 - px0) * (py1 - py0) + (gx1 - gx0) * (gy1 - gy0) - inter
    return np.where(union > 0, inter / union, 0.0)


def wbf(pred_sets, iou_thr=WBF_IOU) -> np.ndarray:
    """Weighted box fusion of per-model (n, 5) arrays: conf-weighted coords, noisy-OR scores."""
    dets = [p for p in pred_sets if len(p)]
    if not dets:
        return np.zeros((0, 5))
    dets = np.concatenate(dets, axis=0)
    dets = dets[np.argsort(-dets[:, 4])]
    clusters = []  # each: [box(4), [confs]]
    for d in dets:
        placed = False
        for c in clusters:
            if iou_matrix(d[None, :4], c[0][None])[0, 0] >= iou_thr:
                confs = np.array(c[1] + [d[4]])
                boxes = np.vstack([np.tile(c[0], (len(c[1]), 1)), d[:4]])
                c[0] = (boxes * confs[:, None]).sum(0) / confs.sum()
                c[1].append(float(d[4]))
                placed = True
                break
        if not placed:
            clusters.append([d[:4].copy(), [float(d[4])]])
    out = np.zeros((len(clusters), 5))
    for i, (box, confs) in enumerate(clusters):
        out[i, :4] = box
        out[i, 4] = 1.0 - np.prod([1.0 - c for c in confs])  # noisy-OR
    return out[np.argsort(-out[:, 4])] if len(out) else out


# ------------------------------------------------------------------------- metrics


def average_precision(cache, gt_by_tile, iou_thr) -> float:
    npos = sum(len(g) for g in gt_by_tile.values())
    if npos == 0:
        return 0.0
    records = [(p[4], t, i) for t, preds in cache.items() for i, p in enumerate(preds)]
    records.sort(key=lambda r: -r[0])
    ious = {t: iou_matrix(cache[t], gt_by_tile.get(t, np.zeros((0, 4)))) for t in cache}
    matched, tp = {t: set() for t in cache}, np.zeros(len(records))
    for n, (_, tile, i) in enumerate(records):
        m = ious[tile]
        if m.shape[1] == 0:
            continue
        for g in np.argsort(-m[i]):
            if m[i, g] < iou_thr:
                break
            if g not in matched[tile]:
                matched[tile].add(g)
                tp[n] = 1
                break
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / npos
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    return float(sum((precision[recall >= r].max() if (recall >= r).any() else 0.0) for r in np.linspace(0, 1, 101)) / 101)


def evaluate(cache, gt_by_tile, gt_sizes, neg_tiles) -> dict:
    """All metrics for one prediction cache."""
    # recall (by size) + precision at CONF_DET
    hit_bins = {b[2]: [0, 0] for b in SIZE_BINS}  # name -> [hits, total]
    tp = fp = 0
    for tile, gts in gt_by_tile.items():
        preds = cache[tile]
        preds = preds[preds[:, 4] > CONF_DET]
        m = iou_matrix(preds, gts)
        claimed = set()
        # precision: greedy highest-conf-first matching
        for i in np.argsort(-preds[:, 4]) if len(preds) else []:
            hit = False
            for g in np.argsort(-m[i]):
                if m[i, g] < IOU_MATCH:
                    break
                if g not in claimed:
                    claimed.add(g)
                    tp += 1
                    hit = True
                    break
            if not hit:
                fp += 1
        # recall per GT
        for g in range(len(gts)):
            name = next(b[2] for b in SIZE_BINS if b[0] <= gt_sizes[tile][g] < b[1])
            hit_bins[name][1] += 1
            if len(preds) and m[:, g].max() >= IOU_MATCH:
                hit_bins[name][0] += 1
    for tile in neg_tiles:  # detections on negative tiles are all false positives
        fp += int((cache[tile][:, 4] > CONF_DET).sum())

    tot_hit = sum(v[0] for v in hit_bins.values())
    tot = sum(v[1] for v in hit_bins.values())
    fp_tiles = sum(1 for t in neg_tiles if (cache[t][:, 4] > CONF_DET).any())
    return {
        "mAP50": average_precision(cache, gt_by_tile, 0.5),
        "mAP50-95": float(np.mean([average_precision(cache, gt_by_tile, t) for t in np.arange(0.5, 0.96, 0.05)])),
        "precision": tp / max(tp + fp, 1),
        "recall": tot_hit / max(tot, 1),
        "bins": {n: (h, t) for n, (h, t) in hit_bins.items()},
        "fp_tiles": (fp_tiles, len(neg_tiles)),
    }


# ---------------------------------------------------------------------------- main


def main():
    img_dir, lbl_dir = DATA_DIR / "images" / SPLIT, DATA_DIR / "labels" / SPLIT
    assert img_dir.is_dir(), f"missing {img_dir}"
    tiles = sorted(p.stem for p in img_dir.glob("*.png"))

    gt_by_tile, gt_sizes, neg_tiles = {}, {}, []
    for stem in tiles:
        h, w = cv2.imread(str(img_dir / f"{stem}.png")).shape[:2]
        lbl = lbl_dir / f"{stem}.txt"
        boxes, sizes = [], []
        if lbl.exists() and lbl.read_text().strip():
            for ln in lbl.read_text().strip().splitlines():
                _, xc, yc, bw, bh = map(float, ln.split())
                boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
                sizes.append(max(bw * w, bh * h))
        if boxes:
            gt_by_tile[stem] = np.array(boxes)
            gt_sizes[stem] = sizes
        else:
            neg_tiles.append(stem)

    n_gt = sum(len(g) for g in gt_by_tile.values())
    print(f"dataset: {DATA_DIR}/{SPLIT}  |  {len(tiles)} tiles  "
          f"({len(gt_by_tile)} positive / {len(neg_tiles)} negative, {n_gt} GT targets)")
    print(f"device: {DEVICE}  |  conf(det)={CONF_DET}  IoU(match)={IOU_MATCH}\n")

    caches, results = {}, {}
    for label, path, is_motion in MODELS:
        print(f"running {label} ({'motion' if is_motion else 'appearance'}) ...", flush=True)
        cache = predict_dataset(load_model(path), tiles, is_motion)
        caches[label] = cache
        results[label] = evaluate(cache, gt_by_tile, gt_sizes, neg_tiles)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if ENSEMBLE and len(MODELS) > 1:
        name = "WBF[" + "+".join(m[0] for m in MODELS) + "]"
        print(f"fusing {name} ...", flush=True)
        fused = {t: wbf([caches[m[0]][t] for m in MODELS]) for t in tiles}
        results[name] = evaluate(fused, gt_by_tile, gt_sizes, neg_tiles)

    # ---- report
    cols = ["mAP50", "mAP50-95", "precision", "recall"]
    hdr = f"{'model':<22}" + "".join(f"{c:>11}" for c in cols) + "".join(f"{b[2]:>10}" for b in SIZE_BINS) + f"{'FP tiles':>12}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for label, r in results.items():
        row = f"{label:<22}" + "".join(f"{r[c]:>11.3f}" for c in cols)
        for b in SIZE_BINS:
            h, t = r["bins"][b[2]]
            row += f"{(f'{h}/{t}'):>10}"
        fh, ft = r["fp_tiles"]
        row += f"{(f'{fh}/{ft}'):>12}"
        print(row)


if __name__ == "__main__":
    main()
