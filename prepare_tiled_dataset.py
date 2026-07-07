#!/usr/bin/env python3
"""Build a tiled (native-resolution crop) AOT motion dataset from part1/ + part3/.

Why tiles: training full 2448x2048 frames at imgsz=640 shrinks a 6 px airborne object to
~1.6 px - below RandomPerspective's wh_thr=2 box filter and below any detection floor. The
100-epoch full-frame A/B measured 1% recall on <=8 px objects for exactly this reason, and
the resize equally destroyed the stab3 motion signal (gates barely opened). Tiles are
TILE x TILE windows cut at native resolution: a 6 px object stays 6 px and the motion map's
measured SNR reaches the model intact.

Per emitted frame (stride FRAME_STRIDE, using stride-1 neighbours for stab3):
  positives - one tile per GT box, window jittered around the box so the target is not
              always centered; labels include every GT box that lands in the window.
  negatives - a no-target window from the same frame (probabilistic), plus windows from
              background frames, so clouds/terrain/horizon appear as explicit negatives.

Motion tiles are cut from the full-resolution stab3 map (see prepare_dataset.py for the
extractor rationale) with the same fixed MOTION_GAIN - no downsampling anywhere.

Unlike prepare_dataset.py, all GT boxes per frame are kept (that script's rows dict was
keyed by image name and silently dropped all but the last box on multi-object frames).

Flights are processed in parallel (they're independent); the flight-level train/val split
is identical to prepare_dataset.py (sorted flights, every VAL_FRACTION_DENOM-th to val).

Usage:
    python prepare_tiled_dataset.py                    # full build -> tiled_aot_dataset/
    python prepare_tiled_dataset.py --max-frames-per-flight 6 --out smoke_tiles
"""

import argparse
import csv
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

from prepare_dataset import MOTION_GAIN, stab3_map

PARTS = ["part1", "part3"]
MAX_PART3_FLIGHTS = 10  # keep parity with prepare_dataset.py's flight selection
FRAME_STRIDE = 2
VAL_FRACTION_DENOM = 5
TILE = 640
JITTER_MARGIN = 8  # px a GT box must keep from the tile edge in its positive tile
NEG_PROB_BOXED = 0.5  # chance of also cutting a negative tile from a frame that has targets
NEG_PROB_BG = 0.3  # chance of cutting a negative tile from a background frame
NEG_CLEARANCE = 16  # a negative window must not contain any GT box center, +- this margin


def load_flights(max_part3: int = MAX_PART3_FLIGHTS):
    """Return {flight_id: (img_dir, {img_name: (frame_no, [boxes...])})} for local flights.

    Each box is (left, top, right, bottom) in native pixels; multi-object frames keep all boxes.
    """
    flights = {}
    for part in PARTS:
        img_root = Path(part) / "Images"
        csv_path = Path(part) / "ImageSets" / "groundtruth.csv"
        if not img_root.exists() or not csv_path.exists():
            continue
        local = sorted(d.name for d in img_root.iterdir() if d.is_dir() and any(d.iterdir()))
        if part == "part3":
            local = local[:max_part3]
        local = set(local)
        if not local:
            continue

        frames = {fid: defaultdict(lambda: [0, []]) for fid in local}
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                fid = row["flight_id"]
                if fid not in local:
                    continue
                entry = frames[fid][row["img_name"]]
                entry[0] = int(row["frame"])
                if row["gt_left"].strip():
                    entry[1].append(
                        (float(row["gt_left"]), float(row["gt_top"]),
                         float(row["gt_right"]), float(row["gt_bottom"]))
                    )
        for fid in local:
            flights[fid] = (img_root / fid, dict(frames[fid]))
    return flights


def clamp_window(cx: float, cy: float, w: int, h: int) -> tuple:
    """Clamp a TILE window centered near (cx, cy) fully inside a (h, w) image."""
    x0 = int(np.clip(cx - TILE / 2, 0, w - TILE))
    y0 = int(np.clip(cy - TILE / 2, 0, h - TILE))
    return x0, y0


def boxes_in_window(boxes: list, x0: int, y0: int) -> list:
    """YOLO-format labels for GT boxes whose center falls inside the window, clipped to it."""
    labels = []
    for l, t, r, b in boxes:
        cx, cy = (l + r) / 2, (t + b) / 2
        if not (x0 <= cx < x0 + TILE and y0 <= cy < y0 + TILE):
            continue
        cl, ct = max(l - x0, 0), max(t - y0, 0)
        cr, cb = min(r - x0, TILE), min(b - y0, TILE)
        if cr - cl < 3 or cb - ct < 3:  # sliver after clipping - would train on noise
            continue
        labels.append(
            f"0 {(cl + cr) / 2 / TILE:.6f} {(ct + cb) / 2 / TILE:.6f} "
            f"{(cr - cl) / TILE:.6f} {(cb - ct) / TILE:.6f}"
        )
    return labels


def sample_negative_window(boxes: list, w: int, h: int, rng) -> tuple | None:
    """A random window containing no GT box center (with clearance); None if not found."""
    for _ in range(5):
        x0 = int(rng.integers(0, w - TILE + 1))
        y0 = int(rng.integers(0, h - TILE + 1))
        if not any(
            x0 - NEG_CLEARANCE <= (l + r) / 2 < x0 + TILE + NEG_CLEARANCE
            and y0 - NEG_CLEARANCE <= (t + b) / 2 < y0 + TILE + NEG_CLEARANCE
            for l, t, r, b in boxes
        ):
            return x0, y0
    return None


def process_flight(task: tuple) -> dict:
    """Emit all tiles for one flight. Runs in a worker process; returns per-split counts."""
    flight_idx, fid, img_dir, frame_rows, split, out_dir, max_frames = task
    img_dir = Path(img_dir)
    rng = np.random.default_rng(flight_idx)  # per-flight seed: deterministic under parallelism

    ordered = sorted(frame_rows.keys(), key=lambda n: frame_rows[n][0])
    emit_indices = list(range(0, len(ordered), FRAME_STRIDE))
    if max_frames:
        emit_indices = emit_indices[:max_frames]

    gray_cache: dict[int, np.ndarray | None] = {}

    def gray(idx: int):
        if not 0 <= idx < len(ordered):
            return None
        if idx not in gray_cache:
            p = img_dir / ordered[idx]
            # Explicit existence check: partially-downloaded flights otherwise flood stderr
            # with an OpenCV findDecoder warning per missing file.
            gray_cache[idx] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None
        return gray_cache[idx]

    counts = {"tiles": 0, "pos": 0, "neg": 0, "boxes": 0}

    def write_tile(img, motion, x0, y0, labels, kind, frame_idx, k):
        name = f"{fid[:8]}_{frame_idx:05d}_{kind}{k}"
        cv2.imwrite(str(out_dir / "images" / split / f"{name}.png"), img[y0:y0 + TILE, x0:x0 + TILE])
        mtile = np.clip(motion[y0:y0 + TILE, x0:x0 + TILE] * MOTION_GAIN, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / "motion" / split / f"{name}.png"), mtile)
        (out_dir / "labels" / split / f"{name}.txt").write_text("\n".join(labels) + ("\n" if labels else ""))
        counts["tiles"] += 1
        counts["pos" if labels else "neg"] += 1
        counts["boxes"] += len(labels)

    for j in emit_indices:
        curr = gray(j)
        if curr is None:
            continue
        boxes = frame_rows[ordered[j]][1]
        h, w = curr.shape

        motion = stab3_map(curr, [gray(j - 1), gray(j + 1)])  # float32, native res, un-gained
        for stale in [k for k in gray_cache if k < j + 1]:
            del gray_cache[stale]

        for k, (l, t, r, b) in enumerate(boxes):
            bw, bh = r - l, b - t
            max_jit = max(TILE / 2 - max(bw, bh) / 2 - JITTER_MARGIN, 0)
            cx = (l + r) / 2 + rng.uniform(-max_jit, max_jit)
            cy = (t + b) / 2 + rng.uniform(-max_jit, max_jit)
            x0, y0 = clamp_window(cx, cy, w, h)
            labels = boxes_in_window(boxes, x0, y0)
            if labels:  # jitter+clamp can push the anchor box out only on degenerate GT; skip then
                write_tile(curr, motion, x0, y0, labels, "p", j, k)

        neg_prob = NEG_PROB_BOXED if boxes else NEG_PROB_BG
        if rng.random() < neg_prob:
            win = sample_negative_window(boxes, w, h, rng)
            if win is not None:
                write_tile(curr, motion, *win, [], "n", j, 0)

    return {"fid": fid, "split": split, **counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="tiled_aot_dataset", help="dataset output directory")
    parser.add_argument("--max-frames-per-flight", type=int, default=None,
                        help="cap emitted frames per flight (smoke tests only)")
    parser.add_argument("--workers", type=int, default=8, help="parallel flight workers")
    args = parser.parse_args()
    out_dir = Path(args.out)

    print("[1/3] Scanning AOT parts and loading ground truth...")
    flights = load_flights()
    if not flights:
        print("No flights with local images found under part1/ or part3/")
        sys.exit(1)
    all_flights = sorted(flights.keys())
    val_flights = {fid for i, fid in enumerate(all_flights) if i % VAL_FRACTION_DENOM == 0}
    print(f"  {len(all_flights)} flights ({len(all_flights) - len(val_flights)} train / {len(val_flights)} val)")

    for split in ["train", "val"]:
        for subdir in ["images", "motion", "labels"]:
            d = out_dir / subdir / split
            d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                f.unlink()
    for stale_cache in (out_dir / "labels").glob("*.cache"):
        stale_cache.unlink()

    print(f"\n[2/3] Cutting tiles ({args.workers} workers)...")
    tasks = [
        (i, fid, str(flights[fid][0]), flights[fid][1],
         "val" if fid in val_flights else "train", out_dir, args.max_frames_per_flight)
        for i, fid in enumerate(all_flights)
    ]
    totals = defaultdict(lambda: defaultdict(int))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(process_flight, tasks):
            print(f"  {res['fid'][:8]}...  ({res['split']}): {res['tiles']} tiles "
                  f"({res['pos']} pos / {res['neg']} neg, {res['boxes']} boxes)")
            for key in ("tiles", "pos", "neg", "boxes"):
                totals[res["split"]][key] += res[key]

    for split in ("train", "val"):
        s = totals[split]
        print(f"  {split}: {s['tiles']} tiles ({s['pos']} pos / {s['neg']} neg, {s['boxes']} boxes)")
    if totals["val"]["tiles"] == 0 or totals["train"]["tiles"] == 0:
        print("Dataset incomplete (empty split)")
        sys.exit(1)

    print("\n[3/3] Creating dataset configuration...")
    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "airborne_object"},
    }
    with open(out_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)
    print(f"  Done. Dataset ready at: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
