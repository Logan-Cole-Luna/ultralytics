#!/usr/bin/env python3
"""Build the real-label AOT motion dataset from part1/ + part3/, split by whole flight.

Uses real ground-truth boxes from groundtruth.csv (gt_left/top/right/bottom), not dummy labels.
Background frames (no box) get an empty label file - a valid YOLO negative, not skipped.
Split is by whole flight (never interleaved frames) so val is a genuinely unseen sequence.

Current-frame images are hard-linked directly to the original AOT PNGs (no resize/recompress
copy) to avoid duplicating image storage - the training pipeline's own LetterBox/RandomPerspective
already resizes to the training imgsz at load time regardless of source resolution. Hard links
(not symlinks) are used because Windows requires elevated privileges for symlinks but not hard
links, and both live on the same volume here.

Motion maps are `stab3` — the winner of the extractor comparison on this data (see
compare_motion_extractors.py and runs/motion_compare/): each stride-1 neighbour frame is
homography-aligned onto the current frame (grid keypoints + pyramidal LK + RANSAC) and the map is
min(|I_t - warp(I_{t-1})|, |I_t - warp(I_{t+1})|) followed by a small Gaussian on the |diff|.
Alignment cancels camera ego-motion (clouds/horizon/terrain), the three-frame min suppresses
registration residue, and the post-blur integrates target energy against sensor noise. The map is
computed at native resolution (differencing after downsampling destroys few-pixel targets), then
stored at MOTION_SCALE with a fixed GAIN so the few-gray-level signal survives uint8 quantization
and the loader's resize. Raw absdiff at 640x480 (the previous scheme) never ranked the target
above background clutter on the benchmark windows.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from compare_motion_extractors import estimate_homography, post_blur

PARTS = ["part1", "part3"]
MAX_PART3_FLIGHTS = 10  # cap part3 (has ~37 flights) to keep prep/train time reasonable
FRAME_STRIDE = 2  # emit every 2nd frame per flight (halves volume; stab3 still uses stride-1 neighbours)
MOTION_SCALE = 0.5  # store motion at half native res: computed at native, downsized after
MOTION_GAIN = 6.0  # fixed gain on the stab3 map before uint8: typical target signal is 8-17 gray
# levels (see runs/motion_compare metrics.json), which would be nearly crushed by uint8 rounding +
# the loader's bilinear resize; a fixed (not per-image) gain keeps frames comparable so an empty
# frame stays dim instead of having its noise stretched to full range.
VAL_FRACTION_DENOM = 5  # ~1 in 5 flights (sorted, deterministic) held out for val


def warp_onto(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Homography-align grayscale frame *src* onto *dst* (identity fallback on failure)."""
    H, _ = estimate_homography(src, dst)
    if H is None:  # featureless scene (e.g. pure sky): raw diff is fine there anyway
        return src
    h, w = dst.shape
    return cv2.warpPerspective(src, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def stab3_map(curr: np.ndarray, neighbours: list) -> np.ndarray:
    """Stabilized three-frame difference of *curr* against its available stride-1 neighbours.

    With both neighbours this is min(back-diff, forward-diff); at sequence edges (or unreadable
    neighbour frames) it degrades to the single-sided stabilized diff, and to zeros with none.
    """
    diffs = [cv2.absdiff(curr, warp_onto(nb, curr)) for nb in neighbours if nb is not None]
    if not diffs:
        return np.zeros(curr.shape, dtype=np.float32)
    return post_blur(np.minimum(*diffs) if len(diffs) == 2 else diffs[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="motion_aot_dataset", help="dataset output directory")
    parser.add_argument("--max-frames-per-flight", type=int, default=None,
                        help="cap emitted frames per flight (smoke tests only)")
    args = parser.parse_args()
    dataset_dir = Path(args.out)

    print("[1/4] Scanning AOT parts and loading ground truth...")
    flight_to_dir = {}
    flight_to_rows = {}

    for part in PARTS:
        img_root = Path(part) / "Images"
        csv_path = Path(part) / "ImageSets" / "groundtruth.csv"
        if not img_root.exists() or not csv_path.exists():
            continue

        local_flights = sorted(d.name for d in img_root.iterdir() if d.is_dir() and any(d.iterdir()))
        if part == "part3":
            local_flights = local_flights[:MAX_PART3_FLIGHTS]
        local_flights = set(local_flights)
        if not local_flights:
            continue

        rows_by_flight = {f: {} for f in local_flights}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fid = row["flight_id"]
                if fid in local_flights:
                    rows_by_flight[fid][row["img_name"]] = row

        for fid in local_flights:
            flight_to_dir[fid] = img_root / fid
            flight_to_rows[fid] = rows_by_flight[fid]

    if not flight_to_dir:
        print("No flights with local images found under part1/ or part3/")
        sys.exit(1)

    # Deterministic flight-level split (whole flights only, no frame interleaving)
    all_flights = sorted(flight_to_dir.keys())
    val_flights = {fid for i, fid in enumerate(all_flights) if i % VAL_FRACTION_DENOM == 0}
    train_flights = set(all_flights) - val_flights

    n_total = sum(len(rows) for rows in flight_to_rows.values())
    n_box_total = sum(1 for rows in flight_to_rows.values() for r in rows.values() if r["gt_left"].strip())
    print(f"  {len(all_flights)} flights ({len(train_flights)} train / {len(val_flights)} val)")
    print(f"  {n_total} frames total, {n_box_total} with real boxes, stride={FRAME_STRIDE}")

    print("\n[2/4] Linking frames, generating stab3 motion maps, and writing labels...")
    for split in ["train", "val"]:
        for subdir in ["images", "motion", "labels"]:
            d = dataset_dir / subdir / split
            d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():  # clear stale contents from any previous prep
                f.unlink()
    for stale_cache in (dataset_dir / "labels").glob("*.cache"):  # force ultralytics label re-scan
        stale_cache.unlink()

    frame_count = 0
    split_counts = {"train": 0, "val": 0}
    box_counts = {"train": 0, "val": 0}

    for fid in all_flights:
        split = "val" if fid in val_flights else "train"
        img_dir = flight_to_dir[fid]
        rows = flight_to_rows[fid]

        # Full stride-1 sequence: stab3 always differences against true 0.1 s neighbours, even
        # though only every FRAME_STRIDE-th frame is emitted as a training sample.
        ordered_all = sorted(rows.keys(), key=lambda n: int(rows[n]["frame"]))
        emit_indices = list(range(0, len(ordered_all), FRAME_STRIDE))
        if args.max_frames_per_flight:
            emit_indices = emit_indices[: args.max_frames_per_flight]

        gray_cache: dict[int, np.ndarray | None] = {}  # sliding window of decoded gray frames

        def gray(idx: int) -> np.ndarray | None:
            """Grayscale frame at *idx* of ordered_all, cached; None if out of range/unreadable."""
            if not 0 <= idx < len(ordered_all):
                return None
            if idx not in gray_cache:
                gray_cache[idx] = cv2.imread(str(img_dir / ordered_all[idx]), cv2.IMREAD_GRAYSCALE)
            return gray_cache[idx]

        emitted = 0
        for j in emit_indices:
            img_name = ordered_all[j]
            row = rows[img_name]
            curr = gray(j)
            if curr is None:
                continue

            size_w = int(float(row["size_width"]))
            size_h = int(float(row["size_height"]))

            # Hard-link the original PNG directly - no resize/recompress copy.
            frame_name = f"frame_{frame_count:06d}.png"
            dest_path = dataset_dir / "images" / split / frame_name
            dest_path.unlink(missing_ok=True)
            os.link(img_dir / img_name, dest_path)

            # stab3 at native resolution, then downsize the *result* and apply fixed gain.
            motion = stab3_map(curr, [gray(j - 1), gray(j + 1)])
            motion = cv2.resize(motion, None, fx=MOTION_SCALE, fy=MOTION_SCALE, interpolation=cv2.INTER_AREA)
            motion = np.clip(motion * MOTION_GAIN, 0, 255).astype(np.uint8)
            # Must match the image file's extension (.png) - MotionYOLODataset._motion_path only
            # swaps the "images"/"motion" directory component, not the extension.
            cv2.imwrite(str(dataset_dir / "motion" / split / f"frame_{frame_count:06d}.png"), motion)

            for stale in [k for k in gray_cache if k < j + 1]:  # keep j+1 (next sample's back-neighbour)
                del gray_cache[stale]

            label_path = dataset_dir / "labels" / split / f"frame_{frame_count:06d}.txt"
            if row["gt_left"].strip():
                left, top = float(row["gt_left"]), float(row["gt_top"])
                right, bottom = float(row["gt_right"]), float(row["gt_bottom"])
                xc = (left + right) / 2 / size_w
                yc = (top + bottom) / 2 / size_h
                bw = (right - left) / size_w
                bh = (bottom - top) / size_h
                with open(label_path, "w") as lf:
                    lf.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                box_counts[split] += 1
            else:
                label_path.touch()

            frame_count += 1
            emitted += 1
            split_counts[split] += 1

        print(f"  {fid[:8]}...  ({split}): {emitted} frames")

    print(f"\n  Created {frame_count} frames")
    print(f"  Train: {split_counts['train']} frames ({box_counts['train']} with boxes)")
    print(f"  Val:   {split_counts['val']} frames ({box_counts['val']} with boxes)")

    if frame_count == 0 or split_counts["val"] == 0:
        print("Dataset incomplete (no val frames)")
        sys.exit(1)

    print("\n[3/4] Creating dataset configuration...")
    data_yaml = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "airborne_object"},
    }
    with open(dataset_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)
    print("  data.yaml created")

    print("\n[4/4] Done.")
    print(f"  Dataset ready at: {dataset_dir.resolve()}")


if __name__ == "__main__":
    main()
