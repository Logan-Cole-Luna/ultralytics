#!/usr/bin/env python3
"""Build the real-label AOT motion dataset from part1/ + part3/, split by whole flight.

Uses real ground-truth boxes from groundtruth.csv (gt_left/top/right/bottom), not dummy labels.
Background frames (no box) get an empty label file - a valid YOLO negative, not skipped.
Split is by whole flight (never interleaved frames) so val is a genuinely unseen sequence.
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PARTS = ["part1", "part3"]
MAX_PART3_FLIGHTS = 10  # cap part3 (has ~37 flights) to keep prep/train time reasonable
FRAME_STRIDE = 2  # take every 2nd frame per flight (halves volume, motion diff still meaningful)
IMG_SIZE_TARGET = (1280, 960)  # (w, h) cap; uniform scale keeps normalized bbox fractions valid.
# AOT objects are extremely small in pixel terms (median ~2.7px at the old 640x480 cap trained at
# imgsz=416); this preserves ~4x more real detail from the 2448x2048 source so objects have enough
# pixels to be representable at the model's stride-8 grid. Train at a matching higher imgsz.
VAL_FRACTION_DENOM = 5  # ~1 in 5 flights (sorted, deterministic) held out for val

dataset_dir = Path("motion_aot_dataset")


def main():
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

    print("\n[2/4] Extracting frames, motion differences, and labels...")
    for split in ["train", "val"]:
        for subdir in ["images", "motion", "labels"]:
            d = dataset_dir / subdir / split
            d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():  # clear stale contents from any previous prep
                f.unlink()

    frame_count = 0
    split_counts = {"train": 0, "val": 0}
    box_counts = {"train": 0, "val": 0}

    for fid in all_flights:
        split = "val" if fid in val_flights else "train"
        img_dir = flight_to_dir[fid]
        rows = flight_to_rows[fid]

        ordered_names = sorted(rows.keys(), key=lambda n: int(rows[n]["frame"]))[::FRAME_STRIDE]

        prev_frame = None
        for img_name in ordered_names:
            row = rows[img_name]
            frame = cv2.imread(str(img_dir / img_name))
            if frame is None:
                continue

            orig_h, orig_w = frame.shape[:2]
            size_w = int(float(row["size_width"]))
            size_h = int(float(row["size_height"]))

            scale = min(IMG_SIZE_TARGET[0] / orig_w, IMG_SIZE_TARGET[1] / orig_h)
            frame_resized = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)))

            frame_name = f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(dataset_dir / "images" / split / frame_name), frame_resized)

            motion = np.zeros_like(frame_resized) if prev_frame is None else cv2.absdiff(frame_resized, prev_frame)
            cv2.imwrite(str(dataset_dir / "motion" / split / frame_name), motion)

            label_path = dataset_dir / "labels" / split / frame_name.replace(".jpg", ".txt")
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

            prev_frame = frame_resized
            frame_count += 1
            split_counts[split] += 1

        print(f"  {fid[:8]}...  ({split}): {len(ordered_names)} frames")

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
