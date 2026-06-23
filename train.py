#!/usr/bin/env python3
"""Training script for motion cross-attention detection on AOT dataset."""

import cv2
import numpy as np
from pathlib import Path
import yaml
import sys
import torch

print("="*70)
print("MOTION CROSS-ATTENTION WITH AOT DATASET")
print("="*70)
print()

# Step 1: Load AOT dataset
print("[1/5] Loading AOT dataset from part1/")
aot_path = Path('part1')
images_dir = aot_path / 'Images'

if not images_dir.exists():
    print(f"Images directory not found at {images_dir}")
    sys.exit(1)

# Discover flights by scanning Images directory directly
flight_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
print(f"Found {len(flight_dirs)} flights")

# Step 2: Extract frames and generate motion differences
print("[2/5] Extracting frames and generating motion differences...")

dataset_dir = Path('motion_aot_dataset')
for split in ['train', 'val']:
    for subdir in ['images', 'motion', 'labels']:
        (dataset_dir / subdir / split).mkdir(parents=True, exist_ok=True)

frame_count = 0
prev_frame = None

for flight_num, flight_dir in enumerate(flight_dirs):

    # Get PNG files
    img_files = sorted(list(flight_dir.glob('*.png')))[:100]  # Use up to 100 frames per flight

    if not img_files:
        print(f"No PNG files in {flight_dir.name}")
        continue

    print(f"  Flight {flight_num+1} ({flight_dir.name}): {len(img_files)} frames")

    for img_path in img_files:
        # Read frame
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        # Resize if needed
        h, w = frame.shape[:2]
        if w > 1000 or h > 1000:
            scale = min(640/w, 480/h)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

        # Determine split (80/20)
        is_train = frame_count % 5 != 0
        split = 'train' if is_train else 'val'

        frame_name = f'frame_{frame_count:06d}.jpg'

        # Save frame
        cv2.imwrite(str(dataset_dir / 'images' / split / frame_name), frame)

        # Save motion difference
        if prev_frame is None:
            motion = np.zeros_like(frame)
        else:
            motion = cv2.absdiff(frame, prev_frame)
        cv2.imwrite(str(dataset_dir / 'motion' / split / frame_name), motion)

        # Save label (dummy: centered object)
        with open(dataset_dir / 'labels' / split / frame_name.replace('.jpg', '.txt'), 'w') as f:
            f.write('0 0.5 0.5 0.2 0.2')

        prev_frame = frame
        frame_count += 1

print(f"  Created {frame_count} frames")

if frame_count == 0:
    print("No frames extracted!")
    sys.exit(1)

# Step 3: Create data.yaml
print("[3/5] Creating dataset configuration...")
data_yaml = {
    'path': str(dataset_dir.resolve()),
    'train': 'images/train',
    'val': 'images/val',
    'nc': 1,
    'names': {0: 'airborne_object'}
}
with open(dataset_dir / 'data.yaml', 'w') as f:
    yaml.dump(data_yaml, f)
print(f"  data.yaml created")

# Step 4: Train
print("[4/5] Training YOLOv8-motion...")
print(f"  Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
print(f"  Frames: {frame_count}")
print()

from ultralytics.models.yolo.motion.train import MotionDetectionTrainer

trainer = MotionDetectionTrainer(overrides={
    'model': 'yolov8-motion.yaml',
    'data': str(dataset_dir / 'data.yaml'),
    'epochs': 5,
    'imgsz': 416,
    'batch': 4,
    'device': 0 if torch.cuda.is_available() else 'cpu',
    'save': True,
    'workers': 2,
    'project': 'motion_aot',
    'name': 'train',
    'exist_ok': True,
})

trainer.train()

# Step 5: Validate
print("[5/5] Validating model...")
best_model = Path(trainer.save_dir) / 'weights' / 'best.pt'

if best_model.exists():
    from ultralytics.models.yolo.motion.val import MotionDetectionValidator

    validator = MotionDetectionValidator(args={
        'model': str(best_model),
        'data': str(dataset_dir / 'data.yaml'),
        'imgsz': 416,
        'batch': 4,
        'device': 0 if torch.cuda.is_available() else 'cpu',
    })
    validator()

    print("" + "="*70)
    print("MOTION DETECTION TRAINING COMPLETE")
    print("="*70)
    print(f"Results:")
    print(f"  Dataset: {frame_count} frames from {len(flight_dirs)} flights")
    print(f"  Model: {best_model}")
    print(f"  Results: {trainer.save_dir}")
    print(f"  Metrics: {trainer.save_dir}/results.csv")
else:
    print(f"Best model not found after training")
    sys.exit(1)
