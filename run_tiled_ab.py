#!/usr/bin/env python3
"""Queue the tiled A/B experiment: tile prep, then base vs motion at native resolution.

Run everything with one line:

    .venv\\Scripts\\python.exe run_tiled_ab.py all

Stages (also runnable individually: `run_tiled_ab.py prep|base|motion`):
  prep    Build tiled_aot_dataset/ - 640x640 native-resolution crops with stab3 motion tiles
          (prepare_tiled_dataset.py). A 6 px object stays 6 px; no resize anywhere.
  base    YOLO26s, 50 epochs on tiles, train_base_recipe.py hyperparameters.
  motion  YOLO26s-motion-p3sr (gate_init=0.1 + per-epoch gate logging), same recipe.

Changes vs the full-frame stab3_100e run, and why:
  - Tiles instead of full frames: that run scored 1% recall on <=8 px objects because
    imgsz=640 on 2448 px frames shrinks them below the wh_thr=2 augmentation filter and
    below any detection floor - and equally crushed the motion map the stream attends to.
  - 50 epochs, patience=20: the full-frame base plateaued by epoch ~10 and drifted down
    after; 100 epochs was overkill and mildly overfitting.
  - gate_init=0.1 (was 0): gradients into the motion pathway scale with tanh(gate), so a
    zero gate starved it at init; gates ended at |tanh| < 0.15 after 100 epochs. Watch
    stab3_tiled_ab/motion26s/gates.csv - if gates open decisively now, the motion stream
    is earning its keep; if they collapse toward 0, motion genuinely doesn't help.
  - Batch sizes (48 base / 24 motion) carry over from the VRAM probe on this RTX 5070 Ti
    (16 GB): exceeding VRAM doesn't fail on Windows, it silently pages over PCIe and
    crawls (~11 min/epoch at batch=96). nbs=64 grad accumulation keeps effective batch
    near the recipe's, so the recipe lr transfers.
"""

import subprocess
import sys
import time
from pathlib import Path

import torch

RECIPE = dict(
    data="tiled_aot_dataset/data.yaml",
    imgsz=640,  # tiles are exactly 640x640 - no letterbox, no resize
    epochs=50,
    patience=20,
    device=0 if torch.cuda.is_available() else "cpu",
    workers=6,
    plots=False,
    project="stab3_tiled_ab",
    exist_ok=True,
    copy_paste=0.01,
    cos_lr=True,
    freeze=None,
    lr0=0.0002,
    lrf=0.01,
    momentum=0.937,
    multi_scale=False,
    optimizer="SGD",
    scale=0.2,
    warmup_bias_lr=0.16,
    iou=0.7,
    single_cls=True,
)


def stage_prep():
    subprocess.run([sys.executable, "prepare_tiled_dataset.py"], check=True)


def stage_base():
    from ultralytics import YOLO

    model = YOLO("yolo26s.pt")  # pretrained
    model.train(**RECIPE, batch=48, name="base26s")


def stage_motion():
    from ultralytics.models.yolo.motion.train import MotionDetectionTrainer

    trainer = MotionDetectionTrainer(
        overrides=dict(
            **RECIPE,
            model="yolo26s-motion-p3sr.yaml",  # gate_init=0.1 set in the yaml motion: section
            pretrained="yolo26s.pt",
            batch=24,
            name="motion26s",
        ),
        motion_warmup_epochs=3,
        motion_lr_mult=5.0,
    )
    trainer.train()


STAGES = {"prep": stage_prep, "base": stage_base, "motion": stage_motion}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in STAGES:
        STAGES[which]()
        return
    if which != "all":
        sys.exit(f"usage: {Path(__file__).name} [prep|base|motion|all]")

    t0 = time.time()
    for name in ("prep", "base", "motion"):
        print(f"\n{'=' * 70}\nSTAGE: {name}  (elapsed {(time.time() - t0) / 3600:.1f} h)\n{'=' * 70}", flush=True)
        result = subprocess.run([sys.executable, __file__, name])
        if result.returncode != 0:
            sys.exit(f"Stage '{name}' failed with exit code {result.returncode}; aborting queue.")
    print(f"\nAll stages complete in {(time.time() - t0) / 3600:.1f} h.")
    print("Results: stab3_tiled_ab/base26s/ and stab3_tiled_ab/motion26s/ (+ gates.csv in motion26s)")


if __name__ == "__main__":
    main()
