#!/usr/bin/env python3
"""Queue the full stab3 experiment: dataset prep, 100-epoch base, 100-epoch motion.

Run everything with one line:

    .venv\\Scripts\\python.exe run_stab3_pipeline.py all

Stages (also runnable individually: `run_stab3_pipeline.py prep|base|motion`):
  prep    Rebuild motion_aot_dataset/ with stab3 motion maps (prepare_dataset.py).
  base    YOLO26s, 100 epochs, the train_base_recipe.py hyperparameters.
  motion  YOLO26s-motion-p3sr, 100 epochs, same recipe adjusted for the dual stream.

`all` runs each stage in its own subprocess (clean CUDA state between trainings) and stops
on the first failure. Results land in stab3_100e/{base26s,motion26s}/.

Batch sizes come from an fwd+bwd AMP memory probe on this machine's RTX 5070 Ti (16 GB):
yolo26s peaks 27.2 GB at batch 96 (the train_base_recipe.py value - tuned for the much smaller
YOLOv8n, not YOLO26s) and 13.7 GB at batch 48; the motion variant peaks 17.0 GB at batch 32 and
12.8 GB at batch 24. Exceeding the card's 16.3 GB doesn't hard-fail on Windows - CUDA silently
pages into WDDM shared system memory over PCIe, which is why an earlier run at batch=96 showed
`GPU_mem 26.2G` in the log and crawled at ~11 min/epoch. batch=48 (base) / 24 (motion) keep true
VRAM headroom; ultralytics' nbs=64 gradient accumulation keeps effective batch close to the
recipe's 96, so the recipe lr (tuned at that effective batch) still transfers.
"""

import subprocess
import sys
import time
from pathlib import Path

import torch

# train_base_recipe.py hyperparameters, minus what each stage overrides (epochs/batch/model/name).
RECIPE = dict(
    data="motion_aot_dataset/data.yaml",
    imgsz=640,
    epochs=100,
    device=0 if torch.cuda.is_available() else "cpu",
    workers=6,
    plots=False,
    project="stab3_100e",
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
    subprocess.run([sys.executable, "prepare_dataset.py"], check=True)


def stage_base():
    from ultralytics import YOLO

    model = YOLO("yolo26s.pt")  # pretrained
    model.train(**RECIPE, batch=48, name="base26s")


def stage_motion():
    from ultralytics.models.yolo.motion.train import MotionDetectionTrainer

    # Same recipe as base for a clean A/B; motion-specific adjustments:
    #   model/batch     p3sr variant + batch 24 (memory probe, see module docstring)
    #   pretrained      start from the same yolo26s.pt (gate=0 keeps init equivalent to base)
    #   warmup/lr_mult  trainer defaults (3 epochs motion-only, 5x lr on the motion pathway) -
    #                   the designed mechanism for training fresh motion modules against a
    #                   converged pretrained backbone over a long schedule. Mosaic/mixup are
    #                   disabled by MotionYOLODataset itself for temporal coherence.
    trainer = MotionDetectionTrainer(
        overrides=dict(
            **RECIPE,
            model="yolo26s-motion-p3sr.yaml",
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
    print("Results: stab3_100e/base26s/  and  stab3_100e/motion26s/")


if __name__ == "__main__":
    main()
