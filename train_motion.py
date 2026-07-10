#!/usr/bin/env python3
"""Train the best-performing motion detector (hybrid) on a given YOLO data.yaml.

`hybrid` = yolo26s-motion-hybrid.yaml: MotionPixelFusion at P3 (pixel-aligned fusion, preserves
few-pixel target signal) + MotionCrossAttention at P4/P5 (scene-level motion reasoning). It won
the fusion-operator study and, ensembled with a plain base model, was the overall top performer
(see MOTION_DETECTION.md). Everything below is hard-coded to that winning configuration - set
DATA_YAML to your dataset and run:

    .venv\\Scripts\\python.exe train_motion.py

The dataset must follow the motion layout (a `motion/` dir beside `images/`, same filenames):

    <path>/
      images/{train,val}/<name>.png    current-frame tiles
      motion/{train,val}/<name>.png    stab3 motion tiles  (build with prepare_tiled_dataset.py)
      labels/{train,val}/<name>.txt    YOLO labels
    data.yaml: path/train/val/nc/names  (standard YOLO detection config)

Missing motion tiles fall back to zeros automatically, so a plain YOLO dataset also trains (the
motion stream just contributes nothing for those samples).
"""

import torch

from ultralytics.models.yolo.motion.train import MotionDetectionTrainer

# ============================== CONFIG (edit these) ==============================
DATA_YAML = "tiled_aot_dataset/data.yaml"   # <-- your dataset config

MODEL = "yolo26s-motion-hybrid.yaml"        # winning architecture (pixel@P3 + attn@P4/P5)
PRETRAINED = "yolo26s.pt"                    # base weights transfer into the current-frame stream

EPOCHS = 60          # hybrid converges by ~epoch 20 on AOT; patience stops earlier if plateaued
PATIENCE = 15
BATCH = 24           # tuned for a 16 GB GPU at imgsz 640; raise/lower to your VRAM
IMGSZ = 640          # train at native tile resolution - do NOT downscale tiny targets
SINGLE_CLS = True    # AOT is one class ("airborne_object"); set False for a multi-class dataset

PROJECT = "train_motion"   # results land in runs/detect/<PROJECT>/<NAME>/
NAME = "hybrid"
# ================================================================================

# Winning recipe (from the ablation study) - lr/schedule/aug that produced the best hybrid.
RECIPE = dict(
    optimizer="SGD",
    lr0=0.0002,
    lrf=0.01,
    momentum=0.937,
    cos_lr=True,
    warmup_bias_lr=0.16,
    scale=0.2,          # RandomPerspective zoom; mosaic/mixup stay off (motion coherence)
    copy_paste=0.01,
    multi_scale=False,
    iou=0.7,
    freeze=None,
    seed=0,             # deterministic, reproducible runs
    deterministic=True,
    plots=False,
    exist_ok=True,
)

# Motion-pathway training: gate_init=0.1 (set in the yaml) feeds gradient from step 1, so no
# freeze-warmup is needed; the 5x lr keeps the fresh motion modules learning once joint training
# starts. These were the settings behind the best hybrid run.
MOTION_WARMUP_EPOCHS = 0
MOTION_LR_MULT = 5.0


def main():
    trainer = MotionDetectionTrainer(
        overrides=dict(
            model=MODEL,
            pretrained=PRETRAINED,
            data=DATA_YAML,
            epochs=EPOCHS,
            patience=PATIENCE,
            batch=BATCH,
            imgsz=IMGSZ,
            single_cls=SINGLE_CLS,
            device=0 if torch.cuda.is_available() else "cpu",
            project=PROJECT,
            name=NAME,
            **RECIPE,
        ),
        motion_warmup_epochs=MOTION_WARMUP_EPOCHS,
        motion_lr_mult=MOTION_LR_MULT,
    )
    trainer.train()
    print(f"\nbest weights: {trainer.save_dir}/weights/best.pt")
    print(f"evaluate with:  test_motion.py  (point MODELS at the best.pt above)")


if __name__ == "__main__":
    main()
