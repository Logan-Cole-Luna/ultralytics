#!/usr/bin/env python3
"""Train a motion detector variant on a YOLO motion dataset, or A/B/C several in one run.

Three architectures are registered (all share the winning motion recipe below):

    hybrid     yolo26s-motion-hybrid.yaml     Detect(P3,P4,P5) - the previous best. Baseline.
    p2det      yolo26s-motion-p2det.yaml      Detect(P2,P3,P4) - detection stride dropped one
                                              octave so a 6 px target is 1.5 grid cells (0.75
                                              at P3). Motion pixel-fused at P2/P3, attn at P4.
    p2det-spd  yolo26s-motion-p2det-spd.yaml  p2det + space-to-depth (Focus) stem: lossless
                                              downsampling (keeps 100% of a point target's
                                              contrast vs ~50%/14% for strided conv at s4/s8).

Why these: AOT airborne objects are ~6 px at native res; 51% are <=8 px. The stock P3/P4/P5
head cannot resolve them (0.75 cells on the finest grid). See MOTION_DETECTION.md analysis.

The dataset must follow the motion layout (a `motion/` dir beside `images/`, same filenames),
built by prepare_tiled_dataset.py:

    <path>/
      images/{train,val}/<name>.png    current-frame tiles
      motion/{train,val}/<name>.png    stab3 motion tiles
      labels/{train,val}/<name>.txt    YOLO labels
    data.yaml: path/train/val/nc/names

Run one variant, or all three sequentially:

    .venv/bin/python train_motion.py --variant p2det
    .venv/bin/python train_motion.py --variant all

Missing motion tiles fall back to zeros automatically, so a plain YOLO dataset also trains.
"""

import argparse

import torch

from ultralytics.models.yolo.motion.train import MotionDetectionTrainer

# ============================== CONFIG (edit these) ==============================
DATA_YAML = "tiled_aot_dataset/data.yaml"   # <-- your dataset config (built by prepare_tiled_dataset.py)

# name -> model yaml. The `s` in the filename selects scale s; the file on disk has no scale letter.
VARIANTS = {
    "hybrid": "yolo26s-motion-hybrid.yaml",
    "p2det": "yolo26s-motion-p2det.yaml",
    "p2det-spd": "yolo26s-motion-p2det-spd.yaml",
    # Fusion-operator ablation (all on the p2det P2/P3/P4 head; attn kept at coarse P4).
    # Narrows down HOW the motion stream is best used at the fine scales.
    "p2det-film": "yolo26s-motion-p2det-film.yaml",          # FiLM multiplicative modulation
    "p2det-cg": "yolo26s-motion-p2det-cg.yaml",              # per-channel (LayerScale) gated pixel fusion
    "p2det-wattn": "yolo26s-motion-p2det-wattn.yaml",        # windowed (local) cross-attention
    "p2det-spatgate": "yolo26s-motion-p2det-spatgate.yaml",  # minimal spatial 'look-here' gate
    "p2det-filmpix": "yolo26s-motion-p2det-filmpix.yaml",    # pixel@P2 + FiLM@P3 (box-quality combo)
}

PRETRAINED = "yolo26s.pt"    # base weights transfer into the current-frame backbone via intersect_dicts;
                             # the P2/P3/P4 head is re-strided so its layers init fresh (expected, fine).

EPOCHS = 60          # hybrid converges by ~epoch 20 on AOT; patience stops earlier if plateaued
PATIENCE = 25
BATCH = 24           # tuned for a 16 GB GPU at imgsz 640. p2det* add a stride-4 (160x160) head, so
                     # they use more activation memory - drop to ~16 if p2det OOMs.
IMGSZ = 640          # train at native tile resolution - do NOT downscale tiny targets
SINGLE_CLS = True    # AOT is one class ("airborne_object"); set False for a multi-class dataset

PROJECT = "train_motion"   # results land in runs/detect/<PROJECT>/<NAME>/
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
# freeze-warmup is needed (warmup was tried and hurt); the 5x lr keeps the fresh motion modules
# learning once joint training starts. These were the settings behind the best hybrid run.
MOTION_WARMUP_EPOCHS = 0
MOTION_LR_MULT = 5.0


def train_one(variant, data, epochs, batch, imgsz, device, motion_degrade, motion_degrade_min, workers=8):
    """Train a single registered variant; returns the run's best.pt path."""
    model = VARIANTS[variant]
    print(f"\n{'=' * 70}\nTraining variant '{variant}'  ({model})  motion_degrade={motion_degrade} workers={workers}\n{'=' * 70}")
    trainer = MotionDetectionTrainer(
        overrides=dict(
            model=model,
            pretrained=PRETRAINED,
            data=data,
            epochs=epochs,
            patience=PATIENCE,
            batch=0.9,
            imgsz=imgsz,
            single_cls=SINGLE_CLS,
            device=device,
            workers=workers,  # 0 => single-process loaders (avoids the val-time dataloader deadlock seen on wattn)
            project=PROJECT,
            name=variant if motion_degrade == 0 else f"{variant}-deg",
            **RECIPE,
        ),
        motion_warmup_epochs=MOTION_WARMUP_EPOCHS,
        motion_lr_mult=MOTION_LR_MULT,
        motion_degrade=motion_degrade,
        motion_degrade_min=motion_degrade_min,
    )
    trainer.train()
    best = f"{trainer.save_dir}/weights/best.pt"
    print(f"[{variant}] best weights: {best}")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="p2det", choices=[*VARIANTS, "all"],
                        help="architecture to train, or 'all' to run every variant sequentially")
    parser.add_argument("--data", default=DATA_YAML, help="dataset data.yaml")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--device", default=None, help="cuda index or 'cpu' (default: auto)")
    parser.add_argument("--motion-degrade", type=float, default=0.0,
                        help="per-sample prob of low-pass degrading the appearance stream so the model "
                             "must rely on motion (0 = off). Try 0.5 to open the motion gate.")
    parser.add_argument("--motion-degrade-min", type=float, default=0.25,
                        help="min kept-resolution fraction when degrading (lower = harsher). Default 0.25.")
    parser.add_argument("--workers", type=int, default=8,
                        help="dataloader workers; use 0 for single-process loaders if a variant deadlocks at val.")
    args = parser.parse_args()

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]

    results = {}
    for v in variants:
        results[v] = train_one(v, args.data, args.epochs, args.batch, args.imgsz, device,
                               args.motion_degrade, args.motion_degrade_min, args.workers)

    print(f"\n{'=' * 70}\nDone. Best weights per variant:")
    for v, best in results.items():
        print(f"  {v:<12} {best}")
    print("evaluate with:  test_motion.py  (point MODELS at the best.pt paths above)")


if __name__ == "__main__":
    main()
