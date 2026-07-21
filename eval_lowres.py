#!/usr/bin/env python3
"""Low-resolution robustness eval for motion detectors.

Sweeps appearance-degradation (a kept-resolution fraction) x motion-{on,off} on one or more
trained motion models and reports mAP50 / mAP50-95 for every cell. Two things it reveals:

  • how each model holds up as the appearance stream is destroyed (the low-res deployment regime -
    e.g. keep=0.25 maps a native ~6 px target to ~1.6 px, matching the base-640 data); and
  • how much of that detection the motion stream is carrying (motion-ON vs motion-OFF at each keep).

Appearance is degraded by area-downscale -> bilinear-upscale to the kept fraction (geometry
preserved, so labels are unchanged); the motion stream is left intact, or zeroed for the ablation.

Usage:
    python eval_lowres.py runs/detect/train_motion/p2det/weights/best.pt
    python eval_lowres.py A/best.pt B/best.pt --data tiled_aot_dataset/data.yaml --batch 16
    python eval_lowres.py best.pt --keeps 1.0 0.5 0.25
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from ultralytics.models.yolo.motion.val import MotionDetectionValidator


class _DegradeValidator(MotionDetectionValidator):
    """MotionDetectionValidator that degrades appearance and/or zeroes motion in preprocess."""

    keep = 1.0          # appearance kept-resolution fraction (1.0 = no degradation)
    zero_motion = False # ablate the motion stream

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        if self.zero_motion and "motion" in batch:
            batch["motion"] = torch.zeros_like(batch["motion"])
            self._motion = batch["motion"]
        if self.keep < 1.0:
            img = batch["img"]
            H, W = img.shape[2:]
            h2, w2 = max(1, round(H * self.keep)), max(1, round(W * self.keep))
            down = F.interpolate(img, size=(h2, w2), mode="area")
            batch["img"] = F.interpolate(down, size=(H, W), mode="bilinear", align_corners=False)
        return batch


def _load_model(weights: str, device: str):
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model = ckpt.get("ema") or ckpt["model"]
    return model.float().to(device).eval()


def eval_model(weights: str, data: str, keeps, batch: int, imgsz: int):
    """Print the keep x motion-{on,off} mAP table for one model."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n=== {weights} ===")
    print(f"{'keep':>5} | {'motion=ON':>18} | {'motion=OFF':>18}")
    print(f"{'':>5} | {'mAP50   mAP50-95':>18} | {'mAP50   mAP50-95':>18}")
    for keep in keeps:
        row = {}
        for zm in (False, True):
            model = _load_model(weights, device)
            val = _DegradeValidator(args=dict(data=data, imgsz=imgsz, batch=batch, half=False,
                                              plots=False, verbose=False, single_cls=True))
            val.keep = keep
            val.zero_motion = zm
            stats = val(model=model)
            row[zm] = (stats["metrics/mAP50(B)"], stats["metrics/mAP50-95(B)"])
        on, off = row[False], row[True]
        print(f"{keep:5.2f} | {on[0]:6.3f}  {on[1]:7.3f}      | {off[0]:6.3f}  {off[1]:7.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", nargs="+", help="one or more best.pt paths")
    p.add_argument("--data", default="tiled_aot_dataset/data.yaml")
    p.add_argument("--keeps", type=float, nargs="+", default=[1.0, 0.5, 0.35, 0.25],
                   help="appearance kept-resolution fractions to sweep")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()
    for w in args.weights:
        eval_model(w, args.data, args.keeps, args.batch, args.imgsz)


if __name__ == "__main__":
    main()
