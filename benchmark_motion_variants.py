#!/usr/bin/env python3
"""Compare speed/size of the 3-layer vs 2-layer motion cross-attention YOLO26s variants.

Usage:
    python benchmark_motion_variants.py
    python benchmark_motion_variants.py --imgsz 416 --batch 8 --iters 50
"""

import argparse
import time

import torch

from ultralytics.nn.tasks import MotionDetectionModel
from ultralytics.utils.torch_utils import model_info, select_device


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark motion cross-attention variants")
    parser.add_argument("--imgsz", type=int, default=416, help="Square input size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for timing")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations (after warmup)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--nc", type=int, default=1, help="Number of classes")
    return parser.parse_args()


def build(cfg: str, nc: int, device: torch.device):
    model = MotionDetectionModel(cfg, ch=3, nc=nc, verbose=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def time_forward(model, imgsz: int, batch: int, iters: int, warmup: int, device: torch.device) -> float:
    """Return mean forward-pass latency in milliseconds."""
    x = torch.randn(batch, 3, imgsz, imgsz, device=device)
    motion = torch.randn(batch, 3, imgsz, imgsz, device=device)

    for _ in range(warmup):
        model(x, motion=motion)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        model(x, motion=motion)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iters * 1000  # ms per forward pass (whole batch)


def main():
    args = parse_args()
    device = select_device("")
    print(f"Device: {device}  |  imgsz={args.imgsz}  batch={args.batch}  iters={args.iters}")
    print("=" * 70)

    variants = [
        ("3-layer (P3+P4+P5)", "yolo26s-motion.yaml"),
        ("2-layer (P3+P4)", "yolo26s-motion-2attn.yaml"),
    ]

    results = []
    for label, cfg in variants:
        print(f"\n--- {label}: {cfg} ---")
        model = build(cfg, args.nc, device)
        n_l, n_p, n_g, flops = model_info(model, detailed=False, verbose=True, imgsz=args.imgsz)
        ms = time_forward(model, args.imgsz, args.batch, args.iters, args.warmup, device)
        fps = 1000 / ms * args.batch
        results.append((label, n_p, flops, ms, fps))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'Variant':<22}{'Params':>12}{'GFLOPs':>10}{'ms/batch':>12}{'img/s':>10}"
    print(header)
    for label, n_p, flops, ms, fps in results:
        print(f"{label:<22}{n_p:>12,}{flops:>10.1f}{ms:>12.2f}{fps:>10.1f}")

    if len(results) == 2:
        (_, p3, _, ms3, _), (_, p2, _, ms2, _) = results  # 3-layer, then 2-layer
        print(f"\n3-layer / 2-layer ratio: {p3 / p2:.2f}x params, {ms3 / ms2:.2f}x latency")


if __name__ == "__main__":
    main()
