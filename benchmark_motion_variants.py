#!/usr/bin/env python3
"""Compare speed/size of base YOLOv8s/YOLO26s against their 3-layer and 2-layer motion
cross-attention variants.

Usage:
    python benchmark_motion_variants.py
    python benchmark_motion_variants.py --imgsz 416 --batch 8 --iters 50
"""

import argparse
import time

import torch

from ultralytics.nn.tasks import DetectionModel, MotionDetectionModel
from ultralytics.utils.torch_utils import model_info, select_device

# (label, yaml cfg, is_motion) - is_motion selects the dual-stream build/forward path
VARIANTS = [
    ("YOLOv8s (base)", "yolov8s.yaml", False),
    ("YOLOv8s-motion (3-layer)", "yolov8s-motion.yaml", True),
    ("YOLOv8s-motion (2-layer)", "yolov8s-motion-2attn.yaml", True),
    ("YOLO26s (base)", "yolo26s.yaml", False),
    ("YOLO26s-motion (3-layer)", "yolo26s-motion.yaml", True),
    ("YOLO26s-motion (2-layer)", "yolo26s-motion-2attn.yaml", True),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark base vs motion cross-attention variants")
    parser.add_argument("--imgsz", type=int, default=416, help="Square input size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for timing")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations (after warmup)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--nc", type=int, default=1, help="Number of classes")
    return parser.parse_args()


def build(cfg: str, nc: int, is_motion: bool, device: torch.device):
    cls = MotionDetectionModel if is_motion else DetectionModel
    model = cls(cfg, ch=3, nc=nc, verbose=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def time_forward(
    model, imgsz: int, batch: int, iters: int, warmup: int, is_motion: bool, device: torch.device
) -> float:
    """Return mean forward-pass latency in milliseconds for one full batch."""
    x = torch.randn(batch, 3, imgsz, imgsz, device=device)
    motion = torch.randn(batch, 3, imgsz, imgsz, device=device) if is_motion else None

    def step():
        model(x, motion=motion) if is_motion else model(x)

    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iters * 1000  # ms per forward pass (whole batch)


def main():
    args = parse_args()
    device = select_device("")
    print(f"Device: {device}  |  imgsz={args.imgsz}  batch={args.batch}  iters={args.iters}")
    print("=" * 70)

    results = []
    for label, cfg, is_motion in VARIANTS:
        print(f"\n--- {label}: {cfg} ---")
        model = build(cfg, args.nc, is_motion, device)
        n_l, n_p, n_g, flops = model_info(model, detailed=False, verbose=True, imgsz=args.imgsz)
        ms = time_forward(model, args.imgsz, args.batch, args.iters, args.warmup, is_motion, device)
        fps = 1000 / ms * args.batch
        results.append((label, n_p, flops, ms, fps))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'Variant':<28}{'Params':>12}{'GFLOPs':>10}{'ms/batch':>12}{'img/s':>10}"
    print(header)
    for label, n_p, flops, ms, fps in results:
        print(f"{label:<28}{n_p:>12,}{flops:>10.1f}{ms:>12.2f}{fps:>10.1f}")

    print()
    by_label = {label: (n_p, ms) for label, n_p, _, ms, _ in results}
    for family in ("YOLOv8s", "YOLO26s"):
        base = f"{family} (base)"
        three = f"{family}-motion (3-layer)"
        two = f"{family}-motion (2-layer)"
        if base in by_label and three in by_label and two in by_label:
            p0, ms0 = by_label[base]
            p3, ms3 = by_label[three]
            p2, ms2 = by_label[two]
            print(
                f"{family}: 3-layer adds {ms3 / ms0:.2f}x latency / {p3 / p0:.2f}x params over base; "
                f"2-layer adds {ms2 / ms0:.2f}x latency / {p2 / p0:.2f}x params over base "
                f"({ms3 / ms2:.2f}x latency, {p3 / p2:.2f}x params, 3-layer vs 2-layer)"
            )


if __name__ == "__main__":
    main()
