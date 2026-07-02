#!/usr/bin/env python3
"""Train motion cross-attention variants on the real AOT dataset and compare accuracy.

Complements benchmark_motion_variants.py (speed/params only, random weights) with actual
detection quality (mAP50, mAP50-95) after training on real ground-truth boxes.

Each config runs in its own subprocess (sweep_one.py) rather than looping in-process: running
multiple configs sequentially in one long-lived process caused a silent hang right after a
validation pass completed (no exception, no traceback), while isolated single-config runs of the
same length completed cleanly every time. Per-config subprocess isolation avoids whatever state
carries over between successive trainer instantiations, and a timeout means one stuck config can't
block the rest of the sweep.

Scope note: AOT objects are extremely small in pixel terms (median ~8px at imgsz=1280), and full
dense cross-attention at P3 is O((H*W)^2) - at high enough resolution to make these objects
representable, dense attention OOMs even at batch=1 on a 16GB GPU. So dense configs run at
imgsz=640 (the highest they fit at reasonable batch) while P3-SR configs run at imgsz=960 (SR
pooling buys headroom for higher resolution). `fraction` caps images/epoch to keep wall time
tractable for a relative comparison (not a fully-converged production run).

Usage:
    python sweep.py
    python sweep.py --epochs 15 --fraction 0.12 --timeout 3600
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# (label, yaml cfg, imgsz, batch) - each config trains at its own practical resolution ceiling
CONFIGS = [
    ("YOLOv8s-motion (3-layer, dense)", "yolov8s-motion.yaml", 640, 4),
    ("YOLOv8s-motion (3-layer, P3 SR)", "yolov8s-motion-p3sr.yaml", 960, 8),
    ("YOLO26s-motion (3-layer, dense)", "yolo26s-motion.yaml", 640, 4),
    ("YOLO26s-motion (3-layer, P3 SR)", "yolo26s-motion-p3sr.yaml", 960, 8),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train and compare motion cross-attention variants")
    parser.add_argument("--data", default="motion_aot_dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fraction", type=float, default=0.12, help="Fraction of train images per epoch")
    parser.add_argument("--project", default="sweep")
    parser.add_argument("--motion_warmup_epochs", type=int, default=3)
    parser.add_argument("--motion_lr_mult", type=float, default=5.0)
    parser.add_argument("--timeout", type=int, default=3600, help="Per-config subprocess timeout (s)")
    parser.add_argument("--out", default="sweep_results.json")
    return parser.parse_args()


def run_one(label: str, cfg: str, imgsz: int, batch: int, args) -> dict:
    print(f"\n{'=' * 70}\n{label}: {cfg} (imgsz={imgsz}, batch={batch})\n{'=' * 70}", flush=True)

    out_file = Path(f"sweep_result_{cfg.replace('.yaml', '')}.json")
    out_file.unlink(missing_ok=True)

    cmd = [
        sys.executable,
        "sweep_one.py",
        "--label", label,
        "--cfg", cfg,
        "--imgsz", str(imgsz),
        "--batch", str(batch),
        "--data", args.data,
        "--epochs", str(args.epochs),
        "--fraction", str(args.fraction),
        "--project", args.project,
        "--motion_warmup_epochs", str(args.motion_warmup_epochs),
        "--motion_lr_mult", str(args.motion_lr_mult),
        "--out", str(out_file),
    ]

    try:
        subprocess.run(cmd, timeout=args.timeout, check=True)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {label} exceeded {args.timeout}s, killed", flush=True)
        return {"label": label, "cfg": cfg, "imgsz": imgsz, "batch": batch, "error": f"timeout after {args.timeout}s"}
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {label}: subprocess exited {e.returncode}", flush=True)
        return {"label": label, "cfg": cfg, "imgsz": imgsz, "batch": batch, "error": f"exit code {e.returncode}"}

    if not out_file.exists():
        return {"label": label, "cfg": cfg, "imgsz": imgsz, "batch": batch, "error": "no result file written"}
    return json.loads(out_file.read_text())


def main():
    args = parse_args()
    results = []
    out_path = Path(args.out)

    for label, cfg, imgsz, batch in CONFIGS:
        result = run_one(label, cfg, imgsz, batch, args)
        results.append(result)
        out_path.write_text(json.dumps(results, indent=2))  # incremental save after each config

    print("\n\n" + "=" * 100)
    print("SWEEP SUMMARY")
    print("=" * 100)
    header = (
        f"{'Variant':<32}{'imgsz':>7}{'Params':>12}{'mAP50':>10}{'mAP50-95':>10}"
        f"{'Precision':>11}{'Recall':>9}"
    )
    print(header)
    for r in results:
        if "error" in r:
            print(f"{r['label']:<32}FAILED: {r['error']}")
            continue
        map50 = f"{r['mAP50']:.3f}" if r["mAP50"] is not None else "n/a"
        map5095 = f"{r['mAP50-95']:.3f}" if r["mAP50-95"] is not None else "n/a"
        prec = f"{r['precision']:.3f}" if r["precision"] is not None else "n/a"
        rec = f"{r['recall']:.3f}" if r["recall"] is not None else "n/a"
        print(
            f"{r['label']:<32}{r['imgsz']:>7}{r['params']:>12,}{map50:>10}{map5095:>10}{prec:>11}{rec:>9}"
        )

    print(f"\nFull results saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
