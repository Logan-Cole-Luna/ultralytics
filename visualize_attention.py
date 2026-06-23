#!/usr/bin/env python3
"""
Visualize motion cross-attention maps for two input images.

Usage:
    python visualize_attention.py img1.jpg img2.jpg
    python visualize_attention.py img1.jpg img2.jpg --model runs/motion_aot/train/weights/best.pt
    python visualize_attention.py img1.jpg img2.jpg --out attention.png
    
    
# With trained weights
python visualize_attention.py motion_aot_dataset/images/val/frame_000000.jpg motion_aot_dataset/images/val/frame_000005.jpg --model "runs/detect/motion_aot/train/weights/best.pt"

# Without weights (random init, shows the mechanism still works)
python visualize_attention.py img1.jpg img2.jpg

# Query a specific point (x,y in [0,1]) to see what that position attends to in motion
python visualize_attention.py img1.jpg img2.jpg --query-point 0.5 0.5

"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize motion cross-attention")
    parser.add_argument("img1", help="Frame t-1 (previous frame)")
    parser.add_argument("img2", help="Frame t (current frame)")
    parser.add_argument("--model", default=None, help="Path to trained .pt weights (optional)")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size")
    parser.add_argument("--out", default="attention_vis.png", help="Output image path")
    parser.add_argument("--query-point", nargs=2, type=float, default=None,
                        metavar=("X", "Y"),
                        help="Normalized [0,1] query point to show attended-to pixels. "
                             "Defaults to showing the global average attention map.")
    return parser.parse_args()


def load_image(path: str, size: int) -> tuple[np.ndarray, torch.Tensor]:
    """Read BGR image, resize to square, return (bgr_uint8, rgb_float_tensor)."""
    bgr = cv2.imread(path)
    if bgr is None:
        print(f"Cannot read image: {path}")
        sys.exit(1)
    bgr = cv2.resize(bgr, (size, size))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3, H, W)
    return rgb, t


def patch_cross_attention(model):
    """Monkey-patch MotionCrossAttention.forward to capture attention weights."""
    from ultralytics.nn.modules.motion import MotionCrossAttention

    captured = []  # list of (layer_idx, attn_tensor, H, W)

    original_forward = MotionCrossAttention.forward

    def patched_forward(self, x_curr, x_motion):
        B, C, H, W = x_curr.shape

        if x_motion.shape[2:] != (H, W):
            x_motion = F.interpolate(x_motion, size=(H, W), mode="bilinear", align_corners=False)

        N = H * W

        q = self.q_proj(x_curr).reshape(B, self.num_heads, self.head_dim, N).transpose(-2, -1)
        k = self.k_proj(x_motion).reshape(B, self.num_heads, self.head_dim, N).transpose(-2, -1)
        v = self.v_proj(x_motion).reshape(B, self.num_heads, self.head_dim, N).transpose(-2, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)

        # Store detached attention for visualization
        captured.append({
            "attn": attn.detach().cpu(),  # (B, heads, N_q, N_k)
            "H": H,
            "W": W,
        })

        out = (attn @ v).transpose(-2, -1).reshape(B, C, H, W)
        out = self.out_proj(out)
        return x_curr + self.gate.tanh() * out

    MotionCrossAttention.forward = patched_forward
    return captured, original_forward, MotionCrossAttention


def attention_to_heatmap(attn: torch.Tensor, H: int, W: int,
                          query_point: tuple | None = None) -> np.ndarray:
    """
    Convert raw attention tensor to a (H, W) heatmap.

    attn: (B, heads, N_q, N_k) where N = H*W
    If query_point (cx, cy) in [0,1]^2 is given, show the row of attn for that
    query token. Otherwise, average over all query tokens to get a global map of
    which motion positions are most attended to.
    """
    # Take first batch, average over heads -> (N_q, N_k)
    a = attn[0].mean(dim=0)  # (N_q, N_k)

    if query_point is not None:
        cx, cy = query_point
        qi = int(cy * (H - 1)) * W + int(cx * (W - 1))
        qi = max(0, min(qi, H * W - 1))
        row = a[qi]  # (N_k,) - attention from this query to all motion tokens
        heatmap = row.reshape(H, W).numpy()
    else:
        # Average attention received by each key/value token
        heatmap = a.mean(dim=0).reshape(H, W).numpy()  # (H, W)

    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a [0,1] heatmap onto an RGB image using jet colormap."""
    h_resized = cv2.resize(heatmap.astype(np.float32), (image_rgb.shape[1], image_rgb.shape[0]))
    cmap = plt.get_cmap("jet")
    colored = (cmap(h_resized)[:, :, :3] * 255).astype(np.uint8)
    blended = (alpha * colored + (1 - alpha) * image_rgb).astype(np.uint8)
    return blended


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load images
    # ------------------------------------------------------------------
    print(f"Loading images...")
    img1_rgb, img1_t = load_image(args.img1, args.imgsz)  # previous frame
    img2_rgb, img2_t = load_image(args.img2, args.imgsz)  # current frame

    # Motion = |current - previous|
    motion_t = (img2_t - img1_t).abs()
    motion_rgb = (motion_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    print("Building model...")
    from ultralytics.nn.tasks import MotionDetectionModel

    model = MotionDetectionModel("yolov8-motion.yaml", nc=1)

    if args.model is not None:
        weights_path = Path(args.model)
        if not weights_path.exists():
            print(f"Weights not found at {weights_path}. Running with random weights.")
        else:
            ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt)
            if hasattr(state, "state_dict"):
                state = state.state_dict()
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)}")
            print(f"  Loaded weights from {weights_path}")
    else:
        print("  No weights provided - using random initialization")

    model.eval()

    # ------------------------------------------------------------------
    # Patch cross-attention to capture attention maps
    # ------------------------------------------------------------------
    captured, original_fwd, MCA_cls = patch_cross_attention(model)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    print("Running forward pass...")
    img_batch = img2_t.unsqueeze(0)       # (1, 3, H, W) - current frame
    motion_batch = motion_t.unsqueeze(0)  # (1, 3, H, W)

    with torch.no_grad():
        _ = model(img_batch, motion=motion_batch)

    # Restore original forward
    MCA_cls.forward = original_fwd

    if not captured:
        print("No cross-attention layers found in model. Check model configuration.")
        sys.exit(1)

    print(f"Captured attention from {len(captured)} cross-attention layer(s)")

    # ------------------------------------------------------------------
    # Visualize
    # ------------------------------------------------------------------
    query_point = tuple(args.query_point) if args.query_point else None
    n_layers = len(captured)

    # Layout: row 0 = inputs (img1, motion, img2), rows 1..n = per-layer attention
    fig = plt.figure(figsize=(14, 4 + 4 * n_layers))
    gs = gridspec.GridSpec(1 + n_layers, 3, figure=fig, hspace=0.4, wspace=0.1)

    # Row 0: input images
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img1_rgb)
    ax.set_title("Frame t-1 (previous)")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(motion_rgb)
    ax.set_title("Motion diff |t - (t-1)|")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(img2_rgb)
    ax.set_title("Frame t (current)")
    ax.axis("off")

    for i, entry in enumerate(captured):
        attn = entry["attn"]
        H, W = entry["H"], entry["W"]

        heatmap = attention_to_heatmap(attn, H, W, query_point)

        # Left: heatmap alone (upsampled)
        ax = fig.add_subplot(gs[i + 1, 0])
        hm_up = cv2.resize(heatmap, (args.imgsz, args.imgsz))
        ax.imshow(hm_up, cmap="jet", vmin=0, vmax=1)
        ax.set_title(f"Layer {i+1} attention ({H}x{W})")
        ax.axis("off")
        plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)

        # Middle: attention overlaid on motion image
        ax = fig.add_subplot(gs[i + 1, 1])
        overlay = overlay_heatmap(motion_rgb, heatmap)
        ax.imshow(overlay)
        ax.set_title(f"Layer {i+1} on motion")
        ax.axis("off")

        # Right: attention overlaid on current frame
        ax = fig.add_subplot(gs[i + 1, 2])
        overlay_curr = overlay_heatmap(img2_rgb, heatmap)
        ax.imshow(overlay_curr)
        mode = f"query ({args.query_point[0]:.2f},{args.query_point[1]:.2f})" if query_point else "avg over queries"
        ax.set_title(f"Layer {i+1} on current ({mode})")
        ax.axis("off")

        gate_val = None
        for module in model.modules():
            from ultralytics.nn.modules.motion import MotionCrossAttention
            if isinstance(module, MotionCrossAttention):
                gate_val = module.gate.item()
                break

    gate_str = f"  Gate value: {gate_val:.4f}  (tanh={np.tanh(gate_val):.4f})" if gate_val is not None else ""
    title = f"Motion Cross-Attention Visualization{gate_str}"
    if query_point:
        title += f"\nQuery point: ({query_point[0]:.2f}, {query_point[1]:.2f})"
    else:
        title += "\nShowing: average attention received by each motion token"
    fig.suptitle(title, fontsize=12)

    out_path = Path(args.out)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
