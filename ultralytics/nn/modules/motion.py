# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Motion cross-attention modules for dual-stream video detection.

These modules let a YOLO model jointly process the current video frame and a
frame-difference (motion) image, fusing motion cues into backbone features via
cross-attention before the detection head.

Dataset convention
------------------
Current frames : <dataset>/images/<split>/<name>.jpg
Motion diffs   : <dataset>/motion/<split>/<name>.jpg  (identical filenames)
Missing motion : falls back to a zero tensor automatically.

Architecture overview
---------------------
  MotionEncoder      – lightweight 4-stage CNN; produces one feature map per
                       stride (x2, x4, x8, x16) from the motion image.
  MotionCrossAttention – residual cross-attention block: current-frame features
                         are the queries, motion features supply keys and values.
                         A learned gate starts at zero so the block contributes
                         nothing at init and activates gradually during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class MotionEncoder(nn.Module):
    """Lightweight CNN encoder that extracts multi-scale features from a motion-difference image.

    Each stage halves the spatial resolution and increases the channel count,
    mirroring the stride progression of the YOLOv8 backbone.

    Args:
        in_channels (int): Channels in the motion image (3 for RGB / absolute diff).
        dims (tuple[int, ...]): Output channels at each downsampling stage.
            Default ``(32, 64, 128, 256)`` gives four scales (P1/2 … P4/16).

    Examples:
        >>> enc = MotionEncoder(in_channels=3, dims=(32, 64, 128, 256))
        >>> feats = enc(torch.zeros(2, 3, 640, 640))
        >>> [f.shape for f in feats]
        [torch.Size([2, 32, 320, 320]), torch.Size([2, 64, 160, 160]),
         torch.Size([2, 128, 80, 80]), torch.Size([2, 256, 40, 40])]
    """

    def __init__(self, in_channels: int = 3, dims: tuple = (32, 64, 128, 256)):
        super().__init__()
        self.stages = nn.ModuleList()
        c = in_channels
        for d in dims:
            self.stages.append(nn.Sequential(Conv(c, d, 3, 2), Conv(d, d, 3, 1)))
            c = d

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return a list of feature maps, one per downsampling stage."""
        out: list[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        return out


class MotionCrossAttention(nn.Module):
    """Cross-attention that lets current-frame features (Q) attend to motion features (K, V).

    The module is inserted after selected backbone layers.  A learnable ``gate``
    parameter initialised to ``0`` ensures the residual contribution starts at
    zero (``tanh(0) = 0``) and grows only as gradient updates push it away from
    zero, giving stable early training behaviour.

    Args:
        curr_dim (int): Channel count of the current-frame feature map.
        motion_dim (int): Channel count of the motion feature map fed as K/V.
        num_heads (int): Desired number of attention heads; automatically
            reduced until ``curr_dim % num_heads == 0`` and
            ``curr_dim // num_heads >= 8``.

    Examples:
        >>> attn = MotionCrossAttention(curr_dim=256, motion_dim=128, num_heads=4)
        >>> x   = torch.randn(2, 256, 80, 80)
        >>> mot = torch.randn(2, 128, 80, 80)
        >>> attn(x, mot).shape
        torch.Size([2, 256, 80, 80])
    """

    def __init__(self, curr_dim: int, motion_dim: int, num_heads: int = 4):
        super().__init__()
        # Clamp num_heads so that head_dim >= 8 and curr_dim is divisible
        while num_heads > 1 and (curr_dim % num_heads != 0 or curr_dim // num_heads < 8):
            num_heads //= 2
        self.num_heads = num_heads
        self.head_dim = curr_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = Conv(curr_dim, curr_dim, 1, act=False)
        self.k_proj = Conv(motion_dim, curr_dim, 1, act=False)
        self.v_proj = Conv(motion_dim, curr_dim, 1, act=False)
        self.out_proj = Conv(curr_dim, curr_dim, 1, act=False)

        # Gate starts at 0 → tanh(0) = 0 → no contribution at init
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x_curr: torch.Tensor, x_motion: torch.Tensor) -> torch.Tensor:
        """Fuse motion cues into current-frame features via residual cross-attention.

        Args:
            x_curr:   (B, C, H, W)   current-frame backbone features.
            x_motion: (B, Cm, Hm, Wm) motion features; resized to (H, W) if needed.

        Returns:
            (B, C, H, W) motion-enhanced features.
        """
        B, C, H, W = x_curr.shape

        if x_motion.shape[2:] != (H, W):
            x_motion = F.interpolate(x_motion, size=(H, W), mode="bilinear", align_corners=False)

        N = H * W

        # Project and reshape to (B, heads, N, head_dim)
        q = self.q_proj(x_curr).reshape(B, self.num_heads, self.head_dim, N).transpose(-2, -1)
        k = self.k_proj(x_motion).reshape(B, self.num_heads, self.head_dim, N).transpose(-2, -1)
        v = self.v_proj(x_motion).reshape(B, self.num_heads, self.head_dim, N).transpose(-2, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(-2, -1).reshape(B, C, H, W)
        out = self.out_proj(out)

        return x_curr + self.gate.tanh() * out
