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
                         A learned tanh gate (default init 0.1) keeps the block
                         near-identity at init while letting the motion pathway
                         receive gradient from the first step.
  MotionPixelFusion    – residual pixel-aligned fusion: motion features are fused
                         with backbone features at matching spatial positions.
                         Preserves the spatial specificity that global attention
                         (especially with K/V spatial reduction) dilutes - the
                         right operator for few-pixel targets at fine scales.
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

    The module is inserted after selected backbone layers.  A learnable scalar
    ``gate`` scales the residual contribution as ``tanh(gate)``.

    Gate initialisation: every gradient reaching the q/k/v/out projections (and the
    upstream MotionEncoder) is scaled by ``tanh(gate)``, so a ``gate_init`` of exactly
    ``0`` starves the whole motion pathway of gradient at init — only the gate itself
    receives signal, and the pathway can start learning only after the gate has drifted
    off zero (a slow, noisy cold start; observed empirically as gates stuck at
    ``|tanh| < 0.15`` after 100 epochs). The default ``gate_init=0.1`` keeps the block
    near-identity (~10% contribution) while letting the motion pathway train from the
    first step. Set ``gate_init=0`` to recover the exact zero-at-init behaviour when a
    pretrained checkpoint must be bit-identical at step 0.

    Args:
        curr_dim (int): Channel count of the current-frame feature map.
        motion_dim (int): Channel count of the motion feature map fed as K/V.
        num_heads (int): Desired number of attention heads; automatically
            reduced until ``curr_dim % num_heads == 0`` and
            ``curr_dim // num_heads >= 8``.
        gate_init (float): Initial value of the raw gate scalar (contribution is
            ``tanh(gate_init)``). See note above.
        sr_ratio (int): Spatial-reduction ratio applied to the motion (K/V) map before attention,
            following PVT's SRA. Full dense cross-attention is O((H*W)^2) in tokens, and at high
            resolutions (e.g. P3) that quadratic term dominates wall-clock cost far more than its
            FLOP count suggests, since the reshape/softmax/matmul chain is memory-bandwidth bound.
            Queries (current-frame) stay at full resolution so the output keeps full spatial
            fidelity; only the K/V side is average-pooled by ``sr_ratio``, cutting the attention
            matrix from ``(H*W) x (H*W)`` to ``(H*W) x (H*W/sr_ratio^2)``. ``sr_ratio=1`` (default)
            reproduces the original full dense attention exactly.

    Examples:
        >>> attn = MotionCrossAttention(curr_dim=256, motion_dim=128, num_heads=4)
        >>> x   = torch.randn(2, 256, 80, 80)
        >>> mot = torch.randn(2, 128, 80, 80)
        >>> attn(x, mot).shape
        torch.Size([2, 256, 80, 80])
    """

    def __init__(
        self, curr_dim: int, motion_dim: int, num_heads: int = 4, sr_ratio: int = 1, gate_init: float = 0.1
    ):
        super().__init__()
        # Clamp num_heads so that head_dim >= 8 and curr_dim is divisible
        while num_heads > 1 and (curr_dim % num_heads != 0 or curr_dim // num_heads < 8):
            num_heads //= 2
        self.num_heads = num_heads
        self.head_dim = curr_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.sr_ratio = sr_ratio

        self.q_proj = Conv(curr_dim, curr_dim, 1, act=False)
        self.k_proj = Conv(motion_dim, curr_dim, 1, act=False)
        self.v_proj = Conv(motion_dim, curr_dim, 1, act=False)
        self.out_proj = Conv(curr_dim, curr_dim, 1, act=False)

        # Residual contribution starts at tanh(gate_init); see class docstring for why not 0.
        self.gate = nn.Parameter(torch.full((1,), float(gate_init)))

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

        x_motion_kv = (
            F.avg_pool2d(x_motion, kernel_size=self.sr_ratio, stride=self.sr_ratio, ceil_mode=True)
            if self.sr_ratio > 1
            else x_motion
        )
        Hk, Wk = x_motion_kv.shape[2:]
        N = H * W
        Nk = Hk * Wk

        # Project and reshape: queries at full resolution, keys/values at (possibly) reduced resolution
        q = self.q_proj(x_curr).reshape(B, self.num_heads, self.head_dim, N).transpose(-2, -1)
        k = self.k_proj(x_motion_kv).reshape(B, self.num_heads, self.head_dim, Nk).transpose(-2, -1)
        v = self.v_proj(x_motion_kv).reshape(B, self.num_heads, self.head_dim, Nk).transpose(-2, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, Nk)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(-2, -1).reshape(B, C, H, W)
        out = self.out_proj(out)

        return x_curr + self.gate.tanh() * out


class MotionPixelFusion(nn.Module):
    """Residual pixel-aligned fusion of motion features into backbone features.

    Motion features are resized to the backbone feature map and fused position-by-position
    (concat -> 3x3 conv -> 1x1 projection), so a motion response at (x, y) enhances the
    backbone features at exactly (x, y). Attention-map analysis on trained models showed the
    motion encoder produces a target-specific peak at P3 (specificity 1.60 on the diagnostic
    tile) that global cross-attention with 4x K/V pooling fails to route (residual specificity
    1.00, pure background wash) - a few-pixel target occupies a fraction of one pooled K/V
    token. Pixel-aligned fusion transmits that peak losslessly at O(N) cost, which is why it
    replaces cross-attention at fine scales in the hybrid configs; cross-attention remains
    preferable at coarse scales where it demonstrably localizes (P4 query-row specificity 1.22).

    The same tanh ``gate`` (init 0.1, see MotionCrossAttention) scales the residual.

    Args:
        curr_dim (int): Channel count of the backbone feature map.
        motion_dim (int): Channel count of the motion feature map.
        gate_init (float): Initial raw gate value; contribution is ``tanh(gate_init)``.

    Examples:
        >>> fuse = MotionPixelFusion(curr_dim=128, motion_dim=128)
        >>> x = torch.randn(2, 128, 80, 80)
        >>> m = torch.randn(2, 128, 40, 40)
        >>> fuse(x, m).shape
        torch.Size([2, 128, 80, 80])
    """

    def __init__(self, curr_dim: int, motion_dim: int, gate_init: float = 0.1, per_channel_gate: bool = False):
        super().__init__()
        self.fuse = Conv(curr_dim + motion_dim, curr_dim, 3)
        self.out_proj = Conv(curr_dim, curr_dim, 1, act=False)
        # per_channel_gate: LayerScale-style vector gate (1, C, 1, 1) - lets the model pick which
        # channels carry motion instead of one global volume knob (scalar gates were observed to
        # stay near their init, so per-channel capacity is the natural next question).
        shape = (1, curr_dim, 1, 1) if per_channel_gate else (1,)
        self.gate = nn.Parameter(torch.full(shape, float(gate_init)))

    def forward(self, x_curr: torch.Tensor, x_motion: torch.Tensor) -> torch.Tensor:
        """Fuse motion cues into current-frame features via a gated pixel-aligned residual."""
        if x_motion.shape[2:] != x_curr.shape[2:]:
            x_motion = F.interpolate(x_motion, size=x_curr.shape[2:], mode="bilinear", align_corners=False)
        res = self.out_proj(self.fuse(torch.cat([x_curr, x_motion], dim=1)))
        return x_curr + self.gate.tanh() * res


class MotionFiLMFusion(nn.Module):
    """Multiplicative (FiLM-style) fusion: motion features modulate backbone features.

    ``out = x * (1 + g*gamma(m)) + g*beta(m)`` with per-position gamma/beta predicted from the
    motion features. Where MotionPixelFusion *adds* an independent motion signal, FiLM lets
    motion *amplify or suppress* the appearance features already present at each location -
    the natural operator when the target is faintly visible in appearance and motion should
    boost exactly those pixels rather than write over them.

    Args:
        curr_dim (int): Backbone feature channels.
        motion_dim (int): Motion feature channels.
        gate_init (float): Initial raw gate; contribution is ``tanh(gate_init)``.
    """

    def __init__(self, curr_dim: int, motion_dim: int, gate_init: float = 0.1):
        super().__init__()
        self.proj = Conv(motion_dim, curr_dim, 3)
        self.gamma = Conv(curr_dim, curr_dim, 1, act=False)
        self.beta = Conv(curr_dim, curr_dim, 1, act=False)
        self.gate = nn.Parameter(torch.full((1,), float(gate_init)))

    def forward(self, x_curr: torch.Tensor, x_motion: torch.Tensor) -> torch.Tensor:
        """Modulate current-frame features with motion-derived per-position scale and shift."""
        if x_motion.shape[2:] != x_curr.shape[2:]:
            x_motion = F.interpolate(x_motion, size=x_curr.shape[2:], mode="bilinear", align_corners=False)
        h = self.proj(x_motion)
        g = self.gate.tanh()
        return x_curr * (1.0 + g * self.gamma(h)) + g * self.beta(h)


class MotionSpatialGate(nn.Module):
    """Minimal fusion: motion predicts a single-channel spatial attention map that scales x.

    ``out = x * (1 + g * A(m))`` with ``A`` in [0, 1] from a two-conv head. Tests the leanest
    hypothesis - that the motion stream's entire value is a "look here" spotlight - with near
    zero parameters. If this matches richer fusions, the motion pathway is over-engineered;
    if it clearly trails, motion contributes feature *content*, not just localization.

    Args:
        curr_dim (int): Backbone feature channels (unused by the head; kept for interface parity).
        motion_dim (int): Motion feature channels.
        gate_init (float): Initial raw gate; contribution is ``tanh(gate_init)``.
    """

    def __init__(self, curr_dim: int, motion_dim: int, gate_init: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(Conv(motion_dim, motion_dim // 2, 3), nn.Conv2d(motion_dim // 2, 1, 1))
        self.gate = nn.Parameter(torch.full((1,), float(gate_init)))

    def forward(self, x_curr: torch.Tensor, x_motion: torch.Tensor) -> torch.Tensor:
        """Scale current-frame features by a motion-derived spatial attention map."""
        if x_motion.shape[2:] != x_curr.shape[2:]:
            x_motion = F.interpolate(x_motion, size=x_curr.shape[2:], mode="bilinear", align_corners=False)
        attn = self.head(x_motion).sigmoid()
        return x_curr * (1.0 + self.gate.tanh() * attn)


class _WindowAttention(nn.Module):
    """Non-overlapping window attention (Swin-style partitioning), shared by the window fusions.

    Queries come from ``xq``, keys/values from ``xkv`` (same spatial size); attention is computed
    independently inside each ``window x window`` tile, so a query can only look at its local
    neighbourhood - locality that global cross-attention lacks and that few-pixel targets need.
    Cost is O(N * window^2) instead of O(N^2).
    """

    def __init__(self, q_dim: int, kv_dim: int, num_heads: int = 4, window: int = 8):
        super().__init__()
        while num_heads > 1 and (q_dim % num_heads != 0 or q_dim // num_heads < 8):
            num_heads //= 2
        self.heads = num_heads
        self.head_dim = q_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.window = window
        self.q_proj = Conv(q_dim, q_dim, 1, act=False)
        self.k_proj = Conv(kv_dim, q_dim, 1, act=False)
        self.v_proj = Conv(kv_dim, q_dim, 1, act=False)
        self.out_proj = Conv(q_dim, q_dim, 1, act=False)

    def _windows(self, t: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
        """(B, C, H, W) -> (B*nWin, win*win, C) after bottom/right zero-padding."""
        t = F.pad(t, (0, pad_w, 0, pad_h))
        B, C, H, W = t.shape
        w = self.window
        t = t.reshape(B, C, H // w, w, W // w, w).permute(0, 2, 4, 3, 5, 1)
        return t.reshape(B * (H // w) * (W // w), w * w, C)

    def forward(self, xq: torch.Tensor, xkv: torch.Tensor) -> torch.Tensor:
        B, C, H, W = xq.shape
        w = self.window
        pad_h, pad_w = (w - H % w) % w, (w - W % w) % w
        q = self._windows(self.q_proj(xq), pad_h, pad_w)
        k = self._windows(self.k_proj(xkv), pad_h, pad_w)
        v = self._windows(self.v_proj(xkv), pad_h, pad_w)
        n, t, _ = q.shape
        q = q.reshape(n, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(n, t, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(n, t, self.heads, self.head_dim).transpose(1, 2)
        out = ((q @ k.transpose(-2, -1)) * self.scale).softmax(-1) @ v
        out = out.transpose(1, 2).reshape(n, t, C)
        # windows -> feature map
        Hp, Wp = H + pad_h, W + pad_w
        out = out.reshape(B, Hp // w, Wp // w, w, w, C).permute(0, 5, 1, 3, 2, 4)
        out = out.reshape(B, C, Hp, Wp)[:, :, :H, :W]
        return self.out_proj(out)


class MotionWindowCrossAttention(nn.Module):
    """Windowed cross-attention: queries attend to motion K/V inside local 8x8 windows only.

    Distinguishes *why* global cross-attention failed at P3: if locality was the missing
    ingredient (a few-pixel target's motion token drowns in a global softmax / SR pooling),
    windowed attention should recover pixel-fusion-level performance while keeping attention's
    content-adaptive routing. If it still trails pixel fusion, attention itself is the wrong
    operator at fine scales on this data.
    """

    def __init__(self, curr_dim: int, motion_dim: int, num_heads: int = 4, window: int = 8, gate_init: float = 0.1):
        super().__init__()
        self.attn = _WindowAttention(curr_dim, motion_dim, num_heads, window)
        self.gate = nn.Parameter(torch.full((1,), float(gate_init)))

    def forward(self, x_curr: torch.Tensor, x_motion: torch.Tensor) -> torch.Tensor:
        """Gated residual windowed cross-attention from current-frame queries to motion K/V."""
        if x_motion.shape[2:] != x_curr.shape[2:]:
            x_motion = F.interpolate(x_motion, size=x_curr.shape[2:], mode="bilinear", align_corners=False)
        return x_curr + self.gate.tanh() * self.attn(x_curr, x_motion)


class MotionPixelFusionSA(nn.Module):
    """Pixel-aligned fusion followed by windowed self-attention over the fused features.

    Two-stage: (1) MotionPixelFusion plants the motion signal at exact positions; (2) a light
    window self-attention over the *fused* map lets nearby positions reason jointly about the
    planted signal (e.g. suppress isolated noise responses, reinforce blob + context patterns).
    Each stage has its own tanh gate, so training can turn off either half independently -
    the trained gate pair is itself the diagnostic for whether post-fusion attention earns
    its cost.
    """

    def __init__(self, curr_dim: int, motion_dim: int, num_heads: int = 4, window: int = 8, gate_init: float = 0.1):
        super().__init__()
        self.pixel = MotionPixelFusion(curr_dim, motion_dim, gate_init=gate_init)
        self.sa = _WindowAttention(curr_dim, curr_dim, num_heads, window)
        self.gate = nn.Parameter(torch.full((1,), float(gate_init)))  # gate for the SA stage

    def forward(self, x_curr: torch.Tensor, x_motion: torch.Tensor) -> torch.Tensor:
        """Pixel-fuse motion, then refine with gated windowed self-attention."""
        fused = self.pixel(x_curr, x_motion)
        return fused + self.gate.tanh() * self.sa(fused, fused)
