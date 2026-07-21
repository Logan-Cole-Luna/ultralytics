# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Motion-aware detection trainer.

Extends DetectionTrainer to:
  • Build MotionYOLODataset (loads current frame + motion-diff image pairs).
  • Instantiate MotionDetectionModel instead of DetectionModel.
  • Normalise the 'motion' tensor in preprocess_batch alongside 'img'.
"""

from __future__ import annotations

import math
import random
from copy import copy
from typing import Any

import torch
import torch.nn as nn

from ultralytics.data import build_dataloader
from ultralytics.data.dataset import MotionYOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import MotionDetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


def build_motion_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32):
    """Build a MotionYOLODataset for training or validation.

    Args:
        cfg: Training configuration namespace.
        img_path (str): Path to the image directory.
        batch (int): Batch size (used for rect-mode padding).
        data (dict): Dataset configuration dict (names, nc, etc.).
        mode (str): ``'train'`` or ``'val'``.
        rect (bool): Enable rectangular batching.
        stride (int): Model stride for padding alignment.

    Returns:
        (MotionYOLODataset): Dataset object ready for use in a DataLoader.
    """
    pad = 0.0 if mode == "train" else 0.5
    fraction = cfg.fraction if mode == "train" else 1.0
    return MotionYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=pad,
        prefix=f"{mode}: ",
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=fraction,
    )


class MotionDetectionTrainer(DetectionTrainer):
    """Trainer for motion-aware YOLOv8 detection.

    Differences from DetectionTrainer:
      - Uses MotionYOLODataset so every batch includes a ``motion`` key.
      - Returns a MotionDetectionModel from get_model().
      - preprocess_batch() additionally normalises the motion tensor.

    Examples:
        >>> args = dict(model="yolov8-motion.yaml", data="coco8.yaml", epochs=10)
        >>> trainer = MotionDetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
        motion_warmup_epochs: int = 3,
        motion_lr_mult: float = 5.0,
        motion_degrade: float = 0.0,
        motion_degrade_min: float = 0.25,
    ):
        """Initialise the motion-aware trainer.

        Args:
            motion_warmup_epochs (int): Number of initial epochs during which the pretrained backbone/head
                are frozen and only ``motion_encoder`` + ``cross_attns`` train, so the gated cross-attention
                can find a useful direction before the (typically small) dataset is fine-tuned jointly.
                A well-converged pretrained backbone otherwise gives these newly-initialised layers little
                gradient signal to work with. Set to 0 to disable.
            motion_lr_mult (float): Learning-rate multiplier applied to ``motion_encoder`` + ``cross_attns``
                parameters (including the attention ``gate``) relative to the rest of the network, and with
                weight decay disabled for this group. Counteracts gradient starvation once warmup ends and
                the backbone unfreezes.
            motion_degrade (float): Probability, per training sample, of low-pass degrading the *appearance*
                stream (downscale->upscale, motion left intact) so few-pixel targets vanish from appearance
                and only the motion stream can locate them. On native-resolution tiles appearance already
                resolves the target, so motion is redundant and the residual gate closes (the grad diagnostic
                shows the gate is *not* starved - it receives a strong signal to decrease). Making appearance
                fail on a fraction of samples removes that redundancy, so the gate opens, and matches the
                low-resolution deployment regime the model must generalise to. ``0`` disables (default).
            motion_degrade_min (float): Minimum kept-resolution fraction when degrading; the per-sample factor
                is drawn uniformly from ``[motion_degrade_min, 0.9]``. Lower = harsher appearance loss.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.motion_warmup_epochs = motion_warmup_epochs
        self.motion_lr_mult = motion_lr_mult
        self.motion_degrade = motion_degrade
        self.motion_degrade_min = motion_degrade_min
        self._grad_accum: dict[str, list[float]] = {}  # per-epoch motion-pathway grad-norm sums
        self._grad_count: int = 0
        if self.motion_warmup_epochs > 0:
            self.add_callback("on_train_epoch_start", self._motion_warmup_step)
        self.add_callback("on_fit_epoch_end", self._log_gates)

    # ------------------------------------------------------------------
    # Gradient-starvation diagnostic
    # ------------------------------------------------------------------

    def optimizer_step(self):
        """Record motion-pathway gradient norms before the parent optimizer step zeroes them.

        The gate-collapse question ("is motion ignored because the signal is useless, or because the
        pathway is gradient-starved?") is only answerable from the gradients: if the motion encoder's
        relative update magnitude (|grad|/|weight|) is far below the backbone's, the pathway is
        starved (small gate -> small gradient into the encoder -> encoder stays noisy -> gate stays
        small), which is a training pathology to fix, not a signal verdict. See gate SNR analysis.
        """
        self._accumulate_grad_norms()
        super().optimizer_step()

    def _accumulate_grad_norms(self):
        """Sum unscaled grad norms and grad/weight ratios per pathway group for the current epoch."""
        model = unwrap_model(self.model)
        if not hasattr(model, "motion_encoder"):
            return
        scaler = getattr(self, "scaler", None)
        scale = float(scaler.get_scale()) if scaler is not None and scaler.is_enabled() else 1.0
        groups: dict[str, list] = {"encoder": [], "fusion": [], "gate": [], "backbone": []}
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            if n.startswith("motion_encoder"):
                groups["encoder"].append(p)
            elif n.startswith("cross_attns"):
                groups["gate" if n.split(".")[-1] == "gate" else "fusion"].append(p)
            elif n.startswith("model."):
                groups["backbone"].append(p)
        # Compute all norms first; skip the whole step if any is non-finite (AMP scaler warmup emits
        # inf grads on skipped steps - counting those would poison the epoch average).
        step = {}
        for k, ps in groups.items():
            if not ps:
                continue
            gnorm = torch.norm(torch.stack([p.grad.detach().float().norm() for p in ps])).item() / scale
            wnorm = torch.norm(torch.stack([p.detach().float().norm() for p in ps])).item()
            if not math.isfinite(gnorm):
                return
            step[k] = (gnorm, gnorm / (wnorm + 1e-12))
        for k, (g, r) in step.items():
            acc = self._grad_accum.setdefault(k, [0.0, 0.0])
            acc[0] += g
            acc[1] += r
        self._grad_count += 1

    def _log_gates(self, *_trainer):
        """Log each cross-attention gate's tanh contribution and append it to save_dir/gates.csv.

        The gate trajectory is the primary diagnostic for whether the motion stream is being
        used: gates stuck near their init mean the model found no exploitable motion signal
        (or the pathway is gradient-starved - see MotionCrossAttention.gate_init).
        """
        model = unwrap_model(self.model)
        if not hasattr(model, "cross_attns"):
            return
        # Mean tanh over every gate parameter in the module: covers scalar gates, per-channel
        # vector gates (pixel_cg), and multi-gate fusions (pixel_sa has a pixel + an SA gate).
        vals = []
        for attn in model.cross_attns:
            gates = [p for n, p in attn.named_parameters() if n.split(".")[-1] == "gate"]
            vals.append(float(torch.stack([g.tanh().mean() for g in gates]).mean().detach()))
        LOGGER.info("motion gates tanh(gate): " + "  ".join(f"[{i}] {v:+.4f}" for i, v in enumerate(vals)))
        csv_path = self.save_dir / "gates.csv"
        if not csv_path.exists():
            csv_path.write_text("epoch," + ",".join(f"gate{i}" for i in range(len(vals))) + "\n")
        with open(csv_path, "a") as f:
            f.write(f"{self.epoch}," + ",".join(f"{v:.6f}" for v in vals) + "\n")

        # Gradient-starvation diagnostic: compare the motion encoder's relative update magnitude
        # (|grad|/|weight|) against the backbone's. encoder << backbone => the pathway is starved.
        if self._grad_count:
            keys = ["encoder", "fusion", "gate", "backbone"]
            means = {k: (v[0] / self._grad_count, v[1] / self._grad_count) for k, v in self._grad_accum.items()}
            LOGGER.info(
                "motion grad (unscaled) | "
                + "  ".join(f"{k}:|g|={means[k][0]:.2e} |g|/|w|={means[k][1]:.2e}" for k in keys if k in means)
            )
            gcsv = self.save_dir / "motion_grad.csv"
            if not gcsv.exists():
                gcsv.write_text("epoch," + ",".join(f"{k}_gnorm,{k}_gwratio" for k in keys) + "\n")
            with open(gcsv, "a") as f:
                row = [str(self.epoch)]
                for k in keys:
                    g, r = means.get(k, (float("nan"), float("nan")))
                    row += [f"{g:.6g}", f"{r:.6g}"]
                f.write(",".join(row) + "\n")
            self._grad_accum = {}
            self._grad_count = 0

    def _motion_warmup_step(self, *_trainer):
        """Freeze everything but the motion pathway for the first `motion_warmup_epochs` epochs.

        Registered as an ``on_train_epoch_start`` callback, which ultralytics invokes as
        ``callback(trainer)``; accepts and ignores that positional arg since this is a bound method
        that already has ``self``.

        Backbone/head parameters live under ``self.model`` (named like ``model.0...``); ``motion_encoder``
        and ``cross_attns`` are separate top-level submodules and never match that prefix, so a single
        ``"model."`` substring cleanly isolates the motion pathway without touching the existing
        `args.freeze` mechanism.
        """
        model = unwrap_model(self.model)
        if self.epoch == 0:
            self.freeze_layer_names = ["model."]
            for k, v in model.named_parameters():
                if v.dtype.is_floating_point:
                    v.requires_grad = not k.startswith("model.")
            LOGGER.info(
                f"Motion warmup: training motion_encoder + cross_attns only for epochs "
                f"0-{self.motion_warmup_epochs - 1}"
            )
        elif self.epoch == self.motion_warmup_epochs:
            freeze_list = (
                self.args.freeze
                if isinstance(self.args.freeze, list)
                else range(self.args.freeze)
                if isinstance(self.args.freeze, int)
                else []
            )
            self.freeze_layer_names = [f"model.{x}." for x in freeze_list] + [".dfl"]
            for k, v in model.named_parameters():
                if v.dtype.is_floating_point:
                    v.requires_grad = not any(x in k for x in self.freeze_layer_names)
            LOGGER.info("Motion warmup complete: unfreezing backbone/head for joint fine-tuning")

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Build the optimizer, then isolate motion-pathway params into their own no-decay, higher-lr group.

        Without this, the attention ``gate`` (and the rest of ``motion_encoder``/``cross_attns``) shares an
        lr and weight-decay schedule tuned for a converged pretrained backbone, which can leave the gate
        stuck near zero once the backbone unfreezes.
        """
        optimizer = super().build_optimizer(
            model, name=name, lr=lr, momentum=momentum, decay=decay, iterations=iterations
        )
        if self.motion_lr_mult == 1.0:
            return optimizer

        motion_param_ids = {
            id(p) for n, p in unwrap_model(model).named_parameters() if not n.startswith("model.")
        }
        if not motion_param_ids:
            return optimizer

        new_groups = []
        for group in optimizer.param_groups:
            keep, moved = [], []
            for p in group["params"]:
                (moved if id(p) in motion_param_ids else keep).append(p)
            group["params"] = keep
            if moved:
                motion_group = {k: v for k, v in group.items() if k != "params"}
                motion_group.update(params=moved, lr=group.get("lr", lr) * self.motion_lr_mult, weight_decay=0.0)
                new_groups.append(motion_group)
        optimizer.param_groups.extend(new_groups)
        LOGGER.info(
            f"optimizer: isolated motion pathway into {len(new_groups)} param group(s) at "
            f"{self.motion_lr_mult}x lr, no weight decay"
        )
        return optimizer

    # ------------------------------------------------------------------
    # Dataset / loader
    # ------------------------------------------------------------------

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build a MotionYOLODataset for the given split.

        Args:
            img_path (str): Path to images directory.
            mode (str): ``'train'`` or ``'val'``.
            batch (int, optional): Batch size for rect-mode padding.

        Returns:
            (MotionYOLODataset): Configured dataset.
        """
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_motion_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct and return a DataLoader backed by MotionYOLODataset.

        Args:
            dataset_path (str): Path to the dataset split.
            batch_size (int): Batch size.
            rank (int): Distributed training rank.
            mode (str): ``'train'`` or ``'val'``.

        Returns:
            (InfiniteDataLoader): DataLoader with motion pairs.
        """
        import numpy as np

        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Instantiate and return a MotionDetectionModel.

        Args:
            cfg (str, optional): Model config YAML path.
            weights (str, optional): Pre-trained weights path.
            verbose (bool): Print model summary.

        Returns:
            (MotionDetectionModel): The motion-aware detection model.
        """
        model = MotionDetectionModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)
        return model

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess_batch(self, batch: dict) -> dict:
        """Preprocess batch: move tensors to device and normalise both img and motion.

        Args:
            batch (dict): Raw batch dict from the DataLoader.

        Returns:
            (dict): Preprocessed batch with float img and motion in [0, 1].
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")

        batch["img"] = batch["img"].float() / 255
        if "motion" in batch:
            batch["motion"] = batch["motion"].float() / 255

        # Optional multi-scale jitter (applied identically to img; motion resized to match)
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = (
                random.randrange(
                    max(self.stride, int(self.args.imgsz * (1.0 - self.args.multi_scale))),
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),
                )
                // self.stride
                * self.stride
            )
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
                if "motion" in batch:
                    batch["motion"] = nn.functional.interpolate(
                        batch["motion"], size=ns, mode="bilinear", align_corners=False
                    )
            batch["img"] = imgs

        # Resolution-degradation augmentation: low-pass the appearance stream on a random subset so
        # few-pixel targets vanish from appearance and only motion can locate them (motion untouched).
        if self.motion_degrade > 0.0:
            img = batch["img"]
            B, _, H, W = img.shape
            mask = torch.rand(B, device=img.device) < self.motion_degrade
            if mask.any():
                f = random.uniform(self.motion_degrade_min, 0.9)  # kept-resolution fraction
                h2, w2 = max(1, round(H * f)), max(1, round(W * f))
                sub = img[mask]
                # area-downsample (anti-aliased) then bilinear upsample back to full size = low-pass
                # that destroys point targets while preserving geometry, so labels need no transform.
                down = nn.functional.interpolate(sub, size=(h2, w2), mode="area")
                img[mask] = nn.functional.interpolate(down, size=(H, W), mode="bilinear", align_corners=False)
                batch["img"] = img

        return batch
