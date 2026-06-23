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

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialise the motion-aware trainer."""
        super().__init__(cfg, overrides, _callbacks)

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

        return batch
