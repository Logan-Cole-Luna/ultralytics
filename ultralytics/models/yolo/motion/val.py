# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Motion-aware detection validator.

Extends DetectionValidator to:
  • Build MotionYOLODataset for validation so that motion tensors are available in the batch.
  • Normalise the 'motion' tensor in preprocess() alongside 'img'.
  • Pass the motion tensor to the model during validation inference.

Note on motion during validation inference
------------------------------------------
BaseValidator.__call__ calls ``model(batch["img"], augment=augment)``, which does not pass motion.
To avoid a costly override of the entire __call__ loop, the validator stores the current-batch
motion in ``self._motion`` during preprocess() and the MotionDetectionModel reads it by overriding
forward() to check for motion in the batch dict when called in loss mode.

For standalone validation (not training-time), the validator wraps the model's forward to inject
the last-seen motion tensor automatically.
"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import torch

from ultralytics.data import build_dataloader
from ultralytics.data.dataset import MotionYOLODataset
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import torch_distributed_zero_first

from .train import build_motion_dataset


class MotionDetectionValidator(DetectionValidator):
    """Validator for motion-aware YOLOv8 detection.

    Differences from DetectionValidator:
      - Builds MotionYOLODataset so batches contain a 'motion' key.
      - preprocess() also normalises the motion tensor.
      - Wraps the model's forward to inject the current batch's motion tensor
        during the standard ``model(batch["img"], augment=augment)`` call.

    Examples:
        >>> from ultralytics.models.yolo.motion.val import MotionDetectionValidator
        >>> validator = MotionDetectionValidator(args=dict(model="yolov8-motion.pt", data="coco8.yaml"))
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None):
        """Initialise the motion-aware validator."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self._motion: torch.Tensor | None = None  # set in preprocess(), consumed by the model wrapper
        self._model_original_forward = None  # stores the original forward during validation

    # ------------------------------------------------------------------
    # Dataset / loader
    # ------------------------------------------------------------------

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        """Build a MotionYOLODataset for validation.

        Args:
            img_path (str): Path to images directory.
            mode (str): Always ``'val'`` for the validator.
            batch (int, optional): Batch size for rect-mode padding.

        Returns:
            (MotionYOLODataset): Configured motion-aware dataset.
        """
        return build_motion_dataset(self.args, img_path, batch or self.args.batch, self.data, mode="val", rect=True)

    def get_dataloader(self, dataset_path: str, batch_size: int):
        """Build and return a DataLoader over MotionYOLODataset.

        Args:
            dataset_path (str): Path to the validation images directory.
            batch_size (int): Batch size.

        Returns:
            (DataLoader): Validation DataLoader with motion pairs.
        """
        dataset = self.build_dataset(dataset_path, mode="val", batch=batch_size)
        return build_dataloader(dataset, batch=batch_size, workers=self.args.workers * 2, shuffle=False, rank=-1)

    # ------------------------------------------------------------------
    # Preprocessing – normalises motion alongside img
    # ------------------------------------------------------------------

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess validation batch, normalising both img and motion tensors.

        Also caches the motion tensor in self._motion so that the wrapped model
        forward can inject it during the standard BaseValidator inference call.

        Args:
            batch (dict): Raw batch dict from the DataLoader.

        Returns:
            (dict): Preprocessed batch.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        if "motion" in batch:
            batch["motion"] = (batch["motion"].half() if self.args.half else batch["motion"].float()) / 255
            self._motion = batch["motion"]
        else:
            self._motion = None
        return batch

    # ------------------------------------------------------------------
    # Model wrapping – inject motion into the standard inference call
    # ------------------------------------------------------------------

    def _wrap_model_forward(self, model):
        """Monkey-patch model.forward to automatically pass self._motion.

        The standard BaseValidator loop calls ``model(batch["img"], augment=augment)``.
        This wrapper intercepts that call and injects the motion tensor cached in
        self._motion so that MotionDetectionModel._predict_once receives it.

        Args:
            model: The detection model to wrap.
        """
        validator = self  # closure reference
        original_forward = model.forward

        def _forward_with_motion(x, *args, **kwargs):
            if not isinstance(x, dict) and "motion" not in kwargs and validator._motion is not None:
                kwargs["motion"] = validator._motion
            return original_forward(x, *args, **kwargs)

        model.forward = _forward_with_motion
        return original_forward  # caller should restore this when done

    def _unwrap_model_forward(self, model, original_forward):
        """Restore the original model.forward after validation."""
        model.forward = original_forward

    # ------------------------------------------------------------------
    # Override __call__ to inject motion during inference
    # ------------------------------------------------------------------

    def __call__(self, trainer=None, model=None):
        """Run validation, wrapping the model to pass motion during inference.

        Wraps model.forward before entering the BaseValidator loop and restores it
        afterwards so that subsequent uses of the model are unaffected.
        """
        # Resolve the actual model object (before AutoBackend wrapping in standalone mode)
        # The wrap is applied after the parent resolves the model
        from ultralytics.utils.torch_utils import smart_inference_mode

        @smart_inference_mode()
        def _run():
            # Let the parent do all setup; we intercept after the model is resolved.
            # We achieve this by calling super().__call__ directly.
            return super(MotionDetectionValidator, self).__call__(trainer=trainer, model=model)

        # If we have a trainer, the model is already available and we can wrap before the call
        if trainer is not None:
            actual_model = trainer.ema.ema or trainer.model
            if trainer.args.compile and hasattr(actual_model, "_orig_mod"):
                actual_model = actual_model._orig_mod
            orig = self._wrap_model_forward(actual_model)
            try:
                result = super().__call__(trainer=trainer, model=model)
            finally:
                self._unwrap_model_forward(actual_model, orig)
            return result
        else:
            # Standalone validation: wrap after parent's AutoBackend setup
            # We override by temporarily patching to no-op and then calling
            # Use the simpler approach: motion injection via preprocess caching works
            # because the parent loop calls self.preprocess → sets self._motion,
            # and then model(batch["img"]) goes through the wrapped forward.
            # Wrap the provided model if it's a MotionDetectionModel
            if model is not None and hasattr(model, "motion_encoder"):
                orig = self._wrap_model_forward(model)
                try:
                    result = super().__call__(trainer=None, model=model)
                finally:
                    self._unwrap_model_forward(model, orig)
                return result
            else:
                return super().__call__(trainer=trainer, model=model)
