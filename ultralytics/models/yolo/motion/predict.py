# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Motion-aware detection predictor.

Extends DetectionPredictor to compute an absolute frame-difference tensor (motion)
on-the-fly from consecutive frames and injects it into the model forward call.

Behaviour per source type:
  - Video / webcam stream: motion = abs(current_frame − previous_frame).
                           First frame uses zero motion.
  - Single image / list:   zero motion (no temporal context available).
  - Explicit motion kwarg: the caller may pass a pre-computed motion tensor
                           via ``model.predict(source, motion=tensor)``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops
from ultralytics.utils.files import increment_path
from pathlib import Path


class MotionDetectionPredictor(DetectionPredictor):
    """Predictor for motion-aware YOLOv8 detection.

    Computes absolute frame-difference images between consecutive frames during
    video inference and passes them as the ``motion`` tensor to the model.

    For image inputs (single frames with no temporal context) the motion tensor
    is all zeros, so the cross-attention gate ensures no degradation relative to
    a baseline model.

    Attributes:
        _prev_frame (torch.Tensor | None): Previous preprocessed frame (BCHW float [0,1]).
        _explicit_motion (torch.Tensor | None): Caller-supplied motion override.

    Examples:
        >>> from ultralytics.models.yolo.motion.predict import MotionDetectionPredictor
        >>> predictor = MotionDetectionPredictor(overrides=dict(model="yolov8-motion.pt", source="video.mp4"))
        >>> for result in predictor.stream_inference("video.mp4"):
        ...     result.show()

        Passing an explicit motion frame::

        >>> results = predictor(source="frame.jpg", motion=motion_tensor)
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        """Initialise the motion-aware predictor."""
        from ultralytics.utils import DEFAULT_CFG

        super().__init__(cfg or DEFAULT_CFG, overrides, _callbacks)
        self._prev_frame: torch.Tensor | None = None
        self._explicit_motion: torch.Tensor | None = None
        self._motion: torch.Tensor | None = None  # motion for the current step

    # ------------------------------------------------------------------
    # Public API – allow callers to pass explicit motion
    # ------------------------------------------------------------------

    def __call__(self, source=None, model=None, stream: bool = False, motion=None, *args, **kwargs):
        """Run inference, optionally accepting a pre-computed motion tensor.

        Args:
            source: Input source (image path, video, webcam, etc.).
            model: Model override.
            stream (bool): Stream results as a generator.
            motion (torch.Tensor | None): Pre-computed motion tensor (BCHW float [0,1]).
                If provided, overrides the automatic frame-difference computation.

        Returns:
            Results list or generator.
        """
        self._explicit_motion = motion
        return super().__call__(source=source, model=model, stream=stream, *args, **kwargs)

    # ------------------------------------------------------------------
    # Preprocessing – compute motion from consecutive frames
    # ------------------------------------------------------------------

    def preprocess(self, im):
        """Preprocess the current frame and compute the motion-difference tensor.

        For the first frame (or when ``_explicit_motion`` is set) the motion tensor
        is set appropriately.  For subsequent frames the motion is computed as the
        absolute pixel-wise difference from the previous frame.

        Args:
            im (torch.Tensor | list[np.ndarray]): Current frame(s).

        Returns:
            (torch.Tensor): Preprocessed current-frame tensor (BCHW float [0,1]).
        """
        # Standard preprocessing → img_tensor is (B, C, H, W) float32 [0, 1]
        img_tensor = super().preprocess(im)

        if self._explicit_motion is not None:
            # Caller supplied explicit motion; resize to match if needed
            mot = self._explicit_motion.to(img_tensor.device)
            if mot.shape[2:] != img_tensor.shape[2:]:
                mot = F.interpolate(mot, size=img_tensor.shape[2:], mode="bilinear", align_corners=False)
            self._motion = mot
        elif self._prev_frame is None:
            # First frame – no previous frame available
            self._motion = torch.zeros_like(img_tensor)
        else:
            prev = self._prev_frame
            if prev.shape != img_tensor.shape:
                prev = F.interpolate(prev, size=img_tensor.shape[2:], mode="bilinear", align_corners=False)
            self._motion = (img_tensor - prev).abs()

        # Store current frame as previous for the next step
        self._prev_frame = img_tensor.detach()

        return img_tensor

    # ------------------------------------------------------------------
    # Inference – inject motion into the model call
    # ------------------------------------------------------------------

    def inference(self, im: torch.Tensor, *args, **kwargs):
        """Run model inference, passing the current motion tensor as a keyword argument.

        Args:
            im (torch.Tensor): Preprocessed current-frame tensor.

        Returns:
            Model output (raw predictions before NMS).
        """
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and not self.source_type.tensor
            else False
        )
        return self.model(
            im,
            augment=self.args.augment,
            visualize=visualize,
            embed=self.args.embed,
            motion=self._motion,
            *args,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Reset motion state between separate predict() calls
    # ------------------------------------------------------------------

    def setup_source(self, source, stride=None):
        """Set up the inference source and reset motion state for a fresh sequence.

        Args:
            source: Input source descriptor.
            stride (int, optional): Model stride.
        """
        super().setup_source(source, stride)
        self._prev_frame = None  # reset temporal state for each new source
        self._explicit_motion = None
