# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Motion-aware YOLO detection task (train / val / predict)."""

from .predict import MotionDetectionPredictor
from .train import MotionDetectionTrainer
from .val import MotionDetectionValidator

__all__ = ("MotionDetectionTrainer", "MotionDetectionValidator", "MotionDetectionPredictor")
