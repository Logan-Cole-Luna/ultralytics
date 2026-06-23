# Motion Cross-Attention Detection

Extension to YOLOv8 that adds a **motion cross-attention stream** for video object detection.  
The model jointly processes the **current frame** and a **motion-difference image** (absolute pixel-wise diff between consecutive frames), fusing temporal motion cues into the backbone via cross-attention at three feature scales.

---

## Table of Contents

1. [What Was Built](#what-was-built)
2. [Architecture](#architecture)
3. [Parameter Overhead](#parameter-overhead)
4. [Data Format](#data-format)
5. [Preparing Motion Diff Images](#preparing-motion-diff-images)
6. [Training](#training)
7. [Validation](#validation)
8. [Inference](#inference)
9. [File Reference](#file-reference)

---

## What Was Built

| Component | Description |
|---|---|
| `MotionEncoder` | Lightweight 4-stage CNN that extracts multi-scale features from the motion-diff image |
| `MotionCrossAttention` | Residual cross-attention block fusing motion cues into backbone features; gate starts at 0 |
| `MotionDetectionModel` | `DetectionModel` subclass that runs dual-stream forward and injects cross-attention after P3, P4, P5 |
| `MotionYOLODataset` | `YOLODataset` subclass that loads a paired motion-diff image alongside every training frame |
| `MotionDetectionTrainer` | Trainer that builds `MotionYOLODataset`, normalises both `img` and `motion` tensors |
| `MotionDetectionValidator` | Validator with motion-aware preprocessing and model-forward wrapping |
| `MotionDetectionPredictor` | Predictor that auto-computes frame difference from consecutive video frames |
| `yolov8-motion.yaml` | Model config identical to `yolov8.yaml` plus a `motion:` section for the new stream |

---

## Architecture

```
Current frame  ──────────────────────────────────────────────────────────────────►
               Conv Conv C2f Conv C2f  Conv C2f  Conv C2f  SPPF
                              │           │                  │
                              │ P3        │ P4               │ P5
                              ▼           ▼                  ▼
                          CrossAttn   CrossAttn          CrossAttn
                              ▲           ▲                  ▲
Motion diff    ──► MotionEncoder ─────────────────────────────
               stage0 stage1 stage2 stage3
               /2    /4     /8     /16
```

**Cross-attention design (per scale):**

```
Q ← current-frame features  (C × H × W)
K,V ← motion features        (Cm × H × W, spatially interpolated to match)

out = x_current + tanh(gate) · CrossAttn(Q, K, V)
```

- `gate` is a scalar `nn.Parameter` initialised to `0` → `tanh(0) = 0` → zero contribution at initialisation.  
  The gate grows naturally during training, so the model can start from any pretrained YOLOv8 checkpoint without degradation.
- Motion features are spatially resized to match the current-frame feature map before attention, so mismatched strides are handled automatically.

---

## Parameter Overhead

For `yolov8n-motion` (nano scale):

| Component | Parameters |
|---|---|
| YOLOv8n backbone + head | 3,157,200 |
| MotionEncoder (4 stages) | 1,173,216 |
| MotionCrossAttention × 3 | 388,611 |
| **Total** | **4,719,027** |

The motion stream adds ~49% more parameters over the nano baseline, mostly in the encoder.  
For larger scales (`s/m/l/x`) the base model dominates and the relative overhead is smaller.

---

## Data Format

Standard YOLO detection layout plus a `motion/` directory sibling to `images/`:

```
dataset/
├── images/
│   ├── train/
│   │   ├── frame0001.jpg
│   │   ├── frame0002.jpg
│   │   └── ...
│   └── val/
│       └── ...
├── motion/                    ← NEW: motion-difference images
│   ├── train/
│   │   ├── frame0001.jpg      ← abs(frame0001 - frame0000)
│   │   ├── frame0002.jpg      ← abs(frame0002 - frame0001)
│   │   └── ...
│   └── val/
│       └── ...
└── labels/
    ├── train/
    │   ├── frame0001.txt
    │   └── ...
    └── val/
        └── ...
```

**Rules:**
- Motion images must have **identical filenames** to their corresponding current frames.
- Any standard image format is accepted (`.jpg`, `.png`, `.bmp`, …).
- If a motion image is missing for a sample, a **zero tensor** is used automatically — training will still work, the model just won't receive motion context for that sample.
- Grayscale motion images are converted to 3-channel (BGR) automatically.

---

## Preparing Motion Diff Images

### From a video (OpenCV)

```python
import cv2, os

video = cv2.VideoCapture("video.mp4")
out_dir = "dataset/motion/train"
os.makedirs(out_dir, exist_ok=True)

ret, prev = video.read()
idx = 0
while True:
    ret, curr = video.read()
    if not ret:
        break
    diff = cv2.absdiff(curr, prev)
    cv2.imwrite(f"{out_dir}/frame{idx:04d}.jpg", diff)
    prev = curr
    idx += 1
    
# Frame 0 has no previous → write zeros or copy first diff
cv2.imwrite(f"{out_dir}/frame0000.jpg", diff * 0)
```

### From an image directory (sorted frames)

```python
import cv2, os, glob

frames = sorted(glob.glob("dataset/images/train/*.jpg"))
out_dir = "dataset/motion/train"
os.makedirs(out_dir, exist_ok=True)

prev = cv2.imread(frames[0])
# first frame: zero motion
cv2.imwrite(f"{out_dir}/{os.path.basename(frames[0])}", prev * 0)

for i in range(1, len(frames)):
    curr = cv2.imread(frames[i])
    diff = cv2.absdiff(curr, prev)
    cv2.imwrite(f"{out_dir}/{os.path.basename(frames[i])}", diff)
    prev = curr
```

### Optical flow (richer motion signal)

```python
import cv2, numpy as np

def flow_to_bgr(flow):
    """Convert optical flow to a 3-channel BGR visualisation."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2     # hue = direction
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                     0.5, 3, 15, 3, 5, 1.2, 0)
motion_img = flow_to_bgr(flow)
cv2.imwrite("motion/train/frame0002.jpg", motion_img)
```

---

## Training

```python
from ultralytics.models.yolo.motion.train import MotionDetectionTrainer

trainer = MotionDetectionTrainer(overrides=dict(
    model="yolov8-motion.yaml",   # fresh model
    # model="yolov8n.pt",         # pretrained checkpoint (gate=0 ensures no degradation)
    data="path/to/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
))
trainer.train()
```

**Fine-tuning from a standard YOLOv8 checkpoint:**  
Because `MotionDetectionModel.load()` uses `intersect_dicts`, all backbone/head weights  
transfer cleanly. The new `motion_encoder` and `cross_attns` parameters start from random  
init, but the `gate=0` ensures the cross-attention adds nothing until the new parameters  
have had time to warm up.

```python
trainer = MotionDetectionTrainer(overrides=dict(
    model="yolov8-motion.yaml",
    # pretrain base weights then the motion modules train from scratch
    data="data.yaml",
    epochs=50,
))
```

**data.yaml** is a standard YOLO dataset config — no changes needed:

```yaml
path: /path/to/dataset
train: images/train
val:   images/val
nc: 80
names: {0: person, 1: car, ...}
```

---

## Validation

```python
from ultralytics.models.yolo.motion.val import MotionDetectionValidator

validator = MotionDetectionValidator(args=dict(
    model="runs/detect/train/weights/best.pt",
    data="data.yaml",
    imgsz=640,
    batch=8,
))
stats = validator()
print(stats)   # mAP50, mAP50-95, etc.
```

The validator automatically:
1. Builds a `MotionYOLODataset` so each batch has a `motion` key.
2. Normalises `motion` alongside `img` in `preprocess()`.
3. Wraps `model.forward` to inject the batch motion into the standard inference call.

---

## Inference

### Video (automatic frame-differencing)

```python
from ultralytics.models.yolo.motion.predict import MotionDetectionPredictor

predictor = MotionDetectionPredictor(overrides=dict(
    model="runs/detect/train/weights/best.pt",
    conf=0.25,
))

for result in predictor.stream_inference("video.mp4"):
    result.show()                         # display
    # result.save("output/")             # save to disk
```

The predictor maintains `_prev_frame` across frames.  
- **Frame 0**: motion = zeros (no previous frame available).  
- **Frame N > 0**: motion = `abs(frame_N − frame_{N-1})`.  
- Each new `stream_inference()` call resets the temporal state automatically.

### Single image (zero motion fallback)

```python
results = predictor("image.jpg")   # motion = zeros; standard detection
```

### Explicit pre-computed motion

```python
import torch, cv2
curr   = torch.from_numpy(cv2.imread("curr.jpg")).permute(2,0,1).float().unsqueeze(0) / 255
motion = torch.from_numpy(cv2.imread("motion.jpg")).permute(2,0,1).float().unsqueeze(0) / 255

results = predictor("curr.jpg", motion=motion)
```

---

## File Reference

### New files

| File | Description |
|---|---|
| [`ultralytics/nn/modules/motion.py`](ultralytics/nn/modules/motion.py) | `MotionEncoder`, `MotionCrossAttention` |
| [`ultralytics/cfg/models/v8/yolov8-motion.yaml`](ultralytics/cfg/models/v8/yolov8-motion.yaml) | Model config (`motion:` section + standard YOLOv8 backbone/head) |
| [`ultralytics/models/yolo/motion/__init__.py`](ultralytics/models/yolo/motion/__init__.py) | Package exports |
| [`ultralytics/models/yolo/motion/train.py`](ultralytics/models/yolo/motion/train.py) | `MotionDetectionTrainer`, `build_motion_dataset` |
| [`ultralytics/models/yolo/motion/val.py`](ultralytics/models/yolo/motion/val.py) | `MotionDetectionValidator` |
| [`ultralytics/models/yolo/motion/predict.py`](ultralytics/models/yolo/motion/predict.py) | `MotionDetectionPredictor` |

### Modified files

| File | Change |
|---|---|
| [`ultralytics/nn/tasks.py`](ultralytics/nn/tasks.py) | Added `MotionDetectionModel` after `DetectionModel` |
| [`ultralytics/data/dataset.py`](ultralytics/data/dataset.py) | Added `MotionYOLODataset` after `YOLOMultiModalDataset` |
| [`ultralytics/nn/modules/__init__.py`](ultralytics/nn/modules/__init__.py) | Exports `MotionEncoder`, `MotionCrossAttention` |
| [`ultralytics/data/__init__.py`](ultralytics/data/__init__.py) | Exports `MotionYOLODataset` |

### YAML motion section

```yaml
motion:
  channels: 3               # motion diff image channels (3 = RGB absolute diff)
  dims: [32, 64, 128, 256]  # MotionEncoder output channels per downsampling stage
  inject_layers: [4, 6, 9]  # backbone layer indices to inject cross-attention after
  motion_feat_scales: [2, 3, 3]
  # Mapping: inject_layers[i] uses MotionEncoder output dims[motion_feat_scales[i]]
  # inject_layers[0]=4 (P3/8)   → dims[2]=128ch at stride-8
  # inject_layers[1]=6 (P4/16)  → dims[3]=256ch at stride-16
  # inject_layers[2]=9 (P5/32)  → dims[3]=256ch, spatially interpolated to stride-32
```

To customise the injection points (e.g. only P4 and P5, or a different backbone):

```yaml
motion:
  inject_layers: [6, 9]
  motion_feat_scales: [3, 3]
```

---

## Design Notes

**Why cross-attention and not concatenation?**  
Concatenation would require modifying the first conv and break all pretrained weight compatibility. Cross-attention lets the current-frame stream remain identical to standard YOLOv8, so any pretrained `.pt` can be used as a starting point.

**Why gate = 0 at init?**  
`tanh(0) = 0` means the motion stream contributes nothing at initialisation, making the model equivalent to standard YOLOv8 at step 0. The gate grows during training as gradient updates push it away from zero. This avoids training instability when starting from a pretrained checkpoint.

**Why is mosaic disabled in `MotionYOLODataset`?**  
Mosaic combines four independent clips into one image. The motion-diff for the composite would be semantically incoherent (motion from four different scenes). Disabling mosaic preserves temporal consistency between `img` and `motion`.
