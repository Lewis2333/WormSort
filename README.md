# WormSort

WormSort is a YOLO-compatible multi-object tracking module designed for **C. elegans worm tracking** in microscopic videos.

It extends the standard tracking pipeline by incorporating worm-specific motion and shape cues, including bounding-box IoU, midpoint motion, skeleton curvature, and skeleton length.

The tracker can be directly called through the Ultralytics YOLO `model.track()` interface.

---

## Features

- Compatible with Ultralytics YOLO tracking interface
- Designed for multi-worm tracking in microscopy videos
- Uses both motion and morphology information
- Supports instance mask-based skeleton feature extraction
- Combines:
  - IoU matching
  - midpoint motion prediction
  - worm curvature similarity
  - skeleton length similarity
- Supports low-confidence detection recovery similar to ByteTrack

---

## Installation

Create a Python environment:

```bash
conda create -n wormsort python=3.9
conda activate wormsort
````

Install dependencies:

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install scipy
pip install scikit-image
pip install networkx
```

---

## File Preparation

download WormSort Program and prepare a tracker configuration file, for example:

```text
wormsort.yaml
```

A typical project structure can be:

```text
.
├── weight/
│   └── best.pt
├── video/
│   └── input.mp4
├── wormsort.yaml
└── demo_track.py
```

---

## Basic Usage

You can call WormSort directly through YOLO:

```python
from ultralytics import YOLO

model = YOLO("weight/best.pt")

results = model.track(
    source="video/input.mp4",
    tracker="wormsort.yaml",
    conf=0.1,
    iou=0.3,
    persist=True,
    save=True
)
```

The tracked video will be saved automatically by YOLO.

---

## Frame-by-Frame Usage

For customized processing, you can also read the video frame by frame:

```python
import cv2
from ultralytics import YOLO

model = YOLO("weight/best.pt")

cap = cv2.VideoCapture("video/input.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        tracker="wormsort.yaml",
        conf=0.1,
        iou=0.3,
        persist=True,
        verbose=False
    )

    if len(results) == 0:
        continue

    result = results[0]

    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().numpy()

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            print(f"ID={track_id}, box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

cap.release()
```

---

## Example: Save Tracking Video

```python
from ultralytics import YOLO

model = YOLO("weight/best.pt")

model.track(
    source="video/input.mp4",
    tracker="wormsort.yaml",
    conf=0.1,
    iou=0.3,
    persist=True,
    save=True,
    project="runs/wormsort",
    name="track_result"
)
```

The output will be saved to:

```text
runs/wormsort/track_result/
```

---

## Example Tracker Config

Create a file named `wormsort.yaml`:

```yaml
# Usage: Place this file at ultralytics/cfg/trackers/wormsort.yaml
#        Then call model.track(source, tracker="wormsort.yaml")

tracker_type: wormsort

# ── Basic tracking parameters (consistent with BoT-SORT/ByteTrack) ──
track_high_thresh: 0.4       # High-confidence detection threshold
track_low_thresh: 0.1        # Low-confidence detection threshold (ByteTrack second stage)
new_track_thresh: 0.7        # Minimum score for new track initialization
track_buffer: 30             # Frame retention for lost tracks
match_thresh: 0.7            # Association threshold for the first matching stage

# ── Inherited BoT-SORT parameters ──
fuse_score: True             # Whether to fuse detection confidence scores
gmc_method: sparseOptFlow    # Global motion compensation method
proximity_thresh: 0.5        # ReID proximity threshold
appearance_thresh: 0.25      # ReID appearance similarity threshold
with_reid: False             # ReID disabled for worm scenarios


# ── Four-modal fusion weights (sum must equal 1.0) ──
w_iou: 0.6                   # Weight for IoU spatial overlap
w_mid: 0.1                   # Weight for skeleton midpoint distance (UKF prediction)
w_curv: 0.2                  # Weight for skeleton curvature similarity
w_len: 0.1                   # Weight for skeleton length consistency

# ── EMA smoothing coefficients ──
alpha_update: 0.7            # EMA alpha for normal updates (higher = more trust in new data)
alpha_reactivate: 0.9        # EMA alpha for track reactivation (higher trust in new data after recovery)

# ── UKF (CTRV motion model) parameters ──
ukf_q_pos: 4.0               # Process noise: position (pixel²)
ukf_q_vel: 2.0               # Process noise: velocity
ukf_q_theta: 0.3             # Process noise: heading angle (rad²)
ukf_q_omega: 0.1             # Process noise: angular velocity
ukf_r_pos: 2.0               # Observation noise: position (skeleton midpoint extraction precision)

# ── Skeleton feature configuration ──
curvature_samples: 10        # Sampling points for curvature calculation
min_skeleton_points: 5       # Minimum valid skeleton points (extraction fails below this threshold)
```

---

## Important Notes

1. WormSort is designed for YOLO models with segmentation masks.
2. If masks are available, WormSort extracts worm skeleton features from the instance masks.
3. If masks are not available, the tracker can still use box-level motion cues, but shape-based matching will be weakened.
4. The tracker should be used with `persist=True` when processing videos frame by frame.
5. The `tracker="wormsort.yaml"` argument tells YOLO to use the WormSort tracker configuration.

---
