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

Place the tracker file into the Ultralytics tracker directory.

For example:

```text
ultralytics/
└── trackers/
    ├── worm_sort.py
    ├── bot_sort.py
    ├── byte_tracker.py
    ├── basetrack.py
    └── utils/
```

Then prepare a tracker configuration file, for example:

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
tracker_type: wormsort

track_high_thresh: 0.25
track_low_thresh: 0.05
new_track_thresh: 0.25
track_buffer: 30
match_thresh: 0.8

w_iou: 0.6
w_mid: 0.1
w_curv: 0.2
w_len: 0.1

alpha_update: 0.7
alpha_reactivate: 0.9

ukf_q_pos: 4.0
ukf_q_vel: 2.0
ukf_q_theta: 0.3
ukf_q_omega: 0.1
ukf_r_pos: 2.0

curvature_samples: 10
min_skeleton_points: 5
```

---

## Important Notes

1. WormSort is designed for YOLO models with segmentation masks.
2. If masks are available, WormSort extracts worm skeleton features from the instance masks.
3. If masks are not available, the tracker can still use box-level motion cues, but shape-based matching will be weakened.
4. The tracker should be used with `persist=True` when processing videos frame by frame.
5. The `tracker="wormsort.yaml"` argument tells YOLO to use the WormSort tracker configuration.


## Minimal Demo

```python
from ultralytics import YOLO

model = YOLO("weight/best.pt")

model.track(
    source="video/input.mp4",
    tracker="wormsort.yaml",
    conf=0.1,
    iou=0.3,
    save=True
)
```

---
