# ByteTrack-CPP

High-performance C++ implementation of [ByteTrack](https://arxiv.org/abs/2110.06864) multi-object tracker with Python bindings via pybind11.

Built for production pipelines where Python-based trackers become bottlenecks. Core logic runs in C++ while exposing a Pythonic API.

## Features

- **C++ core** with Eigen-based Kalman Filter and LAPJV linear assignment
- **pybind11 bindings** — use from Python with native syntax
- **Class-aware matching** — only matches detections with tracks of the same class (like Ultralytics)
- **Multiple input formats** — supports `xywh` and `xyxy` numpy arrays
- **Numpy output** — optional direct numpy array output for zero-overhead pipelines
- **Python properties** — get/set tracker parameters with `tracker.match_thresh = 0.7`
- **Thread-safe ID counter** — `std::atomic` track ID, safe for multi-stream
- **Min box area filter** — filter out small detections before tracking

## Installation

### Requirements

- CMake >= 3.21
- C++17 compiler
- OpenCV (core module)
- Eigen3
- pybind11
- Python 3.8+

### Build & Install

```bash
# Install into your conda/virtualenv
pip install .

# Or editable (note: C++ won't auto-rebuild on changes)
pip install -e .

# To rebuild after C++ changes
rm -rf _skbuild && pip install .
```

### Build C++ only (without Python)

```bash
mkdir build && cd build
cmake ..
make
```

## Quick Start

### Basic usage with detection objects

```python
from bytetrack import BYTETracker, make_object

tracker = BYTETracker(
    max_time_lost=30,
    track_high_thresh=0.5,
    track_low_thresh=0.1,
    new_track_thresh=0.6,
    match_thresh=0.8,
    fuse_score=True,
    class_aware=True,     # only match same-class detections
    min_box_area=100.0,   # ignore tiny boxes
)

# Create detections (x, y, w, h, confidence, class_id)
objects = [
    make_object(100, 200, 50, 120, 0.9, 0),
    make_object(300, 150, 60, 140, 0.85, 0),
]

tracked, lost = tracker.update(objects)

for t in tracked:
    print(f"ID:{t.track_id} bbox:{t.tlbr} score:{t.score} cls:{t.cls}")
```

### With numpy arrays (recommended for YOLO pipelines)

```python
import numpy as np
from bytetrack import BYTETracker

tracker = BYTETracker()

# Detections as numpy array: [x1, y1, x2, y2, score, class_id]
dets = np.array([
    [100, 200, 150, 320, 0.9, 0],
    [300, 150, 360, 290, 0.85, 0],
], dtype=np.float32)

# format="xyxy" for x1,y1,x2,y2 | format="xywh" for x,y,w,h
tracked, lost = tracker.update_from_numpy(dets, format="xyxy")
```

### Pure numpy pipeline (numpy in, numpy out)

```python
# Returns numpy arrays instead of STrack objects
# Output shape: (N, 8) = [x1, y1, x2, y2, track_id, score, cls, state]
tracked_np, lost_np = tracker.update_numpy(dets, format="xyxy")

if len(tracked_np) > 0:
    track_ids = tracked_np[:, 4].astype(int)
    bboxes = tracked_np[:, :4]
```

### Integration with Ultralytics YOLO

```python
import cv2
import numpy as np
from ultralytics import YOLO
from bytetrack import BYTETracker

model = YOLO("yolo11s.pt")
tracker = BYTETracker(max_time_lost=30, track_high_thresh=0.4)

cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()[:, None]
        cls = boxes.cls.cpu().numpy()[:, None]
        dets = np.hstack([xyxy, conf, cls]).astype(np.float32)
        tracked, lost = tracker.update_from_numpy(dets, format="xyxy")
    else:
        tracked, lost = tracker.update([])

    for t in tracked:
        x1, y1, x2, y2 = [int(v) for v in t.tlbr]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{t.track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
```

## API Reference

### `BYTETracker`

#### Constructor

```python
BYTETracker(
    max_time_lost: int = 15,       # frames before a lost track is removed
    track_high_thresh: float = 0.5, # detection confidence for first association
    track_low_thresh: float = 0.1,  # minimum confidence to consider a detection
    new_track_thresh: float = 0.6,  # confidence to initialize a new track
    match_thresh: float = 0.8,      # IoU threshold for matching
    fuse_score: bool = True,        # fuse detection score into IoU cost
    class_aware: bool = True,       # only match detections with same-class tracks
    min_box_area: float = 0.0,      # minimum box area (w*h) to accept
)
```

#### Properties (read/write)

All constructor parameters are also available as properties:

```python
tracker.max_time_lost = 60
tracker.match_thresh = 0.7
tracker.class_aware = False
print(tracker.frame_id)  # read-only
```

#### Methods

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `update(objects)` | `List[Object]` | `(List[STrack], List[STrack])` | Update with Object list |
| `update_from_numpy(dets, format)` | `ndarray (N,5+)` | `(List[STrack], List[STrack])` | Update with numpy, returns STrack |
| `update_numpy(dets, format)` | `ndarray (N,5+)` | `(ndarray, ndarray)` | Update with numpy, returns numpy |
| `reset()` | — | — | Reset tracker state and IDs |
| `reset_id()` | — | — | Static: reset global ID counter |

- `format`: `"xywh"` (default) or `"xyxy"`
- Numpy input columns: `[x, y, w/x2, h/y2, score, class_id(optional)]`
- Numpy output columns: `[x1, y1, x2, y2, track_id, score, cls, state]`

### `STrack`

| Attribute | Type | Description |
|-----------|------|-------------|
| `track_id` | `int` | Unique track ID |
| `tlwh` | `List[float]` | Bounding box `[top, left, width, height]` |
| `tlbr` | `List[float]` | Bounding box `[x1, y1, x2, y2]` |
| `score` | `float` | Detection confidence |
| `cls` | `int` | Class ID |
| `state` | `int` | `TrackState` (0=New, 1=Tracked, 2=Lost, 3=Removed) |
| `is_activated` | `bool` | Whether track has been confirmed |
| `frame_id` | `int` | Last frame this track was updated |
| `start_frame` | `int` | Frame when this track was first seen |
| `tracklet_len` | `int` | Number of consecutive frames tracked |

### `TrackState`

```python
from bytetrack import TrackState

TrackState.New       # 0
TrackState.Tracked   # 1
TrackState.Lost      # 2
TrackState.Removed   # 3
```

### Helper

```python
from bytetrack import make_object

obj = make_object(x, y, w, h, prob, label=0)
```

## How ByteTrack Works

```
Detections (frame N)
    |
    v
[Split by confidence]
    |               |
    v               v
  High-conf       Low-conf
  detections      detections
    |
    v
[1st Association] --- IoU matching with predicted tracks (Kalman Filter)
    |
    |-- Matched: update track
    |-- Unmatched tracks ──> [2nd Association] with low-conf detections
    |                             |-- Matched: update track
    |                             |-- Unmatched: mark as Lost
    |-- Unmatched high-conf dets ──> [3rd Association] with unconfirmed tracks
    |                                    |-- Matched: confirm track
    |                                    |-- Unmatched dets: init new track
    v
[Output] active tracked objects
```

## Project Structure

```
ByteTrack-CPP/
├── include/
│   ├── BYTETracker.h      # Main tracker class
│   ├── STrack.h            # Single track (state + Kalman)
│   ├── kalmanFilter.h      # Kalman Filter
│   ├── dataType.h          # Eigen type aliases
│   └── lapjv.h             # Linear assignment (Jonker-Volgenant)
├── src/
│   ├── BYTETracker.cpp     # Tracker update logic
│   ├── STrack.cpp          # Track lifecycle
│   ├── kalmanFilter.cpp    # Kalman predict/update
│   ├── lapjv.cpp           # LAPJV solver
│   ├── utils.cpp           # IoU, matching, assignment
│   ├── PyBYTETrack.cpp     # pybind11 bindings
│   └── bytetrack/
│       ├── __init__.py     # Python package
│       └── __init__.pyi    # Type stubs for IDE support
├── CMakeLists.txt
├── setup.py
└── pyproject.toml
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## References

- [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)
- [Ultralytics ByteTrack implementation](https://github.com/ultralytics/ultralytics)
