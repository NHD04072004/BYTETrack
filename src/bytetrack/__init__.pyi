from enum import IntEnum
from typing import List, Literal, Tuple

import numpy as np
import numpy.typing as npt

class TrackState(IntEnum):
    New: int
    Tracked: int
    Lost: int
    Removed: int

class RectF:
    x: float
    y: float
    width: float
    height: float
    def __init__(self, x: float = ..., y: float = ..., width: float = ..., height: float = ...) -> None: ...

class Object:
    rect: RectF
    label: int
    prob: float
    def __init__(self) -> None: ...

class STrack:
    is_activated: bool
    track_id: int
    state: int
    score: float
    cls: int
    tlwh: List[float]
    tlbr: List[float]
    frame_id: int
    start_frame: int
    tracklet_len: int
    def __init__(self, tlwh: List[float], score: float, cls: int = 0) -> None: ...
    def get_color(self) -> Tuple[int, int, int]: ...
    def __repr__(self) -> str: ...

class BYTETracker:
    # Properties (read/write)
    max_time_lost: int
    track_high_thresh: float
    track_low_thresh: float
    new_track_thresh: float
    match_thresh: float
    fuse_score: bool
    class_aware: bool
    min_box_area: float
    # Read-only property
    @property
    def frame_id(self) -> int: ...

    def __init__(
        self,
        max_time_lost: int = 15,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        fuse_score: bool = True,
        class_aware: bool = True,
        min_box_area: float = 0.0,
    ) -> None: ...

    def update(self, objects: List[Object]) -> Tuple[List[STrack], List[STrack]]: ...

    def update_from_numpy(
        self,
        detections: npt.NDArray[np.float32],
        format: Literal["xywh", "xyxy"] = "xywh",
    ) -> Tuple[List[STrack], List[STrack]]: ...

    def update_numpy(
        self,
        detections: npt.NDArray[np.float32],
        format: Literal["xywh", "xyxy"] = "xywh",
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Update tracker, returns numpy arrays.
        Input: (N, 5+) with format 'xywh' or 'xyxy'.
        Output: tuple of (tracked, lost) arrays of shape (M, 8)
          columns: [x1, y1, x2, y2, track_id, score, cls, state]
        """
        ...

    def reset(self) -> None: ...

    @staticmethod
    def reset_id() -> None: ...

def make_object(
    x: float,
    y: float,
    w: float,
    h: float,
    prob: float,
    label: int = 0,
) -> Object: ...
