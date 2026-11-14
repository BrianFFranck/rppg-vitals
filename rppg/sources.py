"""Frame sources. Each yields (t_seconds, BGR frame)."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Iterator

import cv2


def open_camera(index: int = 0, width: int = 640, height: int = 480, fps: int = 30,
                fourcc: str = "YUYV", lock_exposure: bool = False) -> cv2.VideoCapture:
    """Open a webcam with settings that suit rPPG.

    YUYV (uncompressed) keeps the tiny colour changes that MJPEG compression can
    smear. If your camera can't do 30 fps in YUYV at this size, use fourcc="MJPG".
    """
    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if lock_exposure:
        # V4L2 convention: 1 = manual, 3 = auto. Auto-exposure / white balance
        # steps show up as large jumps in the colour signal.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    return cap


def camera_frames(cap: cv2.VideoCapture) -> Iterator[tuple[float, object]]:
    t0 = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield time.monotonic() - t0, frame


def video_frames(path: str | Path, realtime: bool = False) -> Iterator[tuple[float, object]]:
    """Frames from a video file, timestamped by frame index / container fps."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    start = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            if realtime:
                lag = t - (time.monotonic() - start)
                if lag > 0:
                    time.sleep(lag)
            yield t, frame
            i += 1
    finally:
        cap.release()


def pure_frames(folder: str | Path) -> Iterator[tuple[float, object]]:
    """PURE dataset: PNG frames named Image<timestamp_ns>.png."""
    files = sorted(Path(folder).rglob("Image*.png"), key=lambda p: int(re.sub(r"\D", "", p.stem)))
    if not files:
        raise FileNotFoundError(f"No Image*.png frames under {folder}")
    t0 = int(re.sub(r"\D", "", files[0].stem))
    for f in files:
        yield (int(re.sub(r"\D", "", f.stem)) - t0) / 1e9, cv2.imread(str(f))
