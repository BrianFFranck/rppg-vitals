"""Face tracking with MediaPipe Face Landmarker and skin-ROI colour averaging."""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

# MediaPipe 478-point face-mesh indices. Each ROI is the convex hull of its points.
# "right"/"left" are from the subject's point of view.
ROI_LANDMARKS: dict[str, list[int]] = {
    "forehead": [104, 69, 108, 151, 337, 299, 333, 334, 296, 336, 9, 107, 66, 105],
    "right_cheek": [50, 101, 118, 117, 123, 187, 205, 36, 142],
    "left_cheek": [280, 330, 347, 346, 352, 411, 425, 266, 371],
}
_EYE_R, _EYE_L = 33, 263  # outer eye corners, for inter-ocular distance (IOD)


def ensure_model(path: str | Path = DEFAULT_MODEL) -> Path:
    """Download the Face Landmarker model on first use."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe face model to {path} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.rename(path)
    return path


@dataclass
class FaceObservation:
    rgb: np.ndarray  # (3,) mean skin colour, R G B
    landmarks: np.ndarray  # (478, 2) pixel coordinates
    rois: dict[str, np.ndarray]  # ROI name -> (k, 2) int32 hull polygon
    motion: float  # mean landmark speed, in inter-ocular distances per second
    center_y: float  # landmark centroid y, in IODs (used for breathing)
    iod: float  # inter-ocular distance in pixels
    n_pixels: int  # skin pixels averaged


class FaceROITracker:
    """Per-frame face landmarks -> mean RGB over forehead + cheeks."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        rois: list[str] | None = None,
        min_confidence: float = 0.5,
    ):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path or DEFAULT_MODEL))),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.roi_names = list(rois or ROI_LANDMARKS)
        self._last_ts_ms = -1
        self._prev: tuple[float, np.ndarray, float] | None = None  # (t, landmarks, iod)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def process(self, frame_bgr: np.ndarray, t: float) -> FaceObservation | None:
        """Track the face in one BGR frame captured at time `t` (seconds)."""
        h, w = frame_bgr.shape[:2]
        # MediaPipe VIDEO mode only needs strictly increasing timestamps.
        ts = max(int(time.monotonic() * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)),
        )
        result = self._landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            self._prev = None
            return None

        lm = np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]], dtype=np.float32)
        iod = float(np.linalg.norm(lm[_EYE_R] - lm[_EYE_L]))
        if iod < 5:
            self._prev = None
            return None

        rois = {
            name: cv2.convexHull(lm[ROI_LANDMARKS[name]].round().astype(np.int32)).reshape(-1, 2)
            for name in self.roi_names
        }
        pts = np.concatenate(list(rois.values()))
        x0, y0 = np.maximum(pts.min(axis=0), 0)
        x1, y1 = np.minimum(pts.max(axis=0) + 1, [w, h])
        if x1 - x0 < 4 or y1 - y0 < 4:
            self._prev = None
            return None

        crop = frame_bgr[y0:y1, x0:x1]
        mask = np.zeros(crop.shape[:2], np.uint8)
        offset = np.array([x0, y0], dtype=np.int32)
        for poly in rois.values():
            cv2.fillConvexPoly(mask, poly - offset, 255)
        # Drop deep shadows and specular highlights / clipped pixels.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mask[(gray < 20) | (gray > 245)] = 0
        n_px = cv2.countNonZero(mask)
        if n_px < 50:
            self._prev = None
            return None
        b, g, r, _ = cv2.mean(crop, mask=mask)

        motion = 0.0
        if self._prev is not None and t > self._prev[0]:
            pt, plm, piod = self._prev
            disp = float(np.linalg.norm(lm - plm, axis=1).mean())
            motion = disp / ((iod + piod) / 2) / (t - pt)
        self._prev = (t, lm, iod)

        return FaceObservation(
            rgb=np.array([r, g, b]),
            landmarks=lm,
            rois=rois,
            motion=motion,
            center_y=float(lm[:, 1].mean() / iod),
            iod=iod,
            n_pixels=n_px,
        )


ROI_COLORS = {"forehead": (80, 220, 255), "right_cheek": (120, 255, 120), "left_cheek": (120, 255, 120)}


def draw_rois(frame_bgr: np.ndarray, obs: FaceObservation | None, color=None) -> np.ndarray:
    """Draw ROI outlines onto a copy of the frame."""
    out = frame_bgr.copy()
    if obs is None:
        return out
    for name, poly in obs.rois.items():
        c = color or ROI_COLORS.get(name, (255, 255, 255))
        cv2.polylines(out, [poly], True, c, 1, cv2.LINE_AA)
    return out
