"""Live webcam heart-rate / breathing-rate demo in an OpenCV window.

  python -m rppg.live                 # webcam 0
  python -m rppg.live --synthetic     # no camera needed
  python -m rppg.live --video clip.mp4

Keys: q / Esc quit,  m cycle method (POS -> CHROM -> GREEN),  r reset.
"""

from __future__ import annotations

import argparse
import csv
import time

import cv2
import numpy as np

from .estimator import EstimatorConfig, VitalsEstimator, VitalsState
from .face import FaceROITracker, draw_rois
from .sources import camera_frames, open_camera, video_frames
from .synthetic import SyntheticSpec, synthetic_frames

DISCLAIMER = "Not a medical device. Accuracy varies with light, motion, skin tone."
STATUS_COLOR = {"ok": (90, 220, 90), "low_signal": (0, 200, 255), "motion": (60, 60, 255),
                "no_face": (60, 60, 255), "warming_up": (220, 220, 220)}


def _plot_strip(wave: np.ndarray | None, width: int, height: int, color=(80, 200, 255)) -> np.ndarray:
    strip = np.full((height, width, 3), 24, np.uint8)
    if wave is None or len(wave) < 2:
        return strip
    w = np.clip(wave, -3, 3)
    xs = np.linspace(0, width - 1, len(w))
    ys = height / 2 - w / 3 * (height / 2 - 4)
    pts = np.column_stack([xs, ys]).round().astype(np.int32)
    cv2.polylines(strip, [pts], False, color, 1, cv2.LINE_AA)
    return strip


def render(frame: np.ndarray, obs, state: VitalsState | None, method: str) -> np.ndarray:
    out = draw_rois(frame, obs)
    h, w = out.shape[:2]
    panel = np.full((172, w, 3), 16, np.uint8)
    if state is not None:
        hr = f"{state.hr_bpm:.0f}" if state.hr_bpm else "--"
        rr = f"{state.rr_bpm:.0f}" if state.rr_bpm else "--"
        col = STATUS_COLOR.get(state.hr.status, (255, 255, 255))
        cv2.putText(panel, f"HR {hr} bpm", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f"RR {rr} /min", (220, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 80), 2, cv2.LINE_AA)
        snr = f"{state.hr.snr_db:.1f} dB" if state.hr.snr_db is not None else "-"
        info = f"{method}  SNR {snr}  {state.fps:.1f} fps"
        cv2.putText(panel, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(panel, state.message, (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        if state.fps and (state.fps < 24 or state.fps_jitter > 0.25):
            cv2.putText(panel, "Unstable frame rate - results unreliable", (w // 2, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1, cv2.LINE_AA)
        panel[92:146] = _plot_strip(state.hr.wave, w, 54)
    cv2.putText(panel, DISCLAIMER, (10, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 255), 1, cv2.LINE_AA)
    return np.vstack([out, panel])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0)
    src.add_argument("--video")
    src.add_argument("--synthetic", action="store_true")
    ap.add_argument("--method", default="POS", choices=["POS", "CHROM", "GREEN"])
    ap.add_argument("--fourcc", default="YUYV", help="YUYV (uncompressed, default) or MJPG")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--lock-exposure", action="store_true", help="disable auto exposure / white balance")
    ap.add_argument("--motion-threshold", type=float, default=1.0)
    ap.add_argument("--log", help="append one CSV row per second: time, HR, RR, status, SNR")
    a = ap.parse_args(argv)

    cfg = EstimatorConfig(method=a.method, motion_threshold=a.motion_threshold)
    est = VitalsEstimator(cfg)
    cap = None
    if a.synthetic:
        frames = synthetic_frames(SyntheticSpec(duration=600))
        realtime = True
    elif a.video:
        frames = video_frames(a.video, realtime=True)
        realtime = False
    else:
        cap = open_camera(a.camera, a.width, a.height, 30, a.fourcc, a.lock_exposure)
        frames = camera_frames(cap)
        realtime = False

    log_f = log_w = None
    if a.log:
        log_f = open(a.log, "a", newline="")
        log_w = csv.writer(log_f)
        if log_f.tell() == 0:
            log_w.writerow(["wall_time", "t", "hr_bpm", "rr_bpm", "status", "snr_db", "fps", "method"])

    methods = ["POS", "CHROM", "GREEN"]
    state = None
    last_update = last_log = -1.0
    start = time.monotonic()
    win = "rPPG vitals (q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        with FaceROITracker() as tracker:
            for t, frame in frames:
                if realtime:  # pace the synthetic generator to wall-clock
                    lag = t - (time.monotonic() - start)
                    if lag > 0:
                        time.sleep(lag)
                obs = tracker.process(frame, t)
                est.push_observation(t, obs)
                if t - last_update >= 0.5:
                    state = est.update()
                    last_update = t
                    if log_w and t - last_log >= 1.0:
                        log_w.writerow([f"{time.time():.3f}", f"{t:.2f}",
                                        f"{state.hr_bpm:.1f}" if state.hr_bpm else "",
                                        f"{state.rr_bpm:.1f}" if state.rr_bpm else "",
                                        state.hr.status,
                                        f"{state.hr.snr_db:.2f}" if state.hr.snr_db is not None else "",
                                        f"{state.fps:.1f}", est.cfg.method])
                        log_f.flush()
                        last_log = t
                cv2.imshow(win, render(frame, obs, state, est.cfg.method))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("m"):
                    est.cfg.method = methods[(methods.index(est.cfg.method) + 1) % len(methods)]
                if key == ord("r"):
                    est.reset()
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        if cap is not None:
            cap.release()
        if log_f:
            log_f.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
