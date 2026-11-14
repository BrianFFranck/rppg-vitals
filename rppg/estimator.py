"""Sliding-window heart-rate and breathing-rate estimation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .dsp import HR_BAND, RR_BAND, bandpass, resample_uniform, spectral_peak
from .methods import METHODS


@dataclass
class EstimatorConfig:
    method: str = "POS"  # POS | CHROM | GREEN
    fs: float = 30.0  # uniform resampling rate for the pulse signal
    hr_window: float = 10.0  # seconds
    rr_window: float = 30.0  # breathing is slow; needs a longer window
    rr_fs: float = 10.0
    motion_threshold: float = 1.0  # landmark speed, inter-ocular distances / s
    max_motion_fraction_hr: float = 0.05  # reject window if >5% of frames move too much
    max_motion_fraction_rr: float = 0.25  # RR drops moving frames instead (see analyze_rr)
    min_face_fraction: float = 0.8  # face must be found in >=80% of frames
    min_snr_db: float = -3.0  # below this the HR estimate is flagged low_signal
    min_rr_snr_db: float = 0.0
    smoothing: int = 5  # median over this many accepted HR windows


@dataclass
class WindowResult:
    status: str  # ok | low_signal | motion | no_face | warming_up
    value: float | None = None  # BPM or breaths/min
    snr_db: float | None = None
    t: np.ndarray | None = None  # uniform time grid of the window
    wave: np.ndarray | None = None  # filtered, z-scored signal
    freqs: np.ndarray | None = None
    power: np.ndarray | None = None
    motion_fraction: float = 0.0
    face_fraction: float = 0.0
    source: str = ""  # for RR: which signal won


def _gate(t, rgb, motion, window, cfg, max_motion_frac):
    """Shared checks. Returns (status, valid_mask, face_frac, motion_frac)."""
    t = np.asarray(t, dtype=float)
    if len(t) < 3 or t[-1] - t[0] < 0.9 * window:
        return "warming_up", None, 0.0, 0.0
    valid = np.isfinite(np.asarray(rgb, dtype=float)[:, 0])
    face_frac = float(valid.mean())
    if face_frac < cfg.min_face_fraction or valid.sum() < 3:
        return "no_face", valid, face_frac, 0.0
    mot = np.asarray(motion, dtype=float)[valid]
    motion_frac = float((mot > cfg.motion_threshold).mean())
    if motion_frac > max_motion_frac:
        return "motion", valid, face_frac, motion_frac
    return "ok", valid, face_frac, motion_frac


def analyze_hr(t, rgb, motion, cfg: EstimatorConfig) -> WindowResult:
    """Heart rate from one window of per-frame mean RGB (NaN rows = no face)."""
    status, valid, ff, mf = _gate(t, rgb, motion, cfg.hr_window, cfg, cfg.max_motion_fraction_hr)
    if status != "ok":
        return WindowResult(status, face_fraction=ff, motion_fraction=mf)
    tv = np.asarray(t, dtype=float)[valid]
    tu, rgb_u = resample_uniform(tv, np.asarray(rgb, dtype=float)[valid], cfg.fs)
    pulse = METHODS[cfg.method](rgb_u, cfg.fs)
    pulse = bandpass(pulse, cfg.fs, *HR_BAND)
    peak = spectral_peak(pulse, cfg.fs, HR_BAND)
    wave = (pulse - pulse.mean()) / (pulse.std() + 1e-12)
    status = "ok" if peak.snr_db >= cfg.min_snr_db else "low_signal"
    return WindowResult(status, peak.per_minute, peak.snr_db, tu, wave, peak.freqs, peak.power, mf, ff)


def analyze_rr(t, rgb, motion, center_y, cfg: EstimatorConfig) -> WindowResult:
    """Breathing rate. Two candidate signals, the cleaner spectral peak wins:

    * intensity: respiration-induced skin-brightness variation (RIIV)
    * head_motion: tiny vertical head movement with each breath
    """
    status, valid, ff, mf = _gate(t, rgb, motion, cfg.rr_window, cfg, cfg.max_motion_fraction_rr)
    if status != "ok":
        return WindowResult(status, face_fraction=ff, motion_fraction=mf)
    # A brief movement shouldn't cost 30 s of breathing data: drop the moving
    # frames and interpolate across them instead of rejecting the window.
    valid = valid & (np.asarray(motion, dtype=float) <= cfg.motion_threshold)
    tv = np.asarray(t, dtype=float)[valid]
    rgb_v = np.asarray(rgb, dtype=float)[valid]
    norm = rgb_v / rgb_v.mean(axis=0)
    candidates = {
        "intensity": norm.mean(axis=1),
        "head_motion": np.asarray(center_y, dtype=float)[valid],
    }
    best = None
    for name, x in candidates.items():
        if not np.all(np.isfinite(x)):
            continue
        tu, xu = resample_uniform(tv, x, cfg.rr_fs)
        xf = bandpass(xu, cfg.rr_fs, *RR_BAND, order=2)
        peak = spectral_peak(xf, cfg.rr_fs, RR_BAND, peak_halfwidth=0.03, use_harmonic=False)
        if best is None or peak.snr_db > best[1].snr_db:
            best = (name, peak, tu, xf)
    if best is None:
        return WindowResult("no_face", face_fraction=ff, motion_fraction=mf)
    name, peak, tu, xf = best
    wave = (xf - xf.mean()) / (xf.std() + 1e-12)
    status = "ok" if peak.snr_db >= cfg.min_rr_snr_db else "low_signal"
    return WindowResult(status, peak.per_minute, peak.snr_db, tu, wave, peak.freqs, peak.power, mf, ff, name)


@dataclass
class VitalsState:
    hr: WindowResult
    rr: WindowResult
    hr_bpm: float | None  # smoothed, None until a good window exists
    rr_bpm: float | None
    fps: float
    fps_jitter: float  # std(dt)/mean(dt)
    buffer_seconds: float
    history: list[tuple[float, float | None, float | None]] = field(default_factory=list)

    @property
    def message(self) -> str:
        s = self.hr.status
        if s == "warming_up":
            return f"Collecting signal... {self.buffer_seconds:.0f}s"
        if s == "no_face":
            return "Face not found consistently - face the camera"
        if s == "motion":
            return "Too much motion - hold still"
        if s == "low_signal":
            return "Weak signal - improve lighting / hold still"
        return "Tracking"


class VitalsEstimator:
    """Streaming wrapper: push frames as they arrive, call update() ~1-2x/s."""

    def __init__(self, cfg: EstimatorConfig | None = None):
        self.cfg = cfg or EstimatorConfig()
        self._keep = max(self.cfg.hr_window, self.cfg.rr_window) + 2.0
        self._t: deque[float] = deque()
        self._rgb: deque[np.ndarray] = deque()
        self._motion: deque[float] = deque()
        self._y: deque[float] = deque()
        self._hr_hist: deque[float] = deque(maxlen=self.cfg.smoothing)
        self.history: deque[tuple[float, float | None, float | None]] = deque(maxlen=600)
        self._last_rr: float | None = None

    def reset(self) -> None:
        self.__init__(self.cfg)

    def push(self, t: float, rgb: np.ndarray | None, motion: float = 0.0, center_y: float = np.nan) -> None:
        if self._t and t <= self._t[-1]:
            return  # out-of-order / duplicate timestamp
        self._t.append(float(t))
        self._rgb.append(np.full(3, np.nan) if rgb is None else np.asarray(rgb, dtype=float))
        self._motion.append(float(motion))
        self._y.append(float(center_y) if rgb is not None else np.nan)
        while self._t and self._t[-1] - self._t[0] > self._keep:
            self._t.popleft()
            self._rgb.popleft()
            self._motion.popleft()
            self._y.popleft()

    def push_observation(self, t: float, obs) -> None:
        """Convenience for a rppg.face.FaceObservation (or None)."""
        if obs is None:
            self.push(t, None)
        else:
            self.push(t, obs.rgb, obs.motion, obs.center_y)

    def _slice(self, seconds: float):
        t = np.fromiter(self._t, float)
        i0 = np.searchsorted(t, t[-1] - seconds)
        rgb = np.array(self._rgb)[i0:]
        return t[i0:], rgb, np.fromiter(self._motion, float)[i0:], np.fromiter(self._y, float)[i0:]

    def update(self) -> VitalsState:
        cfg = self.cfg
        if len(self._t) < 3:
            empty = WindowResult("warming_up")
            return VitalsState(empty, empty, None, None, 0.0, 0.0, 0.0)

        t_all = np.fromiter(self._t, float)
        dt = np.diff(t_all[-min(len(t_all), 91):])  # last ~3 s
        fps = 1.0 / dt.mean() if dt.size and dt.mean() > 0 else 0.0
        jitter = float(dt.std() / dt.mean()) if dt.size and dt.mean() > 0 else 0.0
        span = t_all[-1] - t_all[0]

        t, rgb, mot, _ = self._slice(cfg.hr_window)
        hr = analyze_hr(t, rgb, mot, cfg)
        if hr.status == "ok":
            self._hr_hist.append(hr.value)
        elif hr.status in ("no_face",):
            self._hr_hist.clear()
        hr_bpm = float(np.median(self._hr_hist)) if self._hr_hist and hr.status in ("ok", "low_signal") else None

        t, rgb, mot, y = self._slice(cfg.rr_window)
        rr = analyze_rr(t, rgb, mot, y, cfg)
        if rr.status == "ok":
            self._last_rr = rr.value
        rr_bpm = self._last_rr if rr.status in ("ok", "low_signal") else None

        self.history.append((t_all[-1], hr_bpm, rr_bpm))
        return VitalsState(hr, rr, hr_bpm, rr_bpm, fps, jitter, span, list(self.history))
