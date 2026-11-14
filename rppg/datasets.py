"""Ground-truth loaders for validation.

All loaders return GroundTruth with times in seconds from the start of the
video, plus a contact-PPG waveform and/or device heart-rate readings.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dsp import HR_BAND, bandpass, resample_uniform, spectral_peak


@dataclass
class GroundTruth:
    ppg_t: np.ndarray | None = None
    ppg: np.ndarray | None = None
    hr_t: np.ndarray | None = None
    hr: np.ndarray | None = None

    def reference_bpm(self, t_end: float, window: float, prefer: str = "ppg") -> float | None:
        """Reference HR over [t_end - window, t_end].

        With prefer="ppg" the contact PPG goes through the same spectral
        estimator as the camera signal (the usual convention in rPPG papers).
        Otherwise device HR readings are averaged.
        """
        t0 = t_end - window
        if prefer == "ppg" and self.ppg is not None:
            m = (self.ppg_t >= t0) & (self.ppg_t <= t_end)
            if m.sum() > 10:
                tt, x = self.ppg_t[m], self.ppg[m]
                fs = 1.0 / np.median(np.diff(tt))
                fs = min(fs, 256.0)
                tu, xu = resample_uniform(tt, x, fs)
                if len(xu) > 30:
                    return spectral_peak(bandpass(xu, fs, *HR_BAND), fs, HR_BAND).per_minute
        if self.hr is not None:
            m = (self.hr_t >= t0) & (self.hr_t <= t_end) & (self.hr > 0)
            if m.any():
                return float(self.hr[m].mean())
        return None


def load_ubfc(path: str | Path) -> GroundTruth:
    """UBFC-rPPG. DATASET_2: ground_truth.txt (rows: PPG, HR, time[s]).
    DATASET_1: gtdump.xmp (CSV: time[ms], HR, SpO2, PPG)."""
    path = Path(path)
    if path.suffix == ".xmp":
        a = np.loadtxt(path, delimiter=",")
        t = a[:, 0] / 1000.0
        t -= t[0]
        return GroundTruth(t, a[:, 3], t, a[:, 1])
    rows = [np.array(line.split(), dtype=float) for line in path.read_text().strip().splitlines()]
    ppg, hr, t = rows[0], rows[1], rows[2]
    t = t - t[0]
    return GroundTruth(t, ppg, t, hr)


def load_pure(json_path: str | Path) -> GroundTruth:
    """PURE: <seq>.json with /FullPackage (pulse oximeter) and /Image timestamps (ns)."""
    d = json.loads(Path(json_path).read_text())
    t0 = d["/Image"][0]["Timestamp"] if d.get("/Image") else d["/FullPackage"][0]["Timestamp"]
    pkg = d["/FullPackage"]
    t = (np.array([p["Timestamp"] for p in pkg], dtype=np.int64) - t0) / 1e9
    ppg = np.array([p["Value"]["waveform"] for p in pkg], dtype=float)
    hr = np.array([p["Value"]["pulseRate"] for p in pkg], dtype=float)
    return GroundTruth(t, ppg, t, hr)


def load_csv(path: str | Path) -> GroundTruth:
    """Generic CSV with header `t,bpm` (seconds from video start, e.g. from a
    pulse oximeter or smartwatch you read off while recording)."""
    with open(path, newline="") as f:
        rows = [(float(r["t"]), float(r["bpm"])) for r in csv.DictReader(f)]
    a = np.array(rows)
    return GroundTruth(hr_t=a[:, 0], hr=a[:, 1])


def load_auto(path: str | Path) -> GroundTruth:
    path = Path(path)
    if path.suffix == ".json":
        return load_pure(path)
    if path.suffix == ".csv":
        return load_csv(path)
    return load_ubfc(path)


def find_ubfc_subjects(root: str | Path) -> list[tuple[str, Path, Path]]:
    """(name, video, gt) for every UBFC subject folder under root."""
    out = []
    for vid in sorted(Path(root).rglob("vid.avi")):
        gt = next((p for p in (vid.parent / "ground_truth.txt", vid.parent / "gtdump.xmp") if p.exists()), None)
        if gt:
            out.append((vid.parent.name, vid, gt))
    return out


def find_pure_sequences(root: str | Path) -> list[tuple[str, Path, Path]]:
    """(name, frame_folder, json) for every PURE sequence (e.g. 01-01) under root."""
    out = []
    for js in sorted(Path(root).rglob("*.json")):
        if not re.fullmatch(r"\d{2}-\d{2}", js.stem):
            continue
        frames = js.parent / js.stem
        if not frames.is_dir():
            frames = js.parent
        out.append((js.stem, frames, js))
    return out
