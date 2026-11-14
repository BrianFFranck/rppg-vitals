"""Synthetic test video: a still face photo with a known pulse and breathing
signal injected into the skin pixels.

This exists so you can check the whole pipeline (tracking -> ROI -> POS/CHROM
-> filtering -> BPM) end to end without a webcam or a dataset. It is far
cleaner than real video, so good numbers here do NOT prove real-world accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

DEFAULT_IMAGE = Path(__file__).resolve().parent.parent / "assets" / "astronaut.png"

# Relative pulsatile strength per channel (R, G, B); green is strongest.
# Values from the "pulse signature" used in the POS / PBV literature.
PULSE_SIGNATURE_RGB = np.array([0.33, 0.77, 0.53])


@dataclass
class SyntheticSpec:
    duration: float = 60.0
    fps: float = 30.0
    hr_bpm: float = 72.0
    hr_variation_bpm: float = 4.0  # slow sinusoidal drift around hr_bpm
    rr_bpm: float = 15.0
    pulse_amplitude: float = 0.006  # fractional skin-brightness change
    noise_sigma: float = 2.0  # per-pixel sensor noise (8-bit levels)
    shake: tuple[float, float] | None = (25.0, 29.0)  # seconds of vigorous head motion
    seed: int = 0

    def true_hr(self, t: np.ndarray | float) -> np.ndarray:
        return self.hr_bpm + self.hr_variation_bpm * np.sin(2 * np.pi * np.asarray(t) / 40.0)


def _skin_mask(img_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycrcb, (40, 135, 85), (255, 180, 135)).astype(np.float32) / 255
    return cv2.GaussianBlur(m, (0, 0), 3)[..., None]


def synthetic_frames(spec: SyntheticSpec | None = None, image: str | Path = DEFAULT_IMAGE) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (t_seconds, BGR frame)."""
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    base = cv2.imread(str(image))
    if base is None:
        raise FileNotFoundError(image)
    base_f = base.astype(np.float32)
    skin = _skin_mask(base)
    sig_bgr = PULSE_SIGNATURE_RGB[::-1].astype(np.float32)
    h, w = base.shape[:2]

    n = int(spec.duration * spec.fps)
    t = np.arange(n) / spec.fps
    phase = 2 * np.pi * np.cumsum(spec.true_hr(t) / 60.0) / spec.fps
    # PPG-like waveform: fundamental + 2nd harmonic (dicrotic shape)
    pulse = np.sin(phase) + 0.35 * np.sin(2 * phase - 0.8)
    pulse /= np.abs(pulse).max()
    breath = np.sin(2 * np.pi * spec.rr_bpm / 60.0 * t)
    light = np.cumsum(rng.normal(0, 0.0004, n))  # slow illumination drift
    light -= np.convolve(light, np.ones(150) / 150, mode="same")
    # Pre-generated sensor-noise bank (fresh Gaussian noise per frame is slow).
    noise_bank = rng.normal(0, spec.noise_sigma, (16, h, w, 3)).astype(np.float32)

    for i in range(n):
        ti = t[i]
        # Blood absorbs light: more blood volume -> darker skin.
        gain = 1.0 - spec.pulse_amplitude * pulse[i] * sig_bgr - 0.002 * breath[i]
        frame = base_f * (1 + (gain - 1) * skin) * (1 + light[i])
        frame += np.roll(noise_bank[rng.integers(16)], rng.integers(h), axis=0)

        dx, dy = 0.0, 1.5 * breath[i]  # breathing moves the head slightly
        if spec.shake and spec.shake[0] <= ti < spec.shake[1]:
            dx += 18 * np.sin(2 * np.pi * 2.0 * ti)
            dy += 10 * np.sin(2 * np.pi * 1.3 * ti)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        yield float(ti), np.clip(frame, 0, 255).astype(np.uint8)
