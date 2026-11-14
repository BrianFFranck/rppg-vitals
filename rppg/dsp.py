"""Signal-processing helpers: resampling, filtering, spectral peak picking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

# Physiological search bands (Hz)
HR_BAND = (0.7, 4.0)  # 42-240 BPM
RR_BAND = (0.1, 0.5)  # 6-30 breaths/min


def resample_uniform(t: np.ndarray, x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Linearly resample irregularly-timed samples onto a uniform grid.

    Webcams rarely deliver frames at exactly 30 fps, so everything downstream
    works on a uniform grid built from the real frame timestamps.
    `x` may be 1-D (n,) or 2-D (n, channels).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    tu = np.arange(t[0], t[-1] + 1e-9, 1.0 / fs)
    if x.ndim == 1:
        return tu, np.interp(tu, t, x)
    return tu, np.column_stack([np.interp(tu, t, x[:, c]) for c in range(x.shape[1])])


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass."""
    nyq = fs / 2.0
    hi = min(hi, 0.95 * nyq)
    sos = signal.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    x = signal.detrend(np.asarray(x, dtype=float))
    padlen = min(len(x) - 1, 3 * (2 * len(sos) + 1))
    return signal.sosfiltfilt(sos, x, padlen=padlen)


@dataclass
class Peak:
    freq_hz: float
    snr_db: float
    freqs: np.ndarray  # spectrum restricted to the search band (for plotting)
    power: np.ndarray

    @property
    def per_minute(self) -> float:
        return self.freq_hz * 60.0


def spectral_peak(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float],
    peak_halfwidth: float = 0.1,
    use_harmonic: bool = True,
    nfft_min: int = 8192,
) -> Peak:
    """Dominant frequency in `band` plus a de Haan-style SNR.

    SNR = power near the peak (and its first harmonic) / power elsewhere in
    the band, in dB. Higher is better; ~0 dB means the peak barely stands out.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    nfft = max(nfft_min, 1 << int(np.ceil(np.log2(max(n, 2)))))
    spec = np.abs(np.fft.rfft(x * np.hanning(n), nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)

    in_band = (freqs >= band[0]) & (freqs <= band[1])
    idx = np.flatnonzero(in_band)
    k = idx[np.argmax(spec[idx])]

    # Parabolic interpolation for sub-bin accuracy
    f0 = freqs[k]
    if 0 < k < len(spec) - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        denom = a - 2 * b + c
        if denom != 0:
            f0 = freqs[k] + 0.5 * (a - c) / denom * (freqs[1] - freqs[0])

    sig_mask = np.abs(freqs - f0) <= peak_halfwidth
    if use_harmonic:
        sig_mask |= np.abs(freqs - 2 * f0) <= 2 * peak_halfwidth
    sig_mask &= in_band
    noise_mask = in_band & ~sig_mask
    p_sig = spec[sig_mask].sum()
    p_noise = spec[noise_mask].sum()
    snr = 10 * np.log10(p_sig / p_noise) if p_noise > 0 and p_sig > 0 else float("inf")

    return Peak(float(f0), float(snr), freqs[in_band], spec[in_band])
