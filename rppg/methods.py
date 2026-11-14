"""rPPG pulse-extraction methods.

Every method takes `rgb`, an (N, 3) array of per-frame mean skin colour in
R, G, B order sampled uniformly at `fs` Hz, and returns an (N,) pulse signal
with blood-volume polarity (peaks ~ systole). Output is not yet band-passed.

References
----------
POS   : Wang, den Brinker, Stuijk, de Haan. "Algorithmic Principles of Remote
        PPG." IEEE TBME 2017.
CHROM : de Haan, Jeanne. "Robust Pulse Rate From Chrominance-Based rPPG."
        IEEE TBME 2013.
GREEN : Verkruysse, Svaasand, Nelson. "Remote plethysmographic imaging using
        ambient light." Optics Express 2008.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter1d

from .dsp import HR_BAND, bandpass

_EPS = 1e-9


def _win_len(fs: float, win_sec: float, n: int) -> int:
    return int(min(n, max(2, round(win_sec * fs))))


def pos(rgb: np.ndarray, fs: float, win_sec: float = 1.6) -> np.ndarray:
    """Plane-Orthogonal-to-Skin (POS), overlap-added over short windows."""
    x = np.asarray(rgb, dtype=float)
    n = len(x)
    l = _win_len(fs, win_sec, n)
    # windows: (n-l+1, 3, l)
    w = sliding_window_view(x, l, axis=0)
    cn = w / (w.mean(axis=2, keepdims=True) + _EPS)  # temporal normalisation
    s1 = cn[:, 1] - cn[:, 2]  # G - B
    s2 = -2 * cn[:, 0] + cn[:, 1] + cn[:, 2]  # -2R + G + B
    alpha = s1.std(axis=1, keepdims=True) / (s2.std(axis=1, keepdims=True) + _EPS)
    h = s1 + alpha * s2
    h -= h.mean(axis=1, keepdims=True)

    out = np.zeros(n)
    for i in range(l):  # overlap-add: loop over window offset, not windows
        out[i : i + len(h)] += h[:, i]
    # POS projection yields the inverted blood-volume signal; flip it.
    return -out


def chrom(rgb: np.ndarray, fs: float, win_sec: float = 1.6) -> np.ndarray:
    """Chrominance (CHROM) with Hann-windowed overlap-add."""
    x = np.asarray(rgb, dtype=float)
    n = len(x)
    l = _win_len(fs, win_sec, n)
    l += l % 2  # even, so 50% overlap is exact
    l = min(l, n)

    cn = x / (uniform_filter1d(x, size=l, axis=0, mode="nearest") + _EPS)
    xs = 3 * cn[:, 0] - 2 * cn[:, 1]
    ys = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
    xf = bandpass(xs, fs, *HR_BAND)
    yf = bandpass(ys, fs, *HR_BAND)

    out = np.zeros(n)
    hann = np.hanning(l)
    step = max(1, l // 2)
    starts = list(range(0, n - l + 1, step))
    if starts[-1] != n - l:
        starts.append(n - l)
    for s in starts:
        xw, yw = xf[s : s + l], yf[s : s + l]
        alpha = xw.std() / (yw.std() + _EPS)
        out[s : s + l] += (xw - alpha * yw) * hann
    return out


def green(rgb: np.ndarray, fs: float) -> np.ndarray:
    """Baseline: normalised green channel (inverted: more blood = darker)."""
    g = np.asarray(rgb, dtype=float)[:, 1]
    return -(g / (g.mean() + _EPS) - 1.0)


METHODS = {"POS": pos, "CHROM": chrom, "GREEN": green}
