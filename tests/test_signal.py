"""Signal-level tests on synthetic RGB traces (fast, no camera / MediaPipe)."""

import numpy as np
import pytest

from rppg.dsp import HR_BAND, RR_BAND, bandpass, resample_uniform, spectral_peak
from rppg.estimator import EstimatorConfig, VitalsEstimator, analyze_hr, analyze_rr
from rppg.methods import METHODS

SIG = np.array([0.33, 0.77, 0.53])  # pulsatile strength per channel (R, G, B)


def make_rgb(t, hr_bpm=72.0, amp=0.005, noise=0.002, light_drift=0.02, motion_artifact=0.0, seed=0):
    """Mean skin colour with a pulse, slow illumination drift, noise and an
    optional broadband 'motion' artifact that is the same in all channels."""
    rng = np.random.default_rng(seed)
    pulse = np.sin(2 * np.pi * hr_bpm / 60 * t)
    base = np.array([180.0, 140.0, 110.0])
    light = 1 + light_drift * np.sin(2 * np.pi * 0.05 * t)
    rgb = base * light[:, None] * (1 - amp * pulse[:, None] * SIG)
    if motion_artifact:
        # intensity change (same direction in every channel) at an off-HR frequency
        rgb *= (1 + motion_artifact * np.sin(2 * np.pi * 1.9 * t))[:, None]
    rgb *= 1 + rng.normal(0, noise, rgb.shape)
    return rgb, pulse


@pytest.mark.parametrize("method", ["POS", "CHROM", "GREEN"])
@pytest.mark.parametrize("hr", [50, 72, 110, 150])
def test_methods_recover_heart_rate(method, hr):
    fs = 30
    t = np.arange(0, 10, 1 / fs)
    rgb, _ = make_rgb(t, hr)
    pulse = bandpass(METHODS[method](rgb, fs), fs, *HR_BAND)
    est = spectral_peak(pulse, fs, HR_BAND).per_minute
    assert abs(est - hr) < 2.0


@pytest.mark.parametrize("method", ["POS", "CHROM", "GREEN"])
def test_output_has_blood_volume_polarity(method):
    fs = 30
    t = np.arange(0, 10, 1 / fs)
    rgb, pulse = make_rgb(t, noise=0.0005)
    out = bandpass(METHODS[method](rgb, fs), fs, *HR_BAND)
    assert np.corrcoef(out, pulse)[0, 1] > 0.8


@pytest.mark.parametrize("method", ["POS", "CHROM"])
def test_chrominance_methods_reject_intensity_artifact(method):
    """A strong brightness flicker (e.g. head moving under a lamp) swamps GREEN
    but POS/CHROM project it out."""
    fs = 30
    t = np.arange(0, 10, 1 / fs)
    rgb, _ = make_rgb(t, 72, motion_artifact=0.02)
    green_est = spectral_peak(bandpass(METHODS["GREEN"](rgb, fs), fs, *HR_BAND), fs, HR_BAND).per_minute
    est = spectral_peak(bandpass(METHODS[method](rgb, fs), fs, *HR_BAND), fs, HR_BAND).per_minute
    assert abs(green_est - 1.9 * 60) < 3  # GREEN locks onto the artifact
    assert abs(est - 72) < 2


def test_irregular_frame_timing_is_handled():
    """Webcams deliver jittery timestamps; resampling should absorb it."""
    rng = np.random.default_rng(1)
    dt = np.clip(rng.normal(1 / 27, 0.008, 400), 0.015, None)
    t = np.cumsum(dt)
    t = t[t <= 10.5]
    rgb, _ = make_rgb(t, 84)
    r = analyze_hr(t, rgb, np.zeros(len(t)), EstimatorConfig(method="POS"))
    assert r.status == "ok"
    assert abs(r.value - 84) < 2


def test_motion_window_is_rejected():
    t = np.arange(0, 10, 1 / 30)
    rgb, _ = make_rgb(t)
    motion = np.zeros(len(t))
    motion[100:130] = 3.0  # 1 s of head shaking = 10% of the window
    r = analyze_hr(t, rgb, motion, EstimatorConfig())
    assert r.status == "motion" and r.value is None


def test_missing_face_is_rejected():
    t = np.arange(0, 10, 1 / 30)
    rgb, _ = make_rgb(t)
    rgb[50:150] = np.nan
    r = analyze_hr(t, rgb, np.zeros(len(t)), EstimatorConfig())
    assert r.status == "no_face"


def test_breathing_rate_from_intensity():
    fs = 30
    t = np.arange(0, 32, 1 / fs)
    rgb, _ = make_rgb(t, 70)
    rgb *= (1 + 0.004 * np.sin(2 * np.pi * (12 / 60) * t))[:, None]
    r = analyze_rr(t, rgb, np.zeros(len(t)), np.full(len(t), 5.0), EstimatorConfig())
    assert r.status == "ok"
    assert abs(r.value - 12) < 1.5


def test_breathing_rate_from_head_motion():
    fs = 30
    t = np.arange(0, 32, 1 / fs)
    rgb, _ = make_rgb(t, 70)
    y = 5 + 0.02 * np.sin(2 * np.pi * (18 / 60) * t)
    r = analyze_rr(t, rgb, np.zeros(len(t)), y, EstimatorConfig())
    assert r.source == "head_motion"
    assert abs(r.value - 18) < 1.5


def test_streaming_estimator_warms_up_then_reports():
    est = VitalsEstimator(EstimatorConfig(method="POS"))
    t = np.arange(0, 12, 1 / 30)
    rgb, _ = make_rgb(t, 66)
    states = []
    for i in range(len(t)):
        est.push(t[i], rgb[i], 0.0, 5.0)
        if i % 30 == 29:
            states.append(est.update())
    assert states[0].hr.status == "warming_up"
    assert states[-1].hr.status == "ok"
    assert abs(states[-1].hr_bpm - 66) < 2


def test_resample_uniform_shapes():
    t = np.array([0.0, 0.03, 0.07, 0.1])
    tu, x = resample_uniform(t, np.ones((4, 3)), 30)
    assert x.shape == (len(tu), 3)
    assert np.allclose(np.diff(tu), 1 / 30)


def test_snr_higher_for_clean_signal():
    fs = 30
    t = np.arange(0, 10, 1 / fs)
    clean = np.sin(2 * np.pi * 1.2 * t)
    noisy = clean + np.random.default_rng(0).normal(0, 2, len(t))
    assert spectral_peak(clean, fs, HR_BAND).snr_db > spectral_peak(noisy, fs, HR_BAND).snr_db + 5


def test_rr_band_constant():
    assert RR_BAND == (0.1, 0.5)
