"""Streamlit demo: contactless heart rate & breathing from a webcam.

    uv run streamlit run app.py      ->  http://localhost:8501
"""

from __future__ import annotations

import time
from dataclasses import asdict

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from rppg.estimator import EstimatorConfig, VitalsEstimator
from rppg.face import FaceROITracker, draw_rois
from rppg.sources import camera_frames, open_camera, video_frames
from rppg.synthetic import SyntheticSpec, synthetic_frames

st.set_page_config(page_title="rPPG Vitals", page_icon="❤️", layout="wide")

st.title("Contactless vital signs (rPPG)")
st.warning(
    "**Not a medical device.** This is a research/learning demo. Accuracy depends heavily on "
    "steady lighting, a stable ~30 fps camera, holding still, and skin tone (darker skin "
    "reflects less of the pulsatile signal, so errors are typically larger). "
    "Do not use it for any health decision."
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Source")
    source = st.radio("Input", ["Webcam", "Synthetic test face", "Video file"], label_visibility="collapsed")
    cam_index = st.number_input("Camera index", 0, 10, 0, disabled=source != "Webcam")
    fourcc = st.selectbox("Camera format", ["YUYV", "MJPG"], disabled=source != "Webcam",
                          help="YUYV is uncompressed and preserves subtle colour changes better.")
    lock_exposure = st.checkbox("Lock exposure / white balance", value=False, disabled=source != "Webcam",
                                help="Auto-exposure jumps corrupt the signal. Lock it once lighting is set.")
    video_path = st.text_input("Video path", disabled=source != "Video file")

    st.header("Algorithm")
    method = st.selectbox("Method", ["POS", "CHROM", "GREEN"],
                          help="POS and CHROM cancel most lighting/motion changes; GREEN is a naive baseline.")
    hr_window = st.slider("HR window (s)", 6, 20, 10)
    motion_thr = st.slider("Motion threshold", 0.2, 3.0, 1.0, 0.1,
                           help="Landmark speed in eye-widths per second above which a frame counts as moving.")
    min_snr = st.slider("Min SNR (dB)", -10.0, 10.0, -3.0, 0.5)

    run = st.toggle("▶ Start", value=False)

cfg = EstimatorConfig(method=method, hr_window=float(hr_window), motion_threshold=motion_thr, min_snr_db=min_snr)
cfg_key = (source, cam_index, video_path, *asdict(cfg).values())

ss = st.session_state
if ss.get("cfg_key") != cfg_key:
    ss.cfg_key = cfg_key
    ss.estimator = VitalsEstimator(cfg)
    ss.t_offset = 0.0
ss.setdefault("comparisons", [])
if "tracker" not in ss:
    ss.tracker = FaceROITracker()

# ---------------------------------------------------------------- layout
col_video, col_stats = st.columns([3, 2])
with col_video:
    frame_ph = st.empty()
with col_stats:
    m1, m2 = st.columns(2)
    hr_ph, rr_ph = m1.empty(), m2.empty()
    m3, m4 = st.columns(2)
    snr_ph, fps_ph = m3.empty(), m4.empty()
    status_ph = st.empty()
    st.caption("Pulse waveform (band-passed 0.7-4 Hz, last window)")
    wave_ph = st.empty()
    st.caption("Spectrum - the tallest peak is the heart rate")
    spec_ph = st.empty()
st.caption("Heart rate & breathing over time")
hist_ph = st.empty()

# Comparison with a reference device ------------------------------------
with st.expander("📏 Compare with a pulse oximeter / smartwatch", expanded=True):
    st.write("While tracking, type the reading from your reference device and press **Log**. "
             "The app stores it next to the current rPPG estimate.")
    c1, c2, c3 = st.columns([2, 1, 1])
    ref_val = c1.number_input("Reference HR (BPM)", 30, 220, 70)
    if c2.button("Log", width="stretch"):
        hr_now = ss.get("last_hr")
        ss.comparisons.append({"time": time.strftime("%H:%M:%S"), "method": method,
                               "reference_bpm": ref_val,
                               "rppg_bpm": round(hr_now, 1) if hr_now else None})
    if c3.button("Clear", width="stretch"):
        ss.comparisons = []
    if ss.comparisons:
        df = pd.DataFrame(ss.comparisons)
        valid = df.dropna(subset=["rppg_bpm"])
        if len(valid):
            err = valid.rppg_bpm - valid.reference_bpm
            st.write(f"**{len(valid)} readings** · MAE **{err.abs().mean():.1f} BPM** · "
                     f"bias {err.mean():+.1f} BPM")
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False), "rppg_vs_reference.csv", "text/csv")


def show_state(state):
    hr_ph.metric("Heart rate", f"{state.hr_bpm:.0f} BPM" if state.hr_bpm else "--")
    rr_ph.metric("Breathing", f"{state.rr_bpm:.0f} /min" if state.rr_bpm else "--",
                 help="Needs ~30 s of data. Experimental.")
    snr_ph.metric("Signal quality (SNR)", f"{state.hr.snr_db:.1f} dB" if state.hr.snr_db is not None else "--")
    fps_ph.metric("Frame rate", f"{state.fps:.1f} fps")
    msg = state.message
    if state.fps and (state.fps < 24 or state.fps_jitter > 0.25):
        msg += " · ⚠ unstable frame rate"
    {"ok": status_ph.success, "low_signal": status_ph.warning, "warming_up": status_ph.info}.get(
        state.hr.status, status_ph.error)(msg)
    if state.hr.wave is not None:
        wave_ph.line_chart(pd.DataFrame({"pulse": state.hr.wave}, index=state.hr.t - state.hr.t[-1]),
                           height=150)
        spec = pd.DataFrame({"power": state.hr.power / state.hr.power.max()}, index=state.hr.freqs * 60)
        spec.index.name = "BPM"
        spec_ph.area_chart(spec, height=150)
    hist = [(t, h, r) for t, h, r in state.history]
    if hist:
        df = pd.DataFrame(hist, columns=["t", "HR (BPM)", "Breathing (/min)"]).set_index("t")
        hist_ph.line_chart(df, height=220)


if not run:
    frame_ph.info("Pick a source in the sidebar and press **▶ Start**. "
                  "Sit ~50 cm from the camera, face it, with steady, even light on your face.")
    st.stop()

# ---------------------------------------------------------------- main loop
cap = None
if source == "Webcam":
    try:
        cap = open_camera(int(cam_index), 640, 480, 30, fourcc, lock_exposure)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    frames = camera_frames(cap)
elif source == "Synthetic test face":
    frames = synthetic_frames(SyntheticSpec(duration=600))
else:
    if not video_path:
        st.error("Enter a video path in the sidebar.")
        st.stop()
    frames = video_frames(video_path)

est: VitalsEstimator = ss.estimator
tracker: FaceROITracker = ss.tracker
# Streamlit re-runs the script on every widget interaction; keep the time axis
# continuous so the buffered signal survives (e.g. when you press "Log").
t_offset = est._t[-1] + 1 / 30 if est._t else 0.0
start = time.monotonic()
last_ui = last_frame_ui = -1.0
try:
    for t, frame in frames:
        if source != "Webcam":  # pace files / synthetic to real time
            lag = t - (time.monotonic() - start)
            if lag > 0:
                time.sleep(lag)
        obs = tracker.process(frame, t + t_offset)
        est.push_observation(t + t_offset, obs)
        now = time.monotonic()
        if now - last_frame_ui > 1 / 15:  # ~15 fps preview is plenty
            frame_ph.image(cv2.cvtColor(draw_rois(frame, obs), cv2.COLOR_BGR2RGB), width="stretch")
            last_frame_ui = now
        if now - last_ui > 0.5:
            state = est.update()
            ss.last_hr = state.hr_bpm
            show_state(state)
            last_ui = now
    status_ph.info("Video finished.")
finally:
    if cap is not None:
        cap.release()
