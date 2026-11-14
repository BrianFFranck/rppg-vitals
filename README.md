# rPPG Vitals: heart rate and breathing from a webcam

Measures **heart rate** and **breathing rate** without touching the person, using an
ordinary webcam. This is *remote photoplethysmography* (rPPG): every heartbeat slightly
changes the colour of your skin, too little to see but enough for a camera to pick up.

> ⚠️ **This is not a medical device.** It is a learning and research demo. Accuracy
> depends on lighting, motion, camera frame rate and **skin tone**. Do not use it for any
> health decision.

---

## Where to test it

Everything runs from the project folder:

```bash
cd ~/Desktop/rppg-vitals
uv sync                      # first time only: creates .venv and installs everything
```

### 1. Browser app (recommended)

```bash
uv run streamlit run app.py
```

Then open **http://localhost:8501**. In the sidebar:

1. Pick **Webcam**, or **Synthetic test face** if you just want to see it work.
2. Press **▶ Start**.
3. Sit about 50 cm from the camera, facing it, with steady, even light on your face. Keep
   still for about 10 s; heart rate shows up after ~10 s and breathing after ~30 s.
4. Optional: put on a pulse oximeter or smartwatch, then use **Compare with a pulse
   oximeter / smartwatch** at the bottom. Type the device's reading, press **Log**, and the
   app records it next to its own estimate and shows the running error (MAE). Download the
   readings as a CSV.

You'll see the face with the tracked skin regions (forehead and both cheeks), HR, breathing
rate, signal quality (SNR), measured fps, the pulse waveform, its spectrum and a history plot.

### 2. Lightweight OpenCV window

```bash
uv run rppg-live                       # webcam 0
uv run rppg-live --synthetic           # no camera needed
uv run rppg-live --camera 1 --method CHROM --log session.csv
```

Keys: `q` quit · `m` cycle POS → CHROM → GREEN · `r` reset.

### 3. Validation from the command line

```bash
uv run rppg-eval synthetic --method all --plot results/synthetic.png
```

Runs the full pipeline on a synthetic face video with a known heart rate and prints
MAE/RMSE/correlation, plus a scatter and Bland-Altman plot. See [Validation](#validation)
for real datasets.

### 4. Unit tests

```bash
uv run pytest                 # 27 tests, ~25 s
uv run pytest -m "not slow"   # skip the MediaPipe end-to-end test
```

---

## How it works

```
webcam frame ──► MediaPipe Face Landmarker (478 points)
                   │
                   ├─► forehead + left/right cheek polygons ──► mean R,G,B per frame
                   │                                             (shadows/highlights dropped)
                   └─► landmark speed ──► motion score per frame

per-frame RGB + real timestamps ──► resample to uniform 30 Hz
     │
     ├─ Heart rate (10 s sliding window)
     │    POS / CHROM / GREEN ──► band-pass 0.7–4 Hz ──► FFT peak × 60 = BPM
     │    reject window if >5% of frames move too much or face missing >20%
     │    median of last 5 good windows
     │
     └─ Breathing (30 s window)
          skin-brightness variation  ─┐
          vertical head movement      ├► band-pass 0.1–0.5 Hz ──► FFT peak × 60
                                      └ the one with the cleaner peak wins
```

| Method | Idea | Notes |
|---|---|---|
| **POS** (default) | Projects normalised RGB onto a plane orthogonal to skin tone, which cancels most brightness changes. | Wang et al., IEEE TBME 2017 |
| **CHROM** | Combines two chrominance signals so specular/brightness changes cancel. | de Haan & Jeanne, IEEE TBME 2013 |
| **GREEN** | Just the green channel. | Baseline only; easily fooled by light changes and motion |

**Signal quality (SNR)**: power at the heart-rate peak (plus its first harmonic) compared
with everything else in the 0.7–4 Hz band, in dB. As a rough guide, above ~3 dB is a clean
signal and below ~0 dB the number is questionable. The app shows "Weak signal" below −3 dB
(adjustable).

**Motion rejection**: each frame's motion is the mean landmark speed, in eye-widths per
second. Normal sitting still is around 0.05; talking or turning your head goes above 1. A
heart-rate window with more than 5% moving frames is discarded rather than guessed. For
breathing, moving frames are dropped and filled in across, so one brief movement doesn't
cost 30 s of data.

**Unstable frame rate**: timestamps come from the real capture time, not the nominal 30 fps,
so jittery frames are resampled correctly. The app warns below 24 fps or with high jitter.

---

## Validation

### Against your own pulse oximeter or smartwatch

- **Live:** use the comparison panel in the browser app (see above).
- **Offline:** record a video of yourself (keep it uncompressed or high quality: heavy
  compression destroys the signal) and write a CSV of reference readings, with times in
  seconds from the start of the video:

  ```csv
  t,bpm
  10,72
  20,74
  30,73
  ```

  ```bash
  uv run rppg-eval video my_clip.avi --gt my_readings.csv --ref hr --method all
  ```

Fingertip pulse oximeters are the better reference. Smartwatches are also optical (PPG)
and lag during changes.

### Public datasets

Datasets are free for research, but you have to request or download them yourself. Put them
under `data/` (already git-ignored).

| Dataset | What | Get it |
|---|---|---|
| **UBFC-rPPG** | 42 subjects, 640×480 @ 30 fps, contact PPG reference | Search "UBFC-rPPG dataset" (Université Bourgogne Franche-Comté), download link on the project page |
| **PURE** | 10 subjects × 6 motion conditions, lossless PNG frames, pulse oximeter | Request from TU Ilmenau (Stricker et al., 2014) |

```bash
uv run rppg-eval ubfc data/UBFC-rPPG/DATASET_2 --method all --out results/ubfc.csv --plot results/ubfc.png
uv run rppg-eval pure data/PURE --method all --out results/pure.csv --plot results/pure.png
```

Expected folder layouts (the loaders search recursively):

```
UBFC-rPPG/DATASET_2/subject1/vid.avi + ground_truth.txt      (DATASET_1: gtdump.xmp)
PURE/01-01/01-01.json + PURE/01-01/01-01/Image<ns>.png ...
```

By default the reference heart rate for each window comes from the **contact PPG waveform
passed through the same spectral estimator**, which is the usual convention in rPPG papers.
Use `--ref hr` to average the device's HR readings instead.

Metrics reported: MAE, RMSE, MAPE, Pearson r, % of windows within 5 BPM, Bland-Altman bias
and 95% limits of agreement, and **coverage** (share of windows not rejected for motion or a
missing face). Always report coverage next to MAE: rejecting hard windows lowers MAE.

### Cross-checking against established toolkits

This project implements POS/CHROM/GREEN directly in NumPy/SciPy so the code is easy to read.
To check results against reference implementations, run the same videos through:

- [**rPPG-Toolbox**](https://github.com/ubicomplab/rPPG-Toolbox): many unsupervised and
  deep-learning methods, with UBFC/PURE loaders.
- [**pyVHR**](https://github.com/phuselab/pyVHR): a Python framework for virtual heart rate.

Both are heavy installs and are not required here.

### Results so far

| Data | Method | MAE (BPM) | RMSE | Coverage |
|---|---|---|---|---|
| Synthetic face, 60 s, 68–76 BPM, 4 s head-shake | POS | 0.22 | 0.25 | 74% (shake windows rejected) |
| | CHROM | 0.23 | 0.26 | 74% |
| | GREEN | 0.51 | 0.65 | 74% |

The synthetic video is far easier than real footage: a still photo with an injected pulse
and no real lighting changes. These numbers only show that the pipeline is wired
correctly. **They say nothing about real-world accuracy.** Run UBFC-rPPG and your own
oximeter comparison for that.

---

## Limitations and pitfalls (please state these in any demo)

- **Not a medical device.** No clinical validation.
- **Skin tone:** melanin absorbs more of the light that carries the pulse signal, so the
  signal is weaker on darker skin and errors are typically larger. UBFC-rPPG and PURE are
  both mostly lighter-skinned subjects, so good results on them don't show that it works
  equally well for everyone.
- **Lighting:** needs steady, even, reasonably bright light. Flickering LEDs, screens
  lighting your face, mixed daylight, and backlighting all hurt. Auto-exposure and auto
  white-balance steps look like huge signal jumps, so lock them once the light is set
  (sidebar checkbox, or `--lock-exposure`).
- **Frame rate:** needs a stable ~30 fps. Many webcams lower the fps in dim light. On Linux
  you can stop that:
  ```bash
  v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0
  ```
  (Your HP True Vision camera currently has this **on**.)
- **Compression:** the app asks the camera for uncompressed YUYV. MJPEG/H.264 compression
  can wipe out sub-1% colour changes.
- **Motion and talking** corrupt the signal. Those windows are rejected, not fixed.
- **Breathing rate** is experimental: it needs ~30 s of stillness, and frequency resolution
  over 30 s is about ±1–2 breaths/min.
- Heart rate is an average over the 10 s window. It is not beat-to-beat and gives no HRV.

---

## Project layout

```
app.py                Streamlit browser app
rppg/
  face.py             MediaPipe Face Landmarker, ROI polygons, mean RGB, motion score
  methods.py          POS, CHROM, GREEN
  dsp.py              resampling, band-pass, spectral peak + SNR
  estimator.py        sliding windows, motion/face gating, HR + RR, smoothing
  sources.py          webcam / video file / PURE frame readers
  datasets.py         UBFC-rPPG, PURE and CSV ground truth
  evaluate.py         offline validation CLI (rppg-eval)
  live.py             OpenCV live window (rppg-live)
  synthetic.py        synthetic face video with known HR/RR
tests/                signal-level tests + end-to-end synthetic test
assets/astronaut.png  test face (NASA portrait of Eileen Collins, public domain; via scikit-image)
models/               MediaPipe face model (auto-downloaded on first run)
```

## Troubleshooting

- **"Could not open camera"**: another app (browser tab, video call) has the webcam. Close
  it, or try `--camera 1` or the camera index in the sidebar.
- **Camera works but fps is ~15**: switch the format to MJPG, add light, or disable
  `exposure_dynamic_framerate` (see above).
- **OpenCV window doesn't appear on Wayland**: `QT_QPA_PLATFORM=xcb uv run rppg-live`.
- **Model download fails**: download `face_landmarker.task` manually from the MediaPipe
  model page and put it in `models/`.

Python 3.12 is pinned in `.python-version` because MediaPipe wheels lag behind new Python
releases.
