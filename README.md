# Gaze-Based Cheating Detection

Detects candidates reading from an AI overlay tool (Cluely, Parakeet AI, etc.) during
a recorded interview, from the **gaze signature** alone — repeated left→right reading
sweeps with fast return saccades, a steady line-by-line rhythm, gaze parked off-center
on an overlay region, and suppressed blinking.

Ingests a webcam video and outputs a **cheating verdict + probability + a timeline of
suspicious segments**, plus an HTML report with gaze plots for human review.

Built for an **Apple Silicon Mac (M-series)**. Pure Python; MediaPipe runs the face/iris
tracking, PyTorch uses the Metal (MPS) backend.

---

## How it works (4 stages)

```
video ──▶ [1] extract ──▶ per-frame gaze/head-pose time series  (data/features/*.parquet)
              │
              ▼
          [2] windows ──▶ ~45 behavioral features / 20s window   (data/windows/)
              │
              ▼
          [3] train  ──▶ XGBoost (A) + GRU (B), grouped CV, ship winner  (models/)
              │
              ▼
          [4] predict / report ──▶ verdict + flagged segments + HTML  (reports/)
```

- **Model A** — XGBoost on ~45 engineered window features (reading-sweep rate, saccade
  direction histogram, head–gaze coupling, gaze-grid entropy, blink rate, …). Fully
  interpretable.
- **Model B** — a compact GRU (fallback 1D-CNN) over the raw gaze time series.
- **Model C** — MiniROCKET (`--seq-kind minirocket`): a pretrained-style *fixed*
  convolutional-kernel transform (~10k kernels, nothing to download or learn) over the
  same raw gaze series, feeding a linear head. Strong on small datasets, trains in
  seconds, shipped artifact < 1 MB. Needs `pip install sktime numba`.
- The models are compared with **leakage-safe cross-validation grouped by video** (windows
  from one video never span train/test), and the winner is shipped automatically.
- With a small dataset (tens of videos), Model A usually wins and is interpretable;
  Model B wins if the data is larger.

---

## Results on the initial 19-video dataset (6 cheating / 13 clean)

Leave-one-video-out cross-validation, precision-tuned operating point:

| | Video ROC-AUC | Accuracy | False alarms |
|---|---|---|---|
| First pass (raw) | 0.59 | 68% | 4 |
| **+ calibration + gaze-only features** | **0.86** | **84%** | **0** |

Two fixes drove the gain, both on by default:
1. **Per-candidate gaze calibration** (`Config.calibrate_gaze`) — subtract each person's
   median gaze so reading/overlay behavior is measured as a *deviation from their own
   baseline*, not an absolute position.
2. **Dropping the eye/blink features** — `blink_rate`/`ear_*` separate the classes but are
   confounded by frame rate (blinks are missed at low fps). The bake-off's
   `mediapipe_gaze_head` feature set excludes them and generalizes better.

⚠️ **Honest caveat:** the cheating and clean videos came from *different recording sources*
(a homogeneous test-team capture vs varied interviews), so part of the 0.86 may reflect
capture conditions, not behavior. **Validate on clean videos recorded in the same setup**
(ideally the same people not cheating) before trusting this in production. The three missed
cheaters scored just under the strict threshold — lower `Config.topk`/the tuned threshold to
trade precision for recall.

## Setup (macOS, Apple Silicon)

**Use Python 3.11 or 3.12** for the best wheel coverage (mediapipe + torch).

```bash
# from the project root
python3.12 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# XGBoost needs the OpenMP runtime on macOS:
brew install libomp
```

> **uv alternative:** `uv venv --python 3.12 && uv pip install -r requirements.txt`

MediaPipe downloads its FaceLandmarker model (`models/face_landmarker.task`, ~3.7 MB)
automatically on first run. It needs network access once; after that it runs offline.

Verify the signal math without any video:

```bash
python src/common.py --selftest          # sweep detector on synthetic signals
python src/build_windows.py --selftest    # + window-feature separation
```

---

## Data layout — drop your videos here

```
data/raw/cheating/<source>/   ← videos WHERE the candidate used an AI overlay tool
data/raw/clean/<source>/      ← clean sessions
```

`<source>` names the recording batch/setup (e.g. `platform_a`, `testing_team`) and
drives the cross-source evaluation — keep videos from different capture setups in
different source folders. Files placed directly under `cheating/`/`clean/` get
source `default`. Accepted: `.mp4 .webm .mkv .mov .avi .m4v`; symlinked folders work.

### v2: audio-aware turn engine (recordings WITH audio strongly preferred)

Cluely/Parakeet-style tools have a temporal signature: a flat 2–5 s lag before every
answer, and reading *while speaking* the answer. v2 detects when the candidate is
speaking and conditions every gaze feature on it:

- **Recordings with an audio track**: silero-VAD + lip-sync attribution separate
  interviewer speech from candidate speech → true question→answer latency
  (`gap_dur_cv` low = lag-loop) + reading-while-answering features. This is the
  strong path — supply audio+video recordings whenever possible.
- **Video-only recordings**: lip-motion (mouth channel) detects candidate speech;
  same features, weaker gap semantics. Works automatically as the fallback.

No flags needed — extraction probes each file and writes a `*.turns.json` sidecar
when audio is present.

---

## Run it

```bash
# 1) Extract per-frame gaze (idempotent + resumable — re-run to pick up new videos)
python src/extract_features.py
#    quick smoke test on the first 2 min of each video:
#    python src/extract_features.py --max-seconds 120

# 2) Build windowed features
python src/build_windows.py

# 3) Train, cross-validate, compare, and ship the winner
python src/train.py                       # add --export-onnx for the browser path
#    XGBoost only (skip the neural net):  python src/train.py --skip-seq

# 4a) Overall evaluation report (CV metrics, per-video table, feature importances)
python src/report.py --eval

# 4b) Score a single video + build its review report
python src/predict.py "data/raw/cheating/some_video.webm"
python src/report.py  "data/raw/cheating/some_video.webm"
```

Open the generated `reports/*.html` in a browser.

---

## Outputs

| Path | What |
|---|---|
| `data/features/*.parquet` | per-frame gaze/head-pose/blink time series (one per video) |
| `data/features/_extract_summary.csv` | per-video frame count + **% face-found** quality gate |
| `data/windows/windows.parquet` | window feature matrix + labels + video group |
| `models/model.json` | shipped winner: kind, feature set, tuned threshold, metrics |
| `models/metrics.json` | every model × feature-set combo's CV metrics |
| `models/feature_importances.csv` | interpretable feature ranking |
| `models/config.json` | all constants (also feeds a future JS port) |
| `reports/<video>__prediction.json` | verdict + probability + flagged segments |
| `reports/<video>__report.html` | per-video review (plots + shaded segments) |
| `reports/evaluation_report.html` | overall CV evaluation |

---

## Tuning knobs

Everything lives in the `Config` dataclass in `src/common.py` (and is saved to
`models/config.json`). Common ones:

- `target_fps` (sampling rate), `window_seconds` / `stride_seconds`
- `topk_fraction` — video verdict = mean of the top-k window probabilities
- CLI: `train.py --deploy-margin 0.03` — how much an appearance backend must beat
  MediaPipe-only by to be worth shipping (see deployment note below)
- `train.py --seq-device {auto,cpu,mps}` — Model B device. `auto` stays on CPU because a
  GRU's sequential steps are slower on MPS at these dataset sizes; force `mps` if you
  switch to the 1D-CNN (`--seq-kind cnn`) or have a much larger dataset.

The video-level threshold is **tuned to favor precision** (F0.5) — false cheating
accusations are costly.

---

## Gaze backends & the bake-off

The extractor always runs **MediaPipe** (cheap; iris gaze + head pose + blinks). You can
add an appearance-based gaze model as extra channels and let cross-validation decide if
it's worth it:

```bash
python src/extract_features.py --backend mediapipe+l2cs     # needs: pip install l2cs
```

`train.py` then runs the **same CV over each feature set** (MediaPipe-only vs
MediaPipe+L2CS) and keeps the winner. An appearance backend only wins if it beats
MediaPipe-only by at least `--deploy-margin`, because MediaPipe alone is the one that
runs in-browser later.

---

## Future in-browser deployment (not built yet, but supported)

The selection rules are baked in so nothing blocks a later browser-only inference build:

- MediaPipe FaceLandmarker has an official JS/WASM build (`@mediapipe/tasks-vision`) — the
  **same model** this pipeline uses.
- `train.py --export-onnx` writes `models/model.onnx` (→ `onnxruntime-web`). Model B is
  the most deployable (raw channels, no feature port); Model A is portable too but needs
  the window-feature computation ported to JS (constants are in `models/config.json`).

---

## Troubleshooting

- **`XGBoostError: libxgboost.dylib could not be loaded`** → `brew install libomp`.
- **`module 'mediapipe' has no attribute 'solutions'`** → you're on a Tasks-only wheel;
  this project already uses the Tasks API, so just make sure `mediapipe>=0.10.14` is
  installed and re-run.
- **Training hangs at 0% CPU on Model B** → XGBoost + PyTorch OpenMP conflict. `train.py`
  sets `KMP_DUPLICATE_LIB_OK=TRUE` automatically and runs the GRU folds in isolated
  processes; if it still hangs, run with `OMP_NUM_THREADS=1`. `--skip-seq` always works.
- **A video is flagged `< 70% face-found`** → poor lighting / off-frame / tiny face.
  Check it manually; its features are unreliable.
- **Bogus fps (some webm report 1000 fps)** → handled automatically; extraction samples
  by real timestamps, not frame index.
- **Need ≥ 2 videos per class** for grouped CV; with < 15 per class it uses
  leave-one-video-out automatically.

---

## Notes / caveats

- **Weak labels:** every window of a cheating video inherits the "cheating" label even
  though not every 20 s window contains reading. The top-k video-level aggregation
  absorbs this — always review flagged segments against the raw video before acting.
- **Subtle overlays:** overlays near the top-center of the screen produce small gaze
  offsets; the rhythm, head–gaze-coupling, and blink-suppression features carry the
  signal there, which is why the full feature family is kept.
- **Next lever if results are weak:** per-candidate gaze calibration (baseline-subtract
  each video's median gaze) — cheap to add.
