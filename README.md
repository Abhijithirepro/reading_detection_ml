# Gaze-Based Cheating Detection

Detects candidates reading from an AI overlay tool (Cluely, Parakeet AI, etc.) during
a recorded interview, from the **gaze signature** alone — repeated left→right reading
sweeps with fast return saccades, a steady line-by-line rhythm, gaze parked off-center
on an overlay region, and suppressed blinking.

Ingests a webcam video and outputs a **cheating verdict + probability + a timeline of
suspicious segments**, plus an HTML report with gaze plots for human review.

Built for an **Apple Silicon Mac (M-series)**. Pure Python; MediaPipe runs the face/iris
tracking, PyTorch trains the 1D-CNN. The same CNN also runs fully in-browser
(`frontend/`).

---

## How it works (4 stages)

```
video ──▶ [1] extract ──▶ per-frame gaze/head-pose time series  (data/features/*.parquet)
              │
              ▼
          [2] windows ──▶ 20s windows: raw gaze sequences + behavioral features  (data/windows/)
              │
              ▼
          [3] train  ──▶ 1D-CNN on the raw sequences, grouped CV, ship it  (models/)
              │
              ▼
          [4] predict / report ──▶ verdict + flagged segments + HTML  (reports/)
```

- **The model** — a compact 1D-CNN (two `Conv1d` layers → average pool → linear head,
  ~6k parameters) over each window's six raw channels: `gaze_x, gaze_y, yaw, pitch,
  ear, mouth`, resampled to 240 steps. It outputs one cheating probability per window.
- The engineered window features (reading-sweep rate, off-center gaze, saccade rate,
  blink rate, …) are **not** model inputs; they only produce the human-readable reason
  tags on flagged segments ("reading sweeps", "suppressed blinking", …).
- It is evaluated with **leakage-safe cross-validation grouped by video** (windows from
  one video never span train/test), then retrained on all data and shipped.

---

## Results (106 videos: 27 cheating / 79 clean)

5-fold cross-validation grouped by video, precision-tuned operating point
(threshold 0.959):

| Video ROC-AUC | PR-AUC | Precision | Recall | False alarms | Missed cheaters |
|---|---|---|---|---|---|
| **0.871** | **0.808** | **0.94** | 0.56 | 1 / 79 | 12 / 27 |

**Per-candidate gaze calibration** (`Config.calibrate_gaze`, on by default) subtracts each
person's median gaze, so reading/overlay behavior is measured as a *deviation from their
own baseline*, not an absolute position.

⚠️ **Honest caveat:** the cheating and clean videos partly come from *different recording
sources*, so some of the score may reflect capture conditions, not behavior — check the
cross-source numbers in `reports/evaluation_report.html`. **Validate on clean videos
recorded in the same setup** before trusting this in production. The threshold favors
precision; lower it in `models/model.json` to trade false alarms for recall.

## Setup (macOS, Apple Silicon)

**Use Python 3.11 or 3.12** for the best wheel coverage (mediapipe + torch).

```bash
# from the project root
python3.12 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
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

### Video only — no audio

The pipeline never reads the audio track. The CNN scores only the six visual
channels (gaze, head pose, eye openness, mouth openness), and the reason tags on
flagged segments use speaking-independent gaze features. A lip-motion speaking
detector (`src/speaking.py`, on the mouth channel) still feeds a few engineered
window features, but nothing that affects the verdict uses them.

---

## Run it

```bash
# 1) Extract per-frame gaze (idempotent + resumable — re-run to pick up new videos)
python src/extract_features.py
#    quick smoke test on the first 2 min of each video:
#    python src/extract_features.py --max-seconds 120

# 2) Build windowed features
python src/build_windows.py

# 3) Train, cross-validate, and ship the CNN
python src/train.py                       # add --export-onnx for the browser path

# 4a) Overall evaluation report (CV metrics, cross-source check, per-video table)
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
| `data/windows/sequences.npz` | per-window raw sequences — the CNN's training input |
| `models/model_cnn.pt` | trained CNN weights + normalization mean/std |
| `models/model.json` | shipped CNN: tuned threshold, sequence channels/length, metrics |
| `models/seq_scaler.npz` | per-channel mean/std applied before the CNN |
| `models/model.onnx` | ONNX export of the CNN for the browser (`--export-onnx`) |
| `models/metrics.json` | CNN CV metrics + cross-source + per-source scores |
| `models/cv_video_scores.csv` | per-video out-of-fold scores |
| `models/config.json` | all constants (shared with the JS port in `frontend/`) |
| `reports/<video>__prediction.json` | verdict + probability + flagged segments |
| `reports/<video>__report.html` | per-video review (plots + shaded segments) |
| `reports/evaluation_report.html` | overall CV evaluation |

---

## Tuning knobs

Everything lives in the `Config` dataclass in `src/common.py` (and is saved to
`models/config.json`). Common ones:

- `target_fps` (sampling rate), `window_seconds` / `stride_seconds`
- `topk_fraction` — video verdict = mean of the top-k window probabilities

The video-level threshold is **tuned to favor precision** (F0.5) — false cheating
accusations are costly.

---

## Gaze backend

The extractor runs **MediaPipe** (cheap; iris gaze + head pose + blinks) — the same
FaceLandmarker model the browser build uses. The CNN reads only MediaPipe's channels.

---

## In-browser deployment (`frontend/`)

`frontend/` runs the whole pipeline client-side: MediaPipe via `@mediapipe/tasks-vision`
and the CNN via `onnxruntime-web`. See `frontend/README.md`.

After retraining, refresh its assets by hand — nothing syncs them automatically:

1. `python src/train.py --export-onnx` (needs `pip install onnx onnxscript`)
2. copy `models/model.onnx` and `models/model.json` to `frontend/assets/`
3. convert `models/seq_scaler.npz` to `frontend/assets/seq_scaler.json` (`{"mean": [...], "std": [...]}`)

---

## Troubleshooting

- **`module 'mediapipe' has no attribute 'solutions'`** → you're on a Tasks-only wheel;
  this project already uses the Tasks API, so just make sure `mediapipe>=0.10.14` is
  installed and re-run.
- **Training hangs at 0% CPU** → OpenMP runtime conflict. `train.py` sets
  `KMP_DUPLICATE_LIB_OK=TRUE` automatically and runs the CNN folds in isolated
  processes; if it still hangs, run with `OMP_NUM_THREADS=1`.
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
