# v2 — per-second reading detection, trained on the manual reviews

v2 lives entirely in this folder. The original pipeline (`../src`, `../models`,
`../webapp`, `../data`) is untouched; v2 only **reads** the original videos
(`../data/raw/…`) and the manual reviews (`../manual_review_for_candidate/`).

## Why v2

The manual reviews mark *when* a candidate reads from the screen while answering.
Measured against them, v1 did not localise reading, even on videos it was trained on:

| candidate | reviewed reading | v1 per-second AUC | v1 precision | v1 recall |
|---|---|---|---|---|
| Muthakani (ir8308053) | 11.6 min | 0.71 | 0.67 | 0.54 |
| Marthala (ir8244089) | 4.0 min | 0.57 | 0.17 | 0.69 |
| Sai Koti Reddy (ir8266049) | 6.2 min | 0.48 | 0.15 | 0.25 |

Root causes found:

1. **Weak labels.** v1 labelled every 20 s window of a cheating video "cheating", so
   the CNN learned *who* cheats (person / camera), not *when* they read. It flagged
   9:00–20:20 of Marthala, where the reviewer marked no reading.
2. **Coarse windows.** 20 s windows every 10 s can't resolve 6–30 s reading spans.
3. **Sweep detector pointed the wrong way.** It looked for a fast *leftward* gaze
   return; in these (un-mirrored) recordings the return is rightward in `gaze_x`, so
   the "reading sweeps" reason almost never fired.
4. **Frames dropped.** A keep-rule quirk sampled 25 fps sources at 12.5 fps.
5. **Resolution confound.** 42 of 79 clean videos are 1280×720; every cheating video
   is ≤ 640×360, so landmark noise alone separates the classes.

## What changed

| stage | v2 change | file |
|---|---|---|
| labels | parses the reviewers' CSVs (handles `24:25:00` sheet artefacts), matches by IR number | `src/labels.py` (new) |
| extract | a fixed **15 fps** everywhere — training, CLI, web app — matching production (token-bucket frame sampler: 25/30/60 fps sources give exactly 15, slower or uneven ~15 fps WebMs keep every frame; v1 got 12.5), frames downscaled to ≤ 640 px, MediaPipe **blendshapes** (eyeLook*, eyeBlink*, jawOpen, smile, squint, brow) + face scale | `src/common.py`, `src/extract_features.py` |
| speaking | lip-gap + jawOpen fused visual speaking detector | `src/speaking.py` |
| windows | **8 s windows every 1 s** → one probability per second; **strong labels** from the reviews; ~85 eye + mouth + head features; polarity-free sweep detector; answer-context features | `src/build_windows.py` |
| model | XGBoost on the features, weighted so reviewed reading, reviewed not-reading and clean speech each count | `src/train.py` |
| scoring | per-second score → reading segments (gaps < 3 s merged, < 4 s dropped) → verdict on **total reading time** | `src/reading.py` (new), `src/predict.py` |
| reasons | each segment says why: e.g. "reading while answering · reduced blinking, gaze held on one screen region" | `src/train.py` → `models/model.json` |
| evaluation | per-second agreement with reviews + video-level verdicts; every run appended to `RESULTS_LOG.md` | `src/evaluate.py`, `src/results_log.py` (new) |
| bake-off | 1D-CNN on raw channels, same folds and metrics | `src/train_seq.py` (new) |
| web app | per-second timeline with reading / speaking / reviewer lanes, auto-compare to a manual review, review-CSV upload, segments CSV export | `webapp/` |

## Results

All v2 numbers are **out-of-fold**: each reviewed video is scored by a model that
never saw it (or its labels, including via self-training), and the 18 unreviewed
cheating videos are never scored by a model that trained on them. v1's per-second
numbers are **in-sample** (it trained on these videos), i.e. its best case.

Since 2026-10-13 v2 processes video at **15 fps everywhere** (the production rate). The
25 fps model it replaced is kept in `archive/25fps/`; its numbers stay below for comparison.

### Agreement with the manual reviews (per second)

| candidate | | AUC | precision | recall | IoU | reviewer spans caught |
|---|---|---|---|---|---|---|
| Marthala (4.0 min reading) | v1 | 0.57 | 0.17 | 0.69 | 0.16 | 6 / 9 |
| | v2 @ 25 fps | 0.93 | 0.57 | 0.91 | 0.54 | 9 / 9 |
| | **v2 @ 15 fps** | **0.93** | **0.57** | **0.81** | **0.50** | **9 / 9** |
| Sai Koti Reddy (6.2 min) | v1 | 0.48 | 0.15 | 0.25 | 0.10 | 8 / 29 |
| | v2 @ 25 fps | 0.88 | 0.44 | 0.98 | 0.44 | 28 / 29 |
| | **v2 @ 15 fps** | **0.88** | **0.45** | **0.96** | **0.44** | **28 / 29** |
| Muthakani (11.6 min) | v1 | 0.71 | 0.67 | 0.54 | 0.42 | 14 / 26 |
| | v2 @ 25 fps | 0.96 | 0.98 | 0.64 | 0.63 | 18 / 26 |
| | **v2 @ 15 fps** | **0.95** | **0.90** | **0.78** | **0.72** | **23 / 26** |
| **all three** | v1 | 0.59 | — | — | 0.23 | 28 / 64 |
| | v2 @ 25 fps | 0.93 | 0.62 | 0.78 | 0.53 | 55 / 64 |
| | **v2 @ 15 fps** | **0.92** | **0.63** | **0.84** | **0.55** | **60 / 64** |

On the 79 clean videos v2 flags 1.8 % of seconds (2.0 % at 25 fps): short isolated
segments; the verdict rule keeps all but one clean video "clean".

### Video-level verdicts (100 recordings: 21 cheating / 79 clean)

| | AUC | caught | false alarms | overall | same-platform AUC |
|---|---|---|---|---|---|
| v1 (its own out-of-fold CV) | 0.84 | 10 / 21 | 1 / 79 | 88 / 100 | 0.86 |
| v2 @ 25 fps | 0.85 | 9 / 21 | 1 / 79 | 87 / 100 | 0.83 |
| **v2 @ 15 fps (shipped)** | **0.83** | **10 / 21** | **1 / 79** | **88 / 100** | **0.83** |

At 15 fps v2 catches one more cheater (Chippgiri Reddy) and ties v1, but its ranking
AUC slipped from 0.85 to 0.83: cheating scores moved closer to the clean ones, the tuned
threshold dropped from 0.21 to 0.17, and Sakshi Mishra sits exactly on it. Every verdict
comes with the reading segments that justify it, without v1's resolution confound.
Verdict rule: *average reading probability while the candidate speaks ≥ 0.17*.
Per-video table: [`results/ALL_VIDEOS_RESULTS.md`](results/ALL_VIDEOS_RESULTS.md).
Cheating cases for Google Sheets / Excel: `results/CHEATING_CASES_REPORT.xlsx` — it opens on
"How to read this", a plain-language guide for someone new to the project (how a verdict is
reached, every tab and column with a real example value, the 7 behaviour measurements and
their normal values, colours, glossary); the next tab lists
the 21 cheating candidates; the one after lists every reading moment caught (time, strongest second,
confidence, measured blinks / head / eye movement next to normal speaking, reviewer check);
then one tab per IR id with those moments, the reviewer's spans, the 5 strongest moments,
a minute-by-minute view and the second-by-second probability. The same layout for the 79
clean candidates is `results/NOT_CHEATING_CASES_REPORT.xlsx` (CLEARED / FLAGGED, with every
moment the model wrongly flagged); `results/NOT_CHEATING_CASES_REPORT_v2_only.xlsx` is the
same without the v1 / 25 fps comparison. All come from `src/cheating_cases_xlsx.py`.

*Correction (2026-10-13):* v1 was listed earlier as 9 / 21. Its score for Abhilasha M E
equals its threshold exactly; reading the scores back from CSV moved the last digit and
showed her as missed. v1's own rule (≥) catches her, so v1 is 10 / 21.

### How we got there (each row is a `RESULTS_LOG.md` entry)

| run | 25 fps: spans · recall · video AUC | 15 fps: spans · recall · video AUC | note |
|---|---|---|---|
| v1-baseline | 28 / 64 · — · 0.84 | (v1 unchanged) | in-sample per-second |
| xgb-base | 3 / 64 · 0.10 · 0.86 | 0 / 64 · 0.06 · 0.84 | ranks well but an unseen reader scores far below seen ones |
| xgb-context | 16 / 64 · 0.20 · 0.82 | 15 / 64 · 0.17 · 0.83 | + answer-context features |
| xgb-context-selftrain | 57 / 64 · 0.81 · 0.79 | 62 / 64 · 0.91 · 0.77 | + self-training on unreviewed cheaters: fixes calibration |
| **xgb-selftrain-hardneg** | 55 / 64 · 0.78 · 0.85 | **60 / 64 · 0.84 · 0.83** | + clean-speech hard negatives, speaking-based verdict → **shipped** |
| cnn-strong-labels | 5 / 64 · 0.04 · 0.72 | 6 / 64 · 0.07 · 0.74 | 1D-CNN on raw channels overfits 3 readers |
| cnn-vs-xgb-selftrain (CNN) | 3 / 64 · 0.04 · 0.81 | 4 / 64 · 0.08 · 0.79 | CNN with the shipped recipe, same windows |
| cnn-vs-xgb-selftrain (XGB+CNN average) | 39 / 64 · 0.54 · 0.89 | 40 / 64 · 0.57 · 0.86 | averaging helps the verdict, hurts the timeline |

### What the data says reading looks like

Learned against clean candidates' *speech*: reading while answering comes with
**a still head, reduced blinking, a lowered brow (concentration), eyes raised compared
with when listening, steady eyes with line-by-line drift, and few big eye jumps** —
genuine answers have more "thinking" glances away and head movement. These are the
reason tags on each segment.

### Worth a manual look

- **Cheating videos v2 does not flag (11 of 21).** Nine are below threshold in v1 too:
  Dhiraj Sanjay Kharche, Shweta Chandrakant Gambhire, Gaurav Kumar, Kalluri Praneeth,
  Sampath Pujari, Ishika Vijaywargiya, Thrinath Reddy Dadam, Jagruti Patil, Ram
  Shekhawat. If they cheated another way (phone below the camera, audio prompts), a
  reading detector can't see it; if they did read, reviewing 2–3 of them is the fastest
  way to improve the model. v1 and v2 each catch 10, but not the same 10: v2 adds
  Muthakani and Chippgiri Reddy, v1 adds Shyam Sundar Vallapatla and Bhaskar R. Sakshi
  Mishra sits exactly on the v2 threshold — treat her as borderline.
- **Clean videos that look most like reading**: Komanduri Suhas (the one false
  alarm), Pathan Arshiya Anjum, Salman Mohammed Salar, Ranganathan K.
- **Muthakani's 3 missed spans**: the longest (0:37–1:27, 50 s, 14 % flagged) is where
  the reviewer notes he "keeps looking up to avoid suspicion" — a deliberate
  counter-strategy; 2:02–2:17 and 4:34–4:49 were each 33 % flagged.

Full history of every run: [`RESULTS_LOG.md`](RESULTS_LOG.md) (append-only; JSON copies in `results/`).

## Run it

From the project root (the folder that contains `v2/`):

```bash
# 1) per-frame features (15 fps + blendshapes) -> v2/data/features
venv/bin/python v2/src/extract_features.py --workers 7

# 2) reading windows + strong labels -> v2/data/windows
venv/bin/python v2/src/build_windows.py

# 3) cross-validate, log to RESULTS_LOG.md, ship to v2/models
venv/bin/python v2/src/train.py --exp my-run-name

# 4) one video (CLI) + its HTML report
venv/bin/python v2/src/predict.py "cheating_folder/<video>.mp4" --workers 8
venv/bin/python v2/src/report.py  "cheating_folder/<video>.mp4"

# 5) web app (v1 keeps port 8001; v2 runs on 8002, both can run together)
venv/bin/python v2/webapp/server.py        # http://127.0.0.1:8002
```

After step 1, `bash v2/rerun_all.sh` (add `--with-cnn` for the CNN bake-off) re-runs
everything else in order: windows, the experiment ladder, the shipped model, the v1
baseline, `results/ALL_VIDEOS_RESULTS.md`, both config CSVs, the evaluation report,
the web-app library and the two `results/*CASES_REPORT.xlsx` workbooks. Restart the server afterwards.

Other entry points:

```bash
venv/bin/python v2/src/labels.py                       # list the reviews it finds
venv/bin/python v2/src/evaluate.py --v1-baseline       # re-log the original model's numbers
venv/bin/python v2/src/train.py --drop-group ctx --exp no-context --no-ship   # ablation
venv/bin/python v2/src/train_seq.py --exp cnn          # CNN bake-off (logs, doesn't ship)
venv/bin/python v2/src/report.py --eval                # v2/reports/evaluation_report.html
```

## Experiment tracking (MLflow)

Every run that lands in `RESULTS_LOG.md` (`train.py`, `train_seq.py`, `evaluate.py`) is
also logged to a local MLflow store, so runs can be sorted, charted and diffed:

```bash
venv/bin/python v2/src/tracking.py ui          # prints the runs link, http://127.0.0.1:5050/#/experiments/1/runs
venv/bin/python v2/src/tracking.py backfill    # import results/*.json not yet tracked
venv/bin/python v2/src/tracking.py repair      # fill columns / metrics older runs lack (never overwrites)
```

- **params**: every option of both training scripts (`n/a (… run)` for the other script's,
  `not an option yet` for ones added later), plus `model`, `contender` and `target_fps`, so
  v2 runs line up column for column. **notes**: the run summary.
- **metrics**: `per_second/*` pooled over the reviewed videos (same pooling as the log's
  "ALL reviewed" row), `per_video/<ir>/*`, `video/*` verdicts, `stat_auc/*`.
- **artifacts**: the result JSON, its `RESULTS_LOG.md` section, out-of-fold scores, and
  the shipped model when `train.py` ships (tagged `shipped`).
- Head-to-heads (`train_seq.py --compare-xgb`) get one child run per contender; the run
  itself carries the CNN's metrics (tag `metrics_from`). Charts: the runs page's chart view.

The store (`v2/mlflow.db`, `v2/mlartifacts/`) stays on this machine and is git-ignored.
Tracking is best-effort: without MLflow, or with `V2_TRACKING=0`, runs still go to
`RESULTS_LOG.md`. Set `MLFLOW_TRACKING_URI` to log to a shared MLflow server instead.

## Adding more manual reviews (the biggest lever)

Only 3 candidates are reviewed, so the model has seen reading from 3 people.
Each new review adds a new reader:

1. Save the sheet as CSV in `manual_review_for_candidate/` with the IR number in the
   file name (e.g. `…ir8277721….csv`), same columns as the existing ones.
2. Re-run steps 2 and 3. `build_windows.py` picks the review up automatically; the
   video moves from "unreviewed (test only)" to "reviewed (trained on, held out per fold)".
3. Compare the new `RESULTS_LOG.md` entry with the previous one.

The web app also matches reviews by IR number, so a reviewed candidate shows the
agreement panel as soon as it is uploaded.

## Folder layout

```
v2/
  README.md            this file
  RESULTS_LOG.md       append-only history of every evaluation run
  results/             JSON per run + out-of-fold per-second / per-video scores
  mlflow.db            MLflow runs (git-ignored; `src/tracking.py ui` to browse)
  mlartifacts/         MLflow run files (git-ignored)
  src/                 pipeline (see table above)
  models/              config.json, model.json (thresholds, features, CV metrics, reason
                       references), reading_xgb.json, face_landmarker.task
  data/features/       per-frame parquets (15 fps + blendshapes)
  archive/25fps/       the model shipped before the switch to 15 fps (25 fps features
                       and windows: data/features_25fps/, data/windows_25fps/, local only)
  data/windows/        windows.parquet, videos.parquet
  reports/             CLI predictions + HTML reports, extraction log
  webapp/              v2 server + page (runs/ holds job outputs)
```

## Caveats

- **Three reviewed readers.** Per-second numbers come from leave-one-reviewed-video-out
  folds, so each is "a reader the model never saw", but three people is a small sample.
- **Unreviewed cheating videos are a true hold-out**: they are never trained on, so
  their video-level verdicts are honest. Unlabelled stretches of reviewed videos are
  treated as "not reading".
- **Thresholds are tuned on out-of-fold scores** (per-second: best F1 with ≤ 2 % of
  clean seconds flagged; video: F0.5, favouring precision like v1).
- **15 fps everywhere.** Extraction, training, the CLI and the web app all sample 15 fps
  (`models/config.json`); `model.json` records the rate it was trained on and `predict.py`
  refuses a mismatch. Recordings slower than 15 fps keep every frame.
- The browser-only build (`../frontend/`) still runs the v1 CNN; v2 is server-side. The
  v2 model itself exports to ONNX and runs in Chrome via onnxruntime-web (checked on the
  15 fps model: max difference 3e-7 vs Python); the feature pipeline is not ported to JS yet.
