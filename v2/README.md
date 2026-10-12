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
| extract | every frame up to 25 fps (grid-anchored), frames downscaled to ≤ 640 px, MediaPipe **blendshapes** (eyeLook*, eyeBlink*, jawOpen, smile, squint, brow) + face scale | `src/common.py`, `src/extract_features.py` |
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

### Agreement with the manual reviews (per second)

| candidate | | AUC | precision | recall | IoU | reviewer spans caught |
|---|---|---|---|---|---|---|
| Marthala (4.0 min reading) | v1 | 0.57 | 0.17 | 0.69 | 0.16 | 6 / 9 |
| | **v2** | **0.93** | **0.57** | **0.91** | **0.54** | **9 / 9** |
| Sai Koti Reddy (6.2 min) | v1 | 0.48 | 0.15 | 0.25 | 0.10 | 8 / 29 |
| | **v2** | **0.88** | **0.44** | **0.98** | **0.44** | **28 / 29** |
| Muthakani (11.6 min) | v1 | 0.71 | 0.67 | 0.54 | 0.42 | 14 / 26 |
| | **v2** | **0.96** | **0.98** | **0.64** | **0.63** | **18 / 26** |
| **all three** | v1 | 0.59 | — | — | 0.23 | 28 / 64 |
| | **v2** | **0.93** | **0.62** | **0.78** | **0.53** | **55 / 64** |

On the 79 clean videos v2 flags 2.0 % of seconds (short isolated segments; the verdict
rule keeps all but one clean video "clean").

### Video-level verdicts (100 recordings: 21 cheating / 79 clean)

| | AUC | caught | false alarms | same-platform AUC |
|---|---|---|---|---|
| v1 (its own out-of-fold CV) | 0.84 | 9 / 21 | 1 / 79 | 0.86 |
| **v2 shipped** | **0.85** | **9 / 21** | **1 / 79** | 0.83 |

The verdict is on par with v1, but v2 reaches it without the resolution confound
and every verdict now comes with the reading segments that justify it.
Verdict rule: *average reading probability while the candidate speaks ≥ 0.21*.

### How we got there (each row is a `RESULTS_LOG.md` entry)

| run | spans caught | pooled recall | video AUC | note |
|---|---|---|---|---|
| v1-baseline | 28 / 64 | — | 0.84 | in-sample per-second |
| v2-xgb-base | 3 / 64 | 0.10 | 0.86 | ranks well (AUC 0.79–0.95) but an unseen reader scores far below seen ones |
| v2-xgb-context | 16 / 64 | 0.20 | 0.82 | + answer-context features |
| v2-xgb-context-selftrain | 57 / 64 | 0.81 | 0.79 | + self-training on unreviewed cheaters: fixes calibration |
| **v2-xgb-selftrain-hardneg** | **55 / 64** | **0.78** | **0.85** | + clean-speech hard negatives, speaking-based verdict → **shipped** |
| v2-cnn-strong-labels | 5 / 64 | 0.04 | 0.72 | 1D-CNN on raw channels overfits 3 readers |

### What the data says reading looks like

Learned against clean candidates' *speech*: reading while answering comes with
**reduced blinking, a still head, a lowered brow (concentration), steady eyes with
line-by-line drift, and fewer side-to-side glances** — genuine answers have more
"thinking" glances away and head movement. These are the reason tags on each segment.

### Worth a manual look

- **Cheating videos v2 does not flag (12 of 21).** Six were below threshold in v1
  too: Jagruti Patil, Ram Shekhawat, Sampath Pujari, Thrinath Reddy Dadam, Ishika
  Vijaywargiya, Kalluri Praneeth. If they cheated another way (phone below the camera,
  audio prompts), a reading detector can't see it; if they did read, reviewing 2–3 of
  them is the fastest way to improve the model. v1 and v2 each catch 9, but not the same
  9: v2 adds Muthakani and Abhilasha (whose score sits exactly on the tuned threshold,
  so treat her as borderline), and loses Shyam Sundar Vallapatla and Bhaskar R.
- **Clean videos that look most like reading**: Komanduri Suhas (the one false
  alarm), Ranganathan K, Navaneeth Remesh, Salman Mohammed Salar.
- **Muthakani's 8 missed spans**: the longest (0:37–1:27, 50 s) is where the reviewer
  notes he "keeps looking up to avoid suspicion" — a deliberate counter-strategy; the
  rest are short (8–22 s), and two of them were partly flagged (3:08–3:57 at 41 %,
  21:28–21:49 at 38 %).

Full history of every run: [`RESULTS_LOG.md`](RESULTS_LOG.md) (append-only; JSON copies in `results/`).

## Run it

From the project root (the folder that contains `v2/`):

```bash
# 1) per-frame features (25 fps + blendshapes) -> v2/data/features   (~1.5 h for 100 videos)
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

Other entry points:

```bash
venv/bin/python v2/src/labels.py                       # list the reviews it finds
venv/bin/python v2/src/evaluate.py --v1-baseline       # re-log the original model's numbers
venv/bin/python v2/src/train.py --drop-group ctx --exp no-context --no-ship   # ablation
venv/bin/python v2/src/train_seq.py --exp cnn          # CNN bake-off (logs, doesn't ship)
venv/bin/python v2/src/report.py --eval                # v2/reports/evaluation_report.html
```

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
  src/                 pipeline (see table above)
  models/              config.json, model.json (thresholds, features, CV metrics, reason
                       references), reading_xgb.json, face_landmarker.task
  data/features/       per-frame parquets (25 fps + blendshapes)
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
- The browser-only build (`../frontend/`) still runs the v1 CNN; v2 is server-side.
