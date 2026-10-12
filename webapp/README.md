# webapp — server-side gaze integrity analyser

Upload an interview recording in the browser; a Python backend scores it with the
project's pipeline and the page shows the verdict, **the video itself with the
suspicious parts marked** (heat-strip timeline, red glow + reason while playback is
inside a flagged segment, click-to-jump segments), a per-window chart, the window
table and a detailed gaze-trace report.

```bash
# from the project root, once:
venv/bin/pip install -r webapp/requirements.txt

# run:
venv/bin/python webapp/server.py
# open http://127.0.0.1:8001
```

How it differs from `frontend/`:

| | `frontend/` | `webapp/` (this) |
|---|---|---|
| Where it computes | in the browser tab (WASM) | on the server (Python) |
| Speed | ~1× real time (playback-paced) | ~40× real time on an M5 (9 MediaPipe threads) |
| Matches `src/predict.py` | to within decoder noise (24/25 verdicts) | exactly — it *is* `predict.score_video` |
| Webcam live mode | yes | no (upload only) |
| Video leaves the machine | never | uploaded to the server, deleted after scoring |

---

## What runs where

```
browser (static/)                 server (server.py + engine.py)
─────────────────                 ───────────────────────────────────────────────
upload ── POST /api/jobs ───────▶ save to runs/uploads/<id>.<ext>, queue the job
                                  worker thread:
                                    predict.score_video(...)          ← src/, unchanged maths
                                      extract_one   MediaPipe on 9 CPU threads (warm pool, progress callback)
                                      clean + 20 s windows
                                      CNN           torch on MPS/CUDA/CPU
                                    predict.save_timeline + report.build_video_report
                                    delete the upload + per-frame cache
poll ──── GET /api/jobs/{id} ───▶ stage, seconds processed, result when done
render verdict / chart / table
"Detailed report" ─────────────▶ GET /api/jobs/{id}/report  (report.py's HTML)
```

### Hardware

- **Face tracking (MediaPipe) → CPU, on 90% of the cores.** The decoder thread reads
  frames in order (same keep rule as always) and hands each one to a pool of
  `--threads` worker threads (default: 90% of logical cores, 9 on a 10-core M5), each
  with its own FaceLandmarker. MediaPipe runs in IMAGE mode — every frame is scored on
  its own — so the extracted data is **bit-identical** to a single-threaded run; only
  the speed changes. The pool is created and warmed once at server start (making a
  FaceLandmarker costs ~0.7 s), so every job starts at full speed.
  The Python MediaPipe build has no GPU delegate on macOS, and the 0.959 threshold was
  tuned on CPU landmarks, so the CPU is also the most faithful choice.

  Measured on a 3:53 recording (Apple M5, 10 cores):

  | | face-tracking phase | CPU in use | whole job |
  |---|---|---|---|
  | 1 thread (CLI default) | 14.5 s | ~1.1 cores (11%) | 17.9 s |
  | 9 threads, warm pool (this server) | 3.2 s | ~8.6 cores (86%) | 5.2 s |

  The rest of a job — windowing, the CNN and the report — takes ~1 s per 4 minutes of
  video. Audio is never read (the CNN only uses visual channels).
- **CNN → best torch device**: CUDA if present, else Apple MPS, else CPU
  (`--cnn-device` overrides). On the M5, MPS scores 100 windows in ~1.8 ms against ~64 ms
  on CPU, and the two agree to ~1e-8, so the GPU costs no accuracy. The CNN is tiny
  either way — its device choice doesn't change total time noticeably.
- **Several uploads at once:** `--workers N` analyses N videos in parallel; they split
  the `--threads` budget between them, so total CPU stays at ~90%.

### Video review

When a job finishes, the page plays the recording with the evidence on top of it:

- **Heat strip** under the video — every window's score on the video's own clock
  (red = flagged, brighter blue = higher score, hatched = not analysed, e.g. after
  "analyse first N minutes"). Click it, or focus it and use ←/→ (5 s) and PgUp/PgDn (30 s).
- **Red glow + "Suspicious · <reason>" badge** whenever playback is inside a flagged segment.
- **Now** readout: current time, the score of the window covering it, and its signals.
- **Flagged segments are jump buttons**; chart bars and window-table rows also jump.

The video plays **from your own file** (a browser object URL) — it is never uploaded a
second time, and the server still deletes its copy after scoring. When you reopen a job
later via its `#job=` link, the page asks you to pick the same file again (it stays on
your disk), or plays the server's copy if the server was started with `--keep-uploads`.
Browsers can't play every container: mp4 and webm work; `.avi` / some `.mkv` may not, in
which case the timeline and segments still work and show the times to check.

### Options

```
--host 127.0.0.1      bind address; 0.0.0.0 exposes it on your network (there is NO auth)
--port 8001
--threads N           MediaPipe threads in total (default: 90% of logical cores)
--workers 1           videos analysed in parallel (share the --threads budget)
--cnn-device auto     auto | cpu | mps | cuda
--keep-uploads        keep uploaded videos in runs/uploads/ (enables playback on reopen)
```

API docs (auto-generated): http://127.0.0.1:8001/api/docs

---

## Files

```
webapp/
  server.py              FastAPI routes + static hosting + CLI flags
  engine.py              job queue, device selection, wraps src/predict.py
  requirements.txt       fastapi, uvicorn, python-multipart
  static/                the page (vanilla ES5, no build step)
    index.html
    src/styles/main.css  same design system as frontend/css/style.css
    src/utils/           format.js, http.js (XMLHttpRequest wrappers)
    src/components/      intake.js, progress.js, review.js (video player), chart.js, result.js
    src/main.js          entry point: upload → poll → render
  runs/                  job outputs (git-ignored)
    <id>__web.json         result payload the page renders
    <id>__prediction.json  same format as reports/*__prediction.json
    <id>__timeline.npz     per-sample gaze/head/blink + window probabilities
    <id>__report.html      detailed report (report.py)
```

Results survive a server restart: a finished job's page (`/#job=<id>`) reloads from
`runs/<id>__web.json`. Queued or running jobs do not survive a restart.

## Notes

- **Same verdicts as the CLI.** The engine calls `predict.score_video` with the CNN from
  `models/`. Retrain with `src/train.py` and restart the server to pick up a new model.
- **Privacy.** The upload and its per-frame cache are deleted as soon as the job ends
  (finished, failed or cancelled). The result, timeline and report stay in `runs/`.
- **Minimum length.** Like every other entry point, a recording needs ≥ 20 s of usable
  signal; shorter ones end with a "too short" error.
- **"Analyse first N minutes"** is passed to the extractor as `max_seconds`, so the
  score covers only that span.
