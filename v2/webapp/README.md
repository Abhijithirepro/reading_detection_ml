# v2 webapp — server-side reading detector

Upload an interview recording; the Python backend runs the v2 pipeline
(`v2/src/predict.py`) and the page shows **when** the candidate reads from the
screen while answering, on the video itself.

```bash
# from the project root (the folder containing v2/):
venv/bin/pip install -r v2/webapp/requirements.txt     # once (fastapi, uvicorn, multipart)
venv/bin/python v2/webapp/server.py                    # http://127.0.0.1:8002
```

v1's server (`webapp/`, port 8001) is unchanged; both can run side by side.

## What the page shows

- **Verdict = total reading time** ("Reading 7:32 · 18 segments · rule ≥ m:ss").
  The rule (minimum reading time for CHEATING) and the per-second threshold are
  tuned out-of-fold in `v2/src/train.py` and read from `v2/models/model.json`.
- **Timeline under the video, three lanes, per second:**
  reading probability (brighter = more likely, red = inside a detected reading
  segment) · speaking (lip/jaw motion) · reviewer spans (amber, when a review exists).
- **Reading segments** as jump buttons, each with its signals, e.g.
  "reading while answering · reduced blinking, gaze held on one screen region".
- **Manual review · agreement.** If the filename's IR number (`ir8266049`) matches
  a CSV in `manual_review_for_candidate/`, the comparison is attached automatically:
  spans caught, per-second AUC, precision, recall, and every reviewer span marked
  ✓ caught / ✗ missed (click to play it). Any other review CSV in the same sheet
  format can be loaded with **Load review CSV**.
- **Segments CSV** downloads the detected segments in the reviewers' sheet format
  (start, end, "reading", reason) — open it next to the video and confirm/reject.
- **Detailed report**: eye left↔right and up↕down traces, mouth with the speaking
  lane, per-second probability, red detections vs amber reviewer spans.

## API

```
GET    /api/info                     model, thresholds, hardware, reviews on disk
POST   /api/jobs                     multipart: file, max_minutes (optional) -> {id}
GET    /api/jobs/{id}                status / progress / result
DELETE /api/jobs/{id}                cancel
GET    /api/jobs/{id}/report         detailed HTML report
GET    /api/jobs/{id}/result.json    result payload
GET    /api/jobs/{id}/segments.csv   detected reading segments (reviewers' sheet format)
POST   /api/jobs/{id}/review         multipart review CSV -> agreement with the detections
GET    /api/jobs/{id}/video          the upload (only with --keep-uploads)
```

## Options

```
--host 127.0.0.1    bind address; 0.0.0.0 exposes it on your network (there is NO auth)
--port 8002
--threads N         MediaPipe threads in total (default: 90% of logical cores)
--workers 1         videos analysed in parallel (share the --threads budget)
--keep-uploads      keep uploaded videos in runs/uploads/ (enables playback on reopen)
```

## Notes

- **Same results as the CLI**: the engine calls `predict.score_video`.
  Retrain with `v2/src/train.py` and restart the server to pick up a new model.
- **Speed**: v2 tracks every frame up to 25 fps and computes blendshapes, about
  2× the face-tracking work of v1; the reading model itself takes milliseconds.
- **Privacy**: the upload and its per-frame cache are deleted when the job ends;
  the result, timeline and report stay in `runs/`.
- **Times are on the video clock** (v1 used the analysis clock plus an offset).
