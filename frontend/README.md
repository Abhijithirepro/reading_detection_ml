# Frontend — browser-only gaze integrity analyser

A standalone page that scores an interview recording for AI-overlay reading
**entirely in the browser**, from either an uploaded file or a live webcam
session. Nothing ever leaves the machine: frame sampling, face/gaze tracking,
feature computation and model inference all run client-side, and webcam
sessions are never recorded — frames are analysed in memory and discarded.

A verdict appears as soon as the first 20-second window of usable signal has
been processed, then keeps updating roughly every 10 seconds after that (the
model's own window/stride length) — there's no waiting for an entire upload or
session to finish before seeing anything.

```bash
cd frontend
python3 -m http.server 8000
# open http://localhost:8000
```

A static server is required. Opening `index.html` as a `file://` URL fails —
browsers block the `fetch()` calls for the model and the WASM runtime.

---

## What runs here

```
frame-sampler.js (file)  ──┐   decode in order, keep on Python's schedule
live-source.js (webcam)  ──┤   real-time capture loop, same canvas → same callback
                            ▼
                     session.js           the shared incremental driver — see below
                            │
                            ▼
          mediapipe-backend.js  FaceLandmarker (same .task bundle as Python)
              │                 → gaze_x/y, yaw, pitch, roll, ear, mouth, blink
              ▼
          pipeline.js           clean_video: interpolate → median filter →
              │                 resample → per-candidate gaze calibration
              │                 then slide 20s windows at 10s stride
              ▼
          model-runner.js       (x-mean)/std → CNN via onnxruntime-web → sigmoid
              │
              ▼
          aggregate.js          top-k mean → verdict → merged flagged segments
```

Each module is a direct port of its Python counterpart; the file:line references
in the source headers point at the original. **They must stay in sync** — a drift
in the maths changes the model's input and therefore the verdict.

### The incremental driver (`session.js`)

Both input modes push frames into the same `Session` object — `pushFrame()` is
the one bit of logic literally shared between an uploaded file and a live
camera. Every ~10 seconds of new footage (one `stride_seconds`), `session.js`
recomputes `clean_video` + windowing over the **entire buffer captured so far**
and sends only the *new* window rows through the model — the ONNX call is the
one genuinely expensive step, so it's the only thing kept strictly incremental.
Recomputing the cheap signal-processing steps from scratch each cycle isn't a
shortcut: for an open-ended session there is no "whole recording" to calibrate
gaze against until it ends, so a periodically-refreshed running baseline *is*
the correct semantics, not an approximation of one.

When a session ends (file fully processed, or the user clicks Stop), `finalize()`
does one last cycle and then a **full rescore of every window** against the
final calibration baseline — so the number you actually read or download is
identical to what a single one-shot pass over the same footage would have
produced. This was verified directly: driving a `Session` through ~11,000 real
frames in 59 deliberately-misaligned incremental chunks produced windows and
probabilities that matched a direct one-shot `clean_video`+`buildWindows` call
to **zero difference**.

## Webcam sessions

Click "Use webcam," grant camera permission, then "Start analysis." Frames are
taken from `requestVideoFrameCallback` and kept on the same schedule the file
path uses — the camera's own frame timestamps run through
`extract_features.py`'s keep rule — so a live session is sampled at the same
density as the training data rather than at a fixed 15fps wall clock. A live
`MediaStream` cannot be paused for inference the way an upload can, so if the
tab falls behind (backgrounded, slow machine) capture skips ahead rather than
queuing a backlog, and frames are genuinely lost. A
session ends only when you click Stop (or the camera is disconnected); there is
no auto-cap. Nothing is ever recorded to disk — only the small numeric
per-frame signal (gaze, blink, head pose) lives in memory, and it's discarded
when the session ends.

## The model

`assets/model.onnx` is the 1D-CNN trained by `src/train.py` — the same model the
Python CLI (`src/predict.py`) uses, and the only model in the project. It is a
byte-identical copy of `models/model.onnx`; after a retrain, refresh `assets/` by
hand (see "In-browser deployment" in the top-level README).

Cross-validated on 106 videos (27 cheating / 79 clean):

| video ROC-AUC | PR-AUC | accuracy | precision | recall |
|---|---|---|---|---|
| **0.871** | **0.808** | 87.7% | **0.94** | 0.56 |

The CNN is tuned for precision at its shipped threshold (0.959): it produces very
few false accusations but misses roughly half of the cheating videos. Lower the
threshold in `assets/model.json` to trade that back toward recall — 0.80 gives
recall 0.70 at precision 0.68.

The CNN reads only the six raw gaze channels (no audio-derived features), so the
browser, which never decodes audio, sees exactly what training saw.

## No audio path

The Python pipeline uses silero-VAD plus lip-sync attribution when a recording
has an audio track. None of that is ported: the CNN does not consume any
speaking-derived feature, and every field `window_reasons()` uses is
speaking-independent. So the whole VAD/turn layer — and 55 of the 63 engineered
features — are simply not needed here.

## Verified

The ported maths was diffed against Python on real per-frame data from two
videos (one cheating, one clean), feeding both sides the identical raw signal:

| stage | worst absolute difference |
|---|---|
| `clean_video` channels | 9.8e-15 |
| window reason features | 9.8e-15 |
| **sequence tensor (model input)** | **0 — bit-identical** |
| scaler normalisation | 2.4e-7 (float32 rounding) |
| top-k aggregation | 0 |

`model.onnx` was also checked against the PyTorch checkpoint in Python: max logit
difference 2.3e-05.

**End-to-end browser parity — measured.** 25 videos were scored through real
Chrome (headless-driven, `parity.html`) and through the Python CLI on the same
footage, first 180s of each:

| | value |
|---|---|
| verdict agreement | **24/25 (96%)** |
| cheating videos: same verdict | **5/5** |
| median probability difference | 0.007 |
| mean absolute difference | 0.024 |
| **worst case** | **0.094** |
| bias | −0.016 (Chrome scores systematically lower) |

Kept-frame timestamps match Python exactly: identical frame counts on every
variable-frame-rate file, and `max|Δt| = 0.000000s` on mp4. webm carries a
constant 7ms container offset which cancels in `resample_uniform`
(`common.py:250` anchors the grid at `t[0]`).

**The residual is video decoding and cannot be removed from JS.** Chrome and
ffmpeg produce different pixels from the same webm — systematically ~1.5/255
brighter, up to 6/255, fitting `chrome ≈ 0.989 × ffmpeg + 3.4`. That propagates
to ~0.008 on gaze_x and ~0.014 on gaze_y, and from there into the probability.
The error grows with the score, because that is where the sigmoid is steep:
videos below 0.11 agree to ~0.005, videos above 0.78 differ by up to 0.094.

**What that means operationally.** The shipped threshold (0.959) is precision-
tuned and sits right against the cluster of positives — every true positive in
cross-validation clears it by less than 0.041, median +0.035. So a video more
than ~0.1 from the threshold will get the same verdict in both engines, and a
video inside that band is a coin flip. That is a property of the operating
point, not of the browser: the same sensitivity applies to any change on the
Python side too (a different ffmpeg build, a MediaPipe upgrade).

**Still not verified:** the webcam path against a real camera. `live-source.js`
has only been exercised through `session.js`'s bookkeeping tests with a stubbed
camera/model. It applies the same keep rule as the file path, but a live
`MediaStream` cannot be paused for inference, so unlike uploads it can genuinely
miss frames on a slow machine.

## Known constraints

- **Speed.** Roughly real time — a 15-minute recording takes about 15 minutes,
  since playback is paused for each inference. The "analyse first N minutes"
  control exists for uploads. The sampling rate is *not* adjustable: it is
  dictated by `extract_features.py`'s schedule, not chosen here.
- **`config.json` says `target_fps: 15`, but nothing samples at 15fps.**
  `extract_features.py:129` re-bases its next boundary on each kept frame's own
  timestamp (`next_t = t + 1/15`) rather than stepping a fixed grid, so the
  schedule drifts onto the source's frame grid: 12.5fps on a 25fps recording,
  ~8fps on the variable-rate webm ones, 10fps on a 30fps one where rounding tips
  it to every 3rd frame. Measured across the corpus the real rate ranges 8.0 to
  12.5fps and is never 15. The model was trained on data sampled this way, so
  `frame-sampler.js` reproduces it deliberately — see the parity contract in
  that file. Fixing the drift requires changing both sides at once, plus a
  re-extract and retrain.
- **Frames are decoded in order, not seeked.** Sampling follows the video's own
  presentation timestamps via `requestVideoFrameCallback`, the equivalent of
  OpenCV's `CAP_PROP_POS_MSEC`, and playback is paused during each inference so
  decoding cannot outrun scoring. `video.duration` is never consulted: these
  recordings report it wrongly (one 15-minute file claims 3081s, another
  `Infinity`), so the run ends at `ended` like Python's does.
- **A verdict needs at least 20 seconds of usable signal** (`window_seconds`)
  — shorter uploads or webcam sessions end in a "too short" state with no
  chart/verdict, since the model architecturally cannot score less than one
  window.
- **Face lock below 50% shows a warning** throughout scoring (not just at the
  end); below 70% the Python pipeline already treats features as unreliable.
- Uses ES modules, `async`/`await` and typed arrays rather than the house ES5
  style — `@mediapipe/tasks-vision` and `onnxruntime-web` ship as ES modules and
  the pipeline is inherently async and numeric.
