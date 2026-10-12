/**
 * session.js — the shared incremental driver for both file-upload and webcam
 * input. Neither input source needs to know anything about scoring; both just
 * feed frames into the same Session object.
 *
 * Design: cleanVideo()/buildWindows() (pipeline.js) are synchronous and pure —
 * recomputing them from scratch over the ENTIRE frame buffer every ~stride is
 * trivially cheap at these data sizes (tens of thousands of samples, microseconds
 * of work) and is the *correct* semantics for a live session's gaze-calibration
 * baseline: there is no "the whole recording" to calibrate against until the
 * session ends, so a periodically-recomputed running median is what the
 * Python-side per-video calibration becomes for an open-ended stream, not a
 * compromise. The only genuinely non-trivial cost — the ONNX call — is kept
 * strictly incremental: each cycle scores only the window rows that are new
 * since the last cycle.
 *
 * finalize() then does one full rescore of every window against the final
 * calibration baseline, so the number the user reads/downloads at the end is
 * identical to what a one-shot run over the same footage would have produced —
 * reconciling "rolling updates" with "an exact final answer."
 */

import { cleanVideo, buildWindows } from './pipeline.js';
import { buildResult } from './aggregate.js';
import { loadVideo, sampleFrames, assertNoMissedFrames } from './frame-sampler.js';
import { startLiveCapture } from './live-source.js';

const TOO_SHORT_RE = /too short to analyse/i;

export class Session {
  constructor(cfg, artifact, backend, runner) {
    this.cfg = cfg;
    this.artifact = artifact;
    this.backend = backend;
    this.runner = runner;

    this.frames = [];
    this.times = [];
    this.faceHits = 0;

    this.windows = [];
    this.probs = new Float64Array(0);
    this.scoredWindows = 0;

    // 'warming' | 'scoring' | 'finished' | 'too-short' | 'error'
    this.status = 'warming';
    this.error = null;
    this.stoppedEarly = false;
    this.progress = null;   // {index, total, seconds, endAt} — upload mode only

    this._cycling = false;
    this._lastCycleFrameCount = 0;
    this.strideFrames = Math.max(1, Math.round(cfg.stride_seconds * cfg.target_fps));
    this.startedAt = performance.now();
  }

  /** The one bit of logic literally shared between both input sources. */
  pushFrame(canvas, t) {
    const f = this.backend.process(canvas);
    if (f.face_found) this.faceHits++;
    this.frames.push(f);
    this.times.push(t);
  }

  setProgress(p) { this.progress = p; }

  /** True once the buffer has grown by a full stride since the last cycle. */
  shouldCycle() {
    if (this._cycling) return false;
    return this.frames.length - this._lastCycleFrameCount >= this.strideFrames;
  }

  /** Synchronous half of a cycle: recompute clean+windows, or stay 'warming'. */
  _rebuild() {
    if (this.frames.length < 2) return null;
    try {
      const clean = cleanVideo(this.frames, this.times, this.cfg);
      const built = buildWindows(clean, this.cfg, this.artifact);
      this._lastCycleFrameCount = this.frames.length;
      return built;
    } catch (err) {
      if (TOO_SHORT_RE.test(err.message || '')) {
        this._lastCycleFrameCount = this.frames.length;
        return null;
      }
      throw err;
    }
  }

  /** Async half: score only the window rows that are new since last cycle. */
  async _scoreDelta(built) {
    const { windows, seqFlat, seqLen, nChannels } = built;
    const newCount = windows.length - this.scoredWindows;
    if (newCount <= 0) { this.windows = windows; return; }

    const stride = seqLen * nChannels;
    const slice = seqFlat.subarray(this.scoredWindows * stride, windows.length * stride);
    const newProbs = await this.runner.score(slice, newCount, seqLen, nChannels);

    const merged = new Float64Array(windows.length);
    merged.set(this.probs.subarray(0, this.scoredWindows));
    merged.set(newProbs, this.scoredWindows);

    // Reason-tag features are replaced wholesale — the calibration median can
    // shift slightly as more data arrives, so the *displayed* tags should
    // track the freshest baseline even though only new windows get rescored.
    this.windows = windows;
    this.probs = merged;
    this.scoredWindows = windows.length;
  }

  /** Run one incremental cycle. Safe to call redundantly — cycles never overlap. */
  async runCycle() {
    if (this._cycling) return;
    this._cycling = true;
    try {
      const built = this._rebuild();
      if (built) {
        await this._scoreDelta(built);
        if (this.status === 'warming') this.status = 'scoring';
      }
    } catch (err) {
      this.status = 'error';
      this.error = err.message || String(err);
    } finally {
      this._cycling = false;
    }
  }

  /** Pick up any tail, then do one full rescore for an exact final answer. */
  async finalize() {
    await this.runCycle();

    if (this.status === 'error') return;
    if (this.windows.length === 0) { this.status = 'too-short'; return; }

    try {
      const clean = cleanVideo(this.frames, this.times, this.cfg);
      const built = buildWindows(clean, this.cfg, this.artifact);
      const probs = await this.runner.score(built.seqFlat, built.windows.length, built.seqLen, built.nChannels);
      this.windows = built.windows;
      this.probs = probs;
      this.scoredWindows = built.windows.length;
      this.status = 'finished';
    } catch (err) {
      this.status = 'error';
      this.error = err.message || String(err);
    }
  }

  get faceRatio() {
    return this.frames.length ? this.faceHits / this.frames.length : 0;
  }

  get elapsedSeconds() {
    return (performance.now() - this.startedAt) / 1000;
  }

  /** Cheap, pure — this is exactly what the UI renders after every cycle. */
  snapshot() {
    const result = this.windows.length
      ? buildResult(this.windows, this.probs, this.cfg, this.artifact)
      : null;
    return {
      status: this.status,
      error: this.error,
      stoppedEarly: this.stoppedEarly,
      nFrames: this.frames.length,
      faceRatio: this.faceRatio,
      elapsedSeconds: this.elapsedSeconds,
      progress: this.progress,
      windows: this.windows,
      probs: this.probs,
      result,
    };
  }
}

/**
 * Drive one Session over an uploaded file. sampleFrames() runs unmodified and
 * unthrottled by scoring; an independent 1s ticker watches the growing buffer
 * and opportunistically runs a cycle, decoupled from sampleFrames's own loop
 * (which never awaits its callbacks).
 */
export async function runFileSession(file, cfg, artifact, backend, runner, opts) {
  opts = opts || {};
  const { maxSeconds = null, signal = null, onSnapshot = null } = opts;

  const session = new Session(cfg, artifact, backend, runner);
  onSnapshot?.(session.snapshot());

  const video = await loadVideo(file);
  const ticker = setInterval(() => {
    if (session.shouldCycle()) {
      session.runCycle().then(() => onSnapshot?.(session.snapshot()));
    }
  }, 1000);

  let missed = 0;
  try {
    ({ missed } = await sampleFrames(video, cfg, session.pushFrame.bind(session), {
      maxSeconds,
      signal,
      onProgress: (p) => {
        session.setProgress(p);
        onSnapshot?.(session.snapshot());
      },
    }));
  } catch (err) {
    if (err.name === 'AbortError') session.stoppedEarly = true;
    else throw err;
  } finally {
    clearInterval(ticker);
    URL.revokeObjectURL(video.src);
  }

  // A run the user stopped is allowed to be partial; a run that claims to be
  // complete is not allowed to be silently short of frames the reference
  // pipeline would have seen.
  if (!session.stoppedEarly) assertNoMissedFrames(missed);

  await session.finalize();
  onSnapshot?.(session.snapshot());
  return session;
}

/**
 * Drive one Session over a live camera <video> (already opened by the caller
 * via live-source.js's openCamera — this function only owns the capture LOOP,
 * not the MediaStream's lifecycle; the caller is responsible for stopCamera()).
 * Resolves once the session is stopped (abort signal) or the track ends.
 */
export function runLiveSession(video, cfg, artifact, backend, runner, opts) {
  opts = opts || {};
  const { signal = null, onSnapshot = null } = opts;

  const session = new Session(cfg, artifact, backend, runner);
  onSnapshot?.(session.snapshot());

  return new Promise((resolve) => {
    let settled = false;

    const finish = async (stoppedEarly) => {
      if (settled) return;
      settled = true;
      capture.stop();
      session.stoppedEarly = !!stoppedEarly;
      await session.finalize();
      onSnapshot?.(session.snapshot());
      resolve(session);
    };

    const capture = startLiveCapture(video, cfg, session.pushFrame.bind(session), {
      onProgress: async () => {
        if (session.shouldCycle()) {
          await session.runCycle();
          onSnapshot?.(session.snapshot());
        }
      },
      onTrackEnded: () => { finish(true); },
    });

    if (signal) {
      if (signal.aborted) finish(true);
      else signal.addEventListener('abort', () => finish(true), { once: true });
    }
  });
}
