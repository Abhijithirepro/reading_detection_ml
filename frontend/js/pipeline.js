/**
 * pipeline.js — clean_video (build_windows.py:45-102) plus the windowing loop
 * from predict.py:90-110.
 *
 * The speaking/turn engine is deliberately absent. The shipped CNN scores only
 * the six raw sequence channels, and every field window_reasons() reads
 * (predict.py:125-136) is independent of the speaking mask — so the whole
 * audio/VAD layer, and 55 of the 63 engineered features, are not needed here.
 */

import {
  interpolateGaps, medianFilter1d, resampleUniform, fillWindow, interp,
  median, mean, variance, velocityThreshold, detectSaccades,
  readingSweepFeatures, gridEntropy, safeCorr,
} from './signal.js';

const BASE_CHANNELS = ['gaze_x', 'gaze_y', 'yaw', 'pitch', 'roll', 'ear', 'mouth'];

/** _median_dt — build_windows.py:105. Median positive spacing of the raw timestamps. */
function medianDt(t, fallbackFps) {
  if (t.length < 2) return 1 / fallbackFps;
  const d = [];
  for (let i = 1; i < t.length; i++) if (t[i] - t[i - 1] > 0) d.push(t[i] - t[i - 1]);
  return d.length ? median(d) : 1 / fallbackFps;
}

/**
 * clean_video — build_windows.py:45.
 * @param frames array of the per-frame dicts from MediaPipeBackend.process
 * @param times  matching capture timestamps
 */
export function cleanVideo(frames, times, cfg) {
  const n = frames.length;
  const t = Float64Array.from(times);
  const face = new Uint8Array(n);
  const blink = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    face[i] = frames[i].face_found > 0 ? 1 : 0;
    blink[i] = frames[i].blink;
  }

  // Gaze is trustworthy only when a face is present AND the eye is open.
  const gazeValid = new Uint8Array(n);
  const poseValid = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    poseValid[i] = face[i];
    gazeValid[i] = face[i] && blink[i] === 0 ? 1 : 0;
  }

  const rawDt = medianDt(t, cfg.target_fps);
  const out = {};
  let dt = 1 / cfg.target_fps;

  for (const ch of BASE_CHANNELS) {
    const raw = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const v = frames[i][ch];
      raw[i] = Number.isFinite(v) ? v : NaN;
    }
    const valid = (ch === 'gaze_x' || ch === 'gaze_y') ? gazeValid : poseValid;
    let filled = interpolateGaps(raw, valid, rawDt, cfg.max_gap_seconds);
    filled = medianFilter1d(filled, cfg.median_filter_k);
    const r = resampleUniform(t, filled, cfg.target_fps);
    out[ch] = r.values;
    dt = r.dt;
  }

  const blinkU = resampleUniform(t, blink, cfg.target_fps).values;
  const faceU = resampleUniform(t, Float64Array.from(poseValid), cfg.target_fps).values;
  out.blink = Float64Array.from(blinkU, (v) => (v > 0.5 ? 1 : 0));
  out.valid = faceU;

  // Per-candidate calibration (build_windows.py:86-90): measure reading as a
  // deviation from this person's own neutral gaze, not an absolute position.
  // This is one of the two changes the README credits for the accuracy gain.
  if (cfg.calibrate_gaze) {
    for (const ch of ['gaze_x', 'gaze_y']) {
      const med = median(out[ch]);
      if (Number.isFinite(med)) {
        for (let i = 0; i < out[ch].length; i++) out[ch][i] -= med;
      }
    }
  }

  out.dt = dt;
  out.duration = out.gaze_x.length * dt;
  return out;
}

/** _window_sequence — build_windows.py:396. Resample a window to (seq_len, C). */
function windowSequence(clean, start, end, seqLen, seqChannels) {
  const len = end - start;
  const srcIdx = new Float64Array(seqLen);
  for (let i = 0; i < seqLen; i++) srcIdx[i] = (i * (len - 1)) / (seqLen - 1);
  const base = new Float64Array(len);
  for (let i = 0; i < len; i++) base[i] = i;

  const cols = seqChannels.map((ch) => {
    const a = fillWindow(clean[ch].subarray(start, end));
    return interp(srcIdx, base, a);
  });

  const outRow = new Float32Array(seqLen * seqChannels.length);
  for (let i = 0; i < seqLen; i++) {
    for (let c = 0; c < cols.length; c++) outRow[i * cols.length + c] = cols[c][i];
  }
  return outRow;
}

/**
 * The subset of compute_window_features (build_windows.py:129-234) that
 * window_reasons() actually consults. Computing all 63 would be dead weight —
 * the CNN never sees them.
 */
function reasonFeatures(clean, start, end, cfg) {
  const dt = clean.dt;
  const gx = fillWindow(clean.gaze_x.subarray(start, end));
  const gy = fillWindow(clean.gaze_y.subarray(start, end));
  const yaw = fillWindow(clean.yaw.subarray(start, end));
  const blink = clean.blink.subarray(start, end);
  const n = gx.length;
  const dur = Math.max(n * dt, 1e-6);

  const vthr = velocityThreshold(gx, gy, dt, cfg);
  const sweep = readingSweepFeatures(gx, dt, cfg, vthr);
  const sacc = detectSaccades(gx, gy, dt, cfg, vthr);

  let off = 0;
  for (let i = 0; i < n; i++) {
    if (Math.abs(gx[i]) > cfg.off_center_thresh || Math.abs(gy[i]) > cfg.off_center_thresh) off++;
  }
  const { cells } = gridEntropy(gx, gy, cfg.grid_n);

  // Blink rate counts rising edges, matching build_windows.py:213.
  let events = 0;
  for (let i = 1; i < blink.length; i++) {
    if ((blink[i] > 0.5 ? 1 : 0) - (blink[i - 1] > 0.5 ? 1 : 0) === 1) events++;
  }

  return {
    lines_per_minute: sweep.lines_per_minute,
    autocorr_peak: sweep.autocorr_peak,
    pct_off_center: off / n,
    gaze_grid_cells: cells,
    saccade_rate: (60 * sacc.length) / dur,
    corr_yaw_gazex: safeCorr(yaw, gx),
    yaw_var: variance(yaw),
    blink_rate: (60 * events) / dur,
  };
}

/**
 * Slide the analysis window across the cleaned signal.
 * Mirrors predict.py:90-110. Returns { windows, seqFlat, seqLen, nChannels }.
 */
export function buildWindows(clean, cfg, artifact) {
  const n = clean.gaze_x.length;
  const winLen = Math.round(cfg.window_seconds * cfg.target_fps);
  const stride = Math.max(1, Math.round(cfg.stride_seconds * cfg.target_fps));
  const seqLen = artifact.seq_len;
  const seqChannels = artifact.seq_channels;

  if (n < winLen) {
    throw new Error(
      `Video is too short to analyse: ${(n * clean.dt).toFixed(1)}s of usable signal, ` +
      `but the model needs at least ${cfg.window_seconds}s.`
    );
  }

  const windows = [];
  const seqs = [];
  for (let start = 0; start + winLen <= n; start += stride) {
    const end = start + winLen;
    windows.push({
      win_start_s: +(start * clean.dt).toFixed(2),
      win_end_s: +(end * clean.dt).toFixed(2),
      ...reasonFeatures(clean, start, end, cfg),
    });
    seqs.push(windowSequence(clean, start, end, seqLen, seqChannels));
  }

  const nChannels = seqChannels.length;
  const seqFlat = new Float32Array(seqs.length * seqLen * nChannels);
  for (let i = 0; i < seqs.length; i++) seqFlat.set(seqs[i], i * seqLen * nChannels);

  return { windows, seqFlat, seqLen, nChannels };
}
