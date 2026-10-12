/**
 * signal.js — port of the numerical core in src/common.py.
 *
 * Every function here mirrors a Python original 1:1; the file:line references
 * point at the source of truth. Keep them in sync — a drift here changes the
 * model's input and therefore the verdict.
 */

// ---------------------------------------------------------------- array helpers

export function mean(a) {
  let s = 0, n = 0;
  for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i])) { s += a[i]; n++; }
  return n ? s / n : NaN;
}

export function variance(a) {
  const m = mean(a);
  if (!Number.isFinite(m)) return 0;
  let s = 0, n = 0;
  for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i])) { s += (a[i] - m) ** 2; n++; }
  return n ? s / n : 0;
}

export const std = (a) => Math.sqrt(variance(a));

export function median(a) {
  const v = [];
  for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i])) v.push(a[i]);
  if (!v.length) return NaN;
  v.sort((x, y) => x - y);
  const h = v.length >> 1;
  return v.length % 2 ? v[h] : (v[h - 1] + v[h]) / 2;
}

/** numpy.percentile with linear interpolation (the default method). */
export function percentile(a, q) {
  const v = [];
  for (let i = 0; i < a.length; i++) if (Number.isFinite(a[i])) v.push(a[i]);
  if (!v.length) return NaN;
  v.sort((x, y) => x - y);
  const pos = (q / 100) * (v.length - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return lo === hi ? v[lo] : v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

export function ptp(a) {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i])) continue;
    if (a[i] < mn) mn = a[i];
    if (a[i] > mx) mx = a[i];
  }
  return mx === -Infinity ? 0 : mx - mn;
}

/**
 * numpy.interp — piecewise-linear interpolation with edge clamping.
 * `xp` must be increasing.
 */
export function interp(x, xp, fp) {
  const out = new Float64Array(x.length);
  let j = 0;
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    if (xi <= xp[0]) { out[i] = fp[0]; continue; }
    if (xi >= xp[xp.length - 1]) { out[i] = fp[fp.length - 1]; continue; }
    while (j < xp.length - 2 && xp[j + 1] < xi) j++;
    while (j > 0 && xp[j] > xi) j--;
    const t = (xi - xp[j]) / (xp[j + 1] - xp[j]);
    out[i] = fp[j] + t * (fp[j + 1] - fp[j]);
  }
  return out;
}

/**
 * numpy.gradient with a uniform spacing — second-order central differences in
 * the interior, first-order one-sided at the edges.
 */
export function gradient(a, dt) {
  const n = a.length, out = new Float64Array(n);
  if (n < 2) return out;
  out[0] = (a[1] - a[0]) / dt;
  out[n - 1] = (a[n - 1] - a[n - 2]) / dt;
  for (let i = 1; i < n - 1; i++) out[i] = (a[i + 1] - a[i - 1]) / (2 * dt);
  return out;
}

// --------------------------------------------------- signal cleaning (common.py:193-257)

/** interpolate_gaps — common.py:193. Bridges invalid runs shorter than max_gap_seconds. */
export function interpolateGaps(values, valid, dt, maxGapSeconds) {
  const n = values.length;
  const out = Float64Array.from(values);
  let nValid = 0;
  for (let i = 0; i < n; i++) if (valid[i]) nValid++;
  if (nValid < 2) return out;

  const maxGap = Math.max(1, Math.round(maxGapSeconds / Math.max(dt, 1e-6)));
  for (let i = 0; i < n; i++) if (!valid[i]) out[i] = NaN;

  let i = 0;
  while (i < n) {
    if (!valid[i]) {
      let j = i;
      while (j < n && !valid[j]) j++;
      const run = j - i;
      // Python's `has_left = i - 1 >= 0` / `has_right = j < n`
      if (run <= maxGap && i - 1 >= 0 && j < n) {
        const lo = i - 1, hi = j;
        for (let k = i; k < j; k++) {
          const t = (k - lo) / (hi - lo);
          out[k] = out[lo] + t * (out[hi] - out[lo]);
        }
      }
      i = j;
    } else i++;
  }
  return out;
}

/** median_filter_1d — common.py:226. NaN-aware odd-kernel median filter. */
export function medianFilter1d(values, k) {
  if (!k || k <= 1) return Float64Array.from(values);
  if (k % 2 === 0) k += 1;
  const n = values.length, half = k >> 1;
  const out = Float64Array.from(values);
  const buf = [];
  for (let i = 0; i < n; i++) {
    buf.length = 0;
    const lo = Math.max(0, i - half), hi = Math.min(n, i + half + 1);
    for (let j = lo; j < hi; j++) if (Number.isFinite(values[j])) buf.push(values[j]);
    if (buf.length) {
      buf.sort((a, b) => a - b);
      const h = buf.length >> 1;
      out[i] = buf.length % 2 ? buf[h] : (buf[h - 1] + buf[h]) / 2;
    }
  }
  return out;
}

/** resample_uniform — common.py:244. Irregular (t, v) onto a uniform grid at `fps`. */
export function resampleUniform(t, values, fps) {
  const dt = 1 / fps;
  if (t.length < 2) return { t, values: Float64Array.from(values), dt };
  const t0 = t[0], t1 = t[t.length - 1];
  const n = Math.max(2, Math.floor((t1 - t0) / dt) + 1);
  const tu = new Float64Array(n);
  for (let i = 0; i < n; i++) tu[i] = t0 + i * dt;

  const ft = [], fv = [];
  for (let i = 0; i < values.length; i++) {
    if (Number.isFinite(values[i])) { ft.push(t[i]); fv.push(values[i]); }
  }
  if (ft.length < 2) return { t: tu, values: new Float64Array(n).fill(NaN), dt };
  return { t: tu, values: interp(tu, ft, fv), dt };
}

/** _fill_window — build_windows.py:116. Fills residual NaNs so detectors never see them. */
export function fillWindow(a) {
  const n = a.length;
  const out = Float64Array.from(a);
  let anyFinite = false, allFinite = true;
  for (let i = 0; i < n; i++) {
    if (Number.isFinite(out[i])) anyFinite = true; else allFinite = false;
  }
  if (allFinite) return out;
  if (!anyFinite) return new Float64Array(n);

  const xp = [], fp = [];
  for (let i = 0; i < n; i++) if (Number.isFinite(out[i])) { xp.push(i); fp.push(out[i]); }
  const missing = [];
  for (let i = 0; i < n; i++) if (!Number.isFinite(out[i])) missing.push(i);
  const filled = interp(missing, xp, fp);
  for (let i = 0; i < missing.length; i++) out[missing[i]] = filled[i];
  return out;
}

// ------------------------------------------- saccades & reading sweeps (common.py:272-430)

/** _mad — common.py:272. Median absolute deviation, scaled to approximate std. */
export function mad(x) {
  if (!x.length) return 0;
  const med = median(x);
  const dev = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) dev[i] = Math.abs(x[i] - med);
  return 1.4826 * median(dev);
}

/** velocity_threshold — common.py:280. Adaptive, scale-invariant saccade cutoff. */
export function velocityThreshold(gazeX, gazeY, dt, cfg) {
  if (!cfg.adaptive_saccade) return cfg.saccade_velocity_thresh;
  const vx = gradient(gazeX, dt), vy = gradient(gazeY, dt);
  const speed = new Float64Array(vx.length);
  for (let i = 0; i < vx.length; i++) speed[i] = Math.hypot(vx[i], vy[i]);
  return Math.max(median(speed) + cfg.saccade_velocity_k * mad(speed), 1e-6);
}

/** detect_saccades — common.py:291. Velocity-threshold segmentation on 2D gaze. */
export function detectSaccades(gazeX, gazeY, dt, cfg, velThresh) {
  const n = gazeX.length;
  if (n < 3) return [];
  if (velThresh == null) velThresh = velocityThreshold(gazeX, gazeY, dt, cfg);

  const vx = gradient(gazeX, dt), vy = gradient(gazeY, dt);
  const isSacc = new Uint8Array(n);
  for (let i = 0; i < n; i++) isSacc[i] = Math.hypot(vx[i], vy[i]) > velThresh ? 1 : 0;

  const out = [];
  let i = 0;
  while (i < n) {
    if (isSacc[i]) {
      let j = i;
      while (j < n && isSacc[j]) j++;
      const e = Math.min(j, n - 1);
      const dx = gazeX[e] - gazeX[i], dy = gazeY[e] - gazeY[i];
      const amp = Math.hypot(dx, dy);
      if (amp >= cfg.min_saccade_amplitude) {
        // NB: Python stores the SIGNED x-delta as `amplitude` (common.py:316).
        out.push({ start: i, end: j, amplitude: dx, direction: Math.atan2(dy, dx), duration: (j - i) * dt });
      }
      i = j;
    } else i++;
  }
  return out;
}

/** detect_return_sweeps — common.py:324. Reading "carriage returns" on gaze_x. */
export function detectReturnSweeps(gx, dt, cfg, velThresh) {
  const n = gx.length;
  const lo = percentile(gx, 5), hi = percentile(gx, 95);
  const gazeRange = hi - lo;
  if (!(gazeRange >= cfg.sweep_min_range) || n < 5) return { times: [], progressions: [] };

  const progMin = cfg.sweep_min_progression_frac * gazeRange;
  const retMin = cfg.return_saccade_min_amp_frac * gazeRange;
  const vx = gradient(gx, dt);

  const times = [], progressions = [];
  let i = 0, lastReturnEnd = 0;
  while (i < n) {
    if (vx[i] < -velThresh) {
      let j = i;
      while (j < n && vx[j] < -velThresh) j++;
      const jump = gx[i] - gx[Math.min(j, n - 1)];   // positive = leftward
      if (jump >= retMin) {
        let mn = Infinity, mx = -Infinity;
        for (let k = lastReturnEnd; k <= i; k++) {
          if (gx[k] < mn) mn = gx[k];
          if (gx[k] > mx) mx = gx[k];
        }
        const prog = mx === -Infinity ? 0 : mx - mn;
        if (prog >= progMin) {
          times.push(i * dt);
          progressions.push(prog);
          lastReturnEnd = j;
        }
      }
      i = j;
    } else i++;
  }
  return { times, progressions };
}

/** _autocorr_peak — common.py:417. Brute-force bounded-lag autocorrelation (no FFT). */
export function autocorrPeak(x, dt, maxLagSeconds, minLagSeconds = 0) {
  const n = x.length;
  const m = mean(x);
  const c = new Float64Array(n);
  let varSum = 0;
  for (let i = 0; i < n; i++) { c[i] = x[i] - m; varSum += c[i] * c[i]; }
  if (varSum < 1e-9 || n < 4) return { peak: 0, lag: 0 };

  const maxLag = Math.min(n - 1, Math.round(maxLagSeconds / Math.max(dt, 1e-6)));
  const minLag = Math.max(1, Math.round(minLagSeconds / Math.max(dt, 1e-6)));
  if (maxLag < 2 || minLag >= maxLag) return { peak: 0, lag: 0 };

  let bestVal = -Infinity, bestLag = minLag;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let s = 0;
    for (let i = 0; i + lag < n; i++) s += c[i] * c[i + lag];
    const v = s / varSum;
    if (v > bestVal) { bestVal = v; bestLag = lag; }
  }
  return { peak: bestVal, lag: bestLag * dt };
}

/** reading_sweep_features — common.py:363. The line-by-line reading rhythm. */
export function readingSweepFeatures(gazeX, dt, cfg, velThresh) {
  const n = gazeX.length;
  const duration = Math.max(n * dt, 1e-6);
  const out = {
    return_sweep_count: 0,
    lines_per_minute: 0,
    mean_line_progression: 0,
    sweep_regularity: 0,
    autocorr_peak: 0,
    autocorr_lag_s: 0,
  };
  let anyFinite = false;
  for (let i = 0; i < n; i++) if (Number.isFinite(gazeX[i])) { anyFinite = true; break; }
  if (n < 5 || !anyFinite) return out;

  const gx = Float64Array.from(gazeX);
  const fallback = mean(gx);
  for (let i = 0; i < n; i++) if (!Number.isFinite(gx[i])) gx[i] = Number.isFinite(fallback) ? fallback : 0;

  const gazeRange = percentile(gx, 95) - percentile(gx, 5);
  if (velThresh == null) velThresh = velocityThreshold(gx, new Float64Array(n), dt, cfg);

  if (gazeRange >= cfg.sweep_min_range) {
    const { times, progressions } = detectReturnSweeps(gx, dt, cfg, velThresh);
    out.return_sweep_count = times.length;
    out.lines_per_minute = (60 * times.length) / duration;
    if (progressions.length) out.mean_line_progression = mean(progressions);
    if (times.length >= 3) {
      const gaps = [];
      for (let i = 1; i < times.length; i++) gaps.push(times[i] - times[i - 1]);
      const cv = std(gaps) / (mean(gaps) + 1e-6);
      out.sweep_regularity = Math.max(0, 1 - cv);
    }
  }

  const { peak, lag } = autocorrPeak(gx, dt, cfg.autocorr_max_lag_seconds, cfg.reading_lag_min_s);
  out.autocorr_peak = peak;
  out.autocorr_lag_s = lag;
  return out;
}

// -------------------------------------------------- window stats (build_windows.py:329-343)

/** _grid_entropy — build_windows.py:329. Occupancy entropy over a grid_n² grid on [-1,1]². */
export function gridEntropy(gx, gy, gridN) {
  const counts = new Float64Array(gridN * gridN);
  for (let i = 0; i < gx.length; i++) {
    // Python: np.clip(((g + 1) / 2 * grid_n).astype(int), 0, grid_n - 1).
    // .astype(int) truncates toward zero, which differs from Math.floor for
    // negatives — Math.trunc reproduces it exactly.
    const xs = Math.min(Math.max(Math.trunc(((gx[i] + 1) / 2) * gridN), 0), gridN - 1);
    const ys = Math.min(Math.max(Math.trunc(((gy[i] + 1) / 2) * gridN), 0), gridN - 1);
    counts[ys * gridN + xs] += 1;
  }
  let total = 0, used = 0;
  for (let i = 0; i < counts.length; i++) { total += counts[i]; if (counts[i] > 0) used++; }
  let ent = 0;
  if (total > 0) {
    for (let i = 0; i < counts.length; i++) {
      if (counts[i] > 0) { const p = counts[i] / total; ent -= p * Math.log2(p); }
    }
  }
  return { entropy: ent, cells: used };
}

/** _safe_corr — build_windows.py:340. Pearson correlation, 0 when either side is flat. */
export function safeCorr(a, b) {
  if (std(a) < 1e-9 || std(b) < 1e-9) return 0;
  const ma = mean(a), mb = mean(b);
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] - ma, y = b[i] - mb;
    num += x * y; da += x * x; db += y * y;
  }
  const den = Math.sqrt(da * db);
  return den < 1e-12 ? 0 : num / den;
}
