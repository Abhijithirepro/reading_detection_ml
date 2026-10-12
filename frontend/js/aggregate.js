/**
 * aggregate.js — port of the verdict logic in src/predict.py:116-172.
 * Window probabilities -> video probability -> verdict -> readable segments.
 */

const REASON_ORDER = [
  'reading sweeps',
  'off-center gaze',
  'eyes move, head still',
  'suppressed blinking',
];

/** topk_video_prob — predict.py:116. Mean of the top-k window probabilities. */
export function topkVideoProb(probs, cfg) {
  const p = Array.from(probs).sort((a, b) => b - a);
  const k = Math.max(cfg.topk_min, Math.ceil(cfg.topk_fraction * p.length));
  const take = Math.min(k, p.length);
  let s = 0;
  for (let i = 0; i < take; i++) s += p[i];
  return take ? s / take : 0;
}

/** window_reasons — predict.py:125. Rule-based tags for why a window looks suspicious. */
export function windowReasons(row) {
  const reasons = [];
  const g = (k, d) => (row[k] === undefined ? d : row[k]);

  if (g('lines_per_minute', 0) >= 12 && g('autocorr_peak', 0) >= 0.4) reasons.push('reading sweeps');
  if (g('pct_off_center', 0) >= 0.5 && g('gaze_grid_cells', 9) <= 3) reasons.push('off-center gaze');
  if (g('saccade_rate', 0) >= 20 && Math.abs(g('corr_yaw_gazex', 1)) < 0.25 && g('yaw_var', 1e9) < 25) {
    reasons.push('eyes move, head still');
  }
  if (g('blink_rate', 99) <= 4) reasons.push('suppressed blinking');
  return reasons;
}

/** mmss — predict.py:139. */
export function mmss(sec) {
  const total = Math.round(sec);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

/** merge_segments — predict.py:144. Collapse consecutive flagged windows into ranges. */
export function mergeSegments(windows, probs, flagThreshold) {
  const n = windows.length;
  const segments = [];
  let i = 0;
  while (i < n) {
    if (probs[i] >= flagThreshold) {
      let j = i;
      while (j < n && probs[j] >= flagThreshold) j++;

      const counts = new Map();
      let peak = -Infinity;
      for (let k = i; k < j; k++) {
        if (probs[k] > peak) peak = probs[k];
        for (const r of windowReasons(windows[k])) counts.set(r, (counts.get(r) || 0) + 1);
      }

      let reason = 'elevated gaze anomaly';
      if (counts.size) {
        reason = [...counts.entries()]
          .sort((a, b) => (b[1] - a[1]) || (REASON_ORDER.indexOf(a[0]) - REASON_ORDER.indexOf(b[0])))[0][0];
      }

      const startS = windows[i].win_start_s;
      const endS = windows[j - 1].win_end_s;
      segments.push({
        start_s: startS,
        end_s: endS,
        range: `${mmss(startS)}–${mmss(endS)}`,
        peak_prob: peak,
        reason,
      });
      i = j;
    } else i++;
  }
  return segments;
}

/** score_video's aggregation half — predict.py:178-201. */
export function buildResult(windows, probs, cfg, artifact, flagThreshold = null) {
  const threshold = artifact.threshold;
  const flagThr = flagThreshold ?? threshold;
  const videoProb = topkVideoProb(probs, cfg);

  let nFlagged = 0;
  for (let i = 0; i < probs.length; i++) if (probs[i] >= flagThr) nFlagged++;

  return {
    verdict: videoProb >= threshold ? 'CHEATING' : 'clean',
    probability: videoProb,
    threshold,
    flag_threshold: flagThr,
    n_windows: windows.length,
    n_flagged_windows: nFlagged,
    model: { kind: artifact.kind, feature_set: artifact.feature_set },
    flagged_segments: mergeSegments(windows, probs, flagThr),
  };
}
