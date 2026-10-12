"""
reading.py (v2, new) — window probabilities -> per-second reading score ->
reading segments -> video-level statistics.

Shared by train.py, predict.py and the web app so the numbers they report are
computed one way only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_second(win_start: np.ndarray, win_end: np.ndarray, prob: np.ndarray,
               n_sec: int = None, smooth_s: int = 3, center_radius_s: float = 2.0) -> np.ndarray:
    """Per-second reading probability on the VIDEO clock (second i = [i, i+1)).

    Each second gets the mean probability of the windows whose CENTRE lies within
    `center_radius_s` of it (an 8 s window says most about its middle; spreading
    it over all 8 s it covers smeared segment edges by several seconds). Seconds
    near the ends fall back to any covering window. A centred rolling median of
    `smooth_s` seconds then removes single-window flicker. Unscored seconds = NaN.
    """
    win_start, win_end, prob = map(np.asarray, (win_start, win_end, prob))
    if n_sec is None:
        n_sec = int(np.ceil(win_end.max())) if len(win_end) else 0
    acc, cnt = np.zeros(n_sec), np.zeros(n_sec)
    cov_acc, cov_cnt = np.zeros(n_sec), np.zeros(n_sec)
    for a, b, p in zip(win_start, win_end, prob):
        c = (a + b) / 2
        i0, i1 = max(0, int(np.floor(c - center_radius_s))), min(n_sec, int(np.ceil(c + center_radius_s)))
        acc[i0:i1] += p
        cnt[i0:i1] += 1
        j0, j1 = max(0, int(np.floor(a + 0.5))), min(n_sec, int(np.floor(b + 0.5)))
        cov_acc[j0:j1] += p
        cov_cnt[j0:j1] += 1
    out = np.where(cnt > 0, acc / np.maximum(cnt, 1),
                   np.where(cov_cnt > 0, cov_acc / np.maximum(cov_cnt, 1), np.nan))
    cnt = np.maximum(cnt, cov_cnt)
    if smooth_s and smooth_s > 1:
        out = pd.Series(out).rolling(smooth_s, center=True, min_periods=1).median().to_numpy()
        out[cnt == 0] = np.nan
    return out


def segments(score: np.ndarray, threshold: float, min_len_s: float = 4.0,
             merge_gap_s: float = 3.0) -> list:
    """[(start_s, end_s, peak, mean)] where score >= threshold, short gaps merged,
    short blips dropped."""
    flag = np.nan_to_num(score, nan=0.0) >= threshold
    segs, i, n = [], 0, len(flag)
    while i < n:
        if flag[i]:
            j = i
            while j < n and flag[j]:
                j += 1
            segs.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for s, e in segs:
        if merged and s - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        if e - s >= min_len_s:
            seg = score[s:e]
            out.append((float(s), float(e), float(np.nanmax(seg)), float(np.nanmean(seg))))
    return out


def flagged_mask(segs: list, n_sec: int) -> np.ndarray:
    m = np.zeros(n_sec, dtype=bool)
    for s, e, *_ in segs:
        m[int(s):int(e)] = True
    return m


def video_stats(score: np.ndarray, speaking_s: float, threshold: float,
                min_len_s: float = 4.0, merge_gap_s: float = 3.0, speaking=None) -> dict:
    """Video-level evidence: how much of the recording looks like reading.
    `speaking` (optional, per second 0..1) adds statistics over speaking seconds."""
    segs = segments(score, threshold, min_len_s, merge_gap_s)
    read_s = float(sum(e - s for s, e, *_ in segs))
    valid = score[np.isfinite(score)]
    top = np.sort(valid)[::-1]
    k = max(30, int(0.1 * len(top)))
    sp_scores = np.array([])
    if speaking is not None:
        sp = np.asarray(speaking, dtype=float)[:len(score)]
        sp_scores = score[:len(sp)][(sp >= 0.5) & np.isfinite(score[:len(sp)])]
    return dict(
        speech_prob_mean=float(sp_scores.mean()) if sp_scores.size >= 10 else 0.0,
        speech_prob_p90=float(np.percentile(sp_scores, 90)) if sp_scores.size >= 10 else 0.0,
        reading_seconds=read_s,
        reading_segments=len(segs),
        reading_share_of_speech=read_s / max(speaking_s, 1.0),
        reading_share=read_s / max(len(valid), 1),
        top10_mean=float(top[:k].mean()) if len(top) else 0.0,
        p95=float(np.percentile(valid, 95)) if len(valid) else 0.0,
    )


def mmss(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"
