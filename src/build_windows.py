"""
build_windows.py — Stage 2: per-frame time series -> windowed behavioral features.

For every parquet in data/features/, this:
  1. Cleans the gaze signal (drop blink/no-face frames, interpolate short gaps,
     median filter) and resamples onto a uniform grid at cfg.target_fps.
  2. Slides a window (cfg.window_seconds, cfg.stride_seconds) over the video.
  3. Computes ~50 behavioral features per window (used for the reason tags) plus
     a fixed-length raw-channel sequence for the CNN.
  4. Writes:
       data/windows/windows.parquet        feature matrix + label + group(video)
       data/windows/sequences.npz          X_seq (n, L, C) for the 1D-CNN
       data/windows/_feature_columns.json   feature names + sequence metadata

Windows inherit their video's label (weak labels — video-level top-k aggregation
in Stage 3 absorbs the fact that not every window of a cheating video cheats).

Usage:
    python src/build_windows.py
    python src/build_windows.py --window 20 --stride 10
    python src/build_windows.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    Config, DEFAULT_CONFIG, FEATURES_DIR, WINDOWS_DIR, MODELS_DIR,
    BASE_CHANNELS, L2CS_CHANNELS, SEQ_CHANNELS,
    interpolate_gaps, median_filter_1d, resample_uniform,
    detect_saccades, reading_sweep_features, run_selftest,
)


# --------------------------------------------------------------------------- #
# Per-video cleaning -> uniform channel arrays
# --------------------------------------------------------------------------- #
def clean_video(df: pd.DataFrame, cfg: Config, has_l2cs: bool):
    """Return dict of uniform-grid arrays + dt, resampled at cfg.target_fps.

    Also attaches the speaking analysis from lip motion (visual VAD on `mouth`):
    `speaking` (per-sample 0/1 mask) and `_turns` (video-level answer turns).
    Only engineered window features use them; the CNN's sequence channels don't.
    """
    df = df.sort_values("frame_idx").reset_index(drop=True)
    t = df["t"].to_numpy(dtype=float)
    face = df["face_found"].to_numpy() if "face_found" in df else np.ones(len(df))
    blink = df["blink"].to_numpy() if "blink" in df else np.zeros(len(df))

    # gaze is valid only when a face is present AND the eye is open
    gaze_valid = (face > 0) & (blink == 0)
    pose_valid = face > 0

    out = {}
    for ch in BASE_CHANNELS:
        raw = df[ch].to_numpy(dtype=float) if ch in df else np.full(len(df), np.nan)
        valid = gaze_valid if ch in ("gaze_x", "gaze_y") else pose_valid
        filled = interpolate_gaps(raw, valid, _median_dt(t), cfg.max_gap_seconds)
        filled = median_filter_1d(filled, cfg.median_filter_k)
        _, uni, dt = resample_uniform(t, filled, cfg.target_fps)
        out[ch] = uni

    # blink + validity resampled by nearest (as rates)
    _, blink_u, dt = resample_uniform(t, blink.astype(float), cfg.target_fps)
    _, facev_u, _ = resample_uniform(t, pose_valid.astype(float), cfg.target_fps)
    out["blink"] = (blink_u > 0.5).astype(float)
    out["valid"] = facev_u

    if has_l2cs:
        for ch in L2CS_CHANNELS:
            raw = df[ch].to_numpy(dtype=float) if ch in df else np.full(len(df), np.nan)
            filled = interpolate_gaps(raw, pose_valid & np.isfinite(raw), _median_dt(t), cfg.max_gap_seconds)
            _, uni, _ = resample_uniform(t, filled, cfg.target_fps)
            out[ch] = uni

    # Per-candidate calibration: subtract each video's own median gaze so that a
    # person's neutral head/eye offset isn't mistaken for signal. Reading/overlay
    # behavior is a DEVIATION from each person's baseline, not an absolute position.
    if cfg.calibrate_gaze:
        for ch in ("gaze_x", "gaze_y"):
            med = np.nanmedian(out[ch])
            if np.isfinite(med):
                out[ch] = out[ch] - med

    # Speaking analysis from lip motion (visual VAD on the `mouth` channel).
    from speaking import visual_speaking
    n = len(out["gaze_x"])
    res = visual_speaking(out["mouth"], out["valid"] > 0.5, dt, cfg)
    out["speaking"] = res.speaking_mask.astype(float)
    out["_turns"] = res.turns
    out["_speak_source"] = res.source

    out["dt"] = dt
    out["duration"] = float(n * dt)
    return out


def _median_dt(t: np.ndarray) -> float:
    if len(t) < 2:
        return 1.0 / DEFAULT_CONFIG.target_fps
    d = np.diff(t)
    d = d[d > 0]
    return float(np.median(d)) if d.size else 1.0 / DEFAULT_CONFIG.target_fps


# --------------------------------------------------------------------------- #
# Window feature computation (~50 features)
# --------------------------------------------------------------------------- #
def _fill_window(a: np.ndarray) -> np.ndarray:
    """Fill remaining NaNs inside a window (ffill/bfill/mean) for the detectors."""
    a = a.astype(float).copy()
    if np.isfinite(a).all():
        return a
    if not np.isfinite(a).any():
        return np.zeros_like(a)
    idx = np.arange(len(a))
    fin = np.isfinite(a)
    a[~fin] = np.interp(idx[~fin], idx[fin], a[fin])
    return a


def compute_window_features(win: dict, cfg: Config, has_l2cs: bool) -> dict:
    dt = win["dt"]
    gx = _fill_window(win["gaze_x"])
    gy = _fill_window(win["gaze_y"])
    yaw = _fill_window(win["yaw"])
    pitch = _fill_window(win["pitch"])
    roll = _fill_window(win["roll"])
    ear = _fill_window(win["ear"])
    blink = win["blink"]
    valid = win["valid"]
    n = len(gx)
    dur = max(n * dt, 1e-6)

    f = {}
    f["valid_ratio"] = float(np.mean(valid))

    # one adaptive velocity threshold for this window, shared by all detectors
    from common import velocity_threshold
    vthr = velocity_threshold(gx, gy, dt, cfg)
    f["velocity_thresh"] = float(vthr)

    # ---- reading-sweep signature (6) ----
    f.update(reading_sweep_features(gx, dt, cfg, vel_thresh=vthr))

    # ---- saccade stats (rate, amp mean/std, dur mean, 8-bin dir hist, h:v ratio) ----
    sacc = detect_saccades(gx, gy, dt, cfg, vel_thresh=vthr)
    amps = np.array([abs(s.amplitude) for s in sacc]) if sacc else np.array([])
    durs = np.array([s.duration for s in sacc]) if sacc else np.array([])
    dirs = np.array([s.direction for s in sacc]) if sacc else np.array([])
    f["saccade_rate"] = 60.0 * len(sacc) / dur
    f["saccade_amp_mean"] = float(amps.mean()) if amps.size else 0.0
    f["saccade_amp_std"] = float(amps.std()) if amps.size else 0.0
    f["saccade_dur_mean"] = float(durs.mean()) if durs.size else 0.0
    hist = np.zeros(8)
    if dirs.size:
        bins = ((dirs + np.pi) / (2 * np.pi) * 8).astype(int) % 8
        for b in bins:
            hist[b] += 1
        hist = hist / hist.sum()
    for k in range(8):
        f[f"sacc_dir_bin{k}"] = float(hist[k])
    # horizontal:vertical saccade energy ratio
    if sacc:
        dxs = np.array([abs(np.cos(s.direction) * s.amplitude) for s in sacc])
        dys = np.array([abs(np.sin(s.direction) * s.amplitude) for s in sacc])
        f["sacc_hv_ratio"] = float(dxs.sum() / (dys.sum() + 1e-6))
    else:
        f["sacc_hv_ratio"] = 0.0
    # direction entropy (readers are strongly bimodal L/R -> low entropy)
    p = hist[hist > 0]
    f["sacc_dir_entropy"] = float(-(p * np.log2(p)).sum()) if p.size else 0.0

    # ---- fixation stats (count, mean duration, dispersion) ----
    fix_count, fix_dur_mean, fix_disp = _fixation_stats(gx, gy, dt, cfg, vthr)
    f["fixation_count"] = fix_count
    f["fixation_dur_mean"] = fix_dur_mean
    f["fixation_dispersion"] = fix_disp

    # ---- gaze distribution ----
    f["gaze_x_mean"] = float(np.mean(gx))
    f["gaze_y_mean"] = float(np.mean(gy))
    f["gaze_x_var"] = float(np.var(gx))
    f["gaze_y_var"] = float(np.var(gy))
    f["gaze_xy_var_ratio"] = float(np.var(gx) / (np.var(gy) + 1e-6))
    f["gaze_x_range"] = float(np.ptp(gx))
    f["gaze_y_range"] = float(np.ptp(gy))
    off = (np.abs(gx) > cfg.off_center_thresh) | (np.abs(gy) > cfg.off_center_thresh)
    f["pct_off_center"] = float(np.mean(off))
    speed = np.hypot(np.gradient(gx, dt), np.gradient(gy, dt))
    f["gaze_speed_mean"] = float(np.mean(speed))
    f["gaze_speed_std"] = float(np.std(speed))
    ent, ncells = _grid_entropy(gx, gy, cfg.grid_n)
    f["gaze_grid_entropy"] = ent
    f["gaze_grid_cells"] = ncells

    # ---- head-gaze coupling ----
    f["corr_yaw_gazex"] = _safe_corr(yaw, gx)
    f["corr_pitch_gazey"] = _safe_corr(pitch, gy)
    f["yaw_var"] = float(np.var(yaw))
    f["pitch_var"] = float(np.var(pitch))
    f["roll_var"] = float(np.var(roll))
    f["pct_head_averted"] = float(np.mean(np.abs(yaw) > cfg.head_averted_yaw_deg))

    # ---- blink / EAR (reading suppresses blinking) ----
    blink_events = int(np.sum(np.diff((blink > 0.5).astype(int)) == 1))
    f["blink_rate"] = 60.0 * blink_events / dur
    f["ear_mean"] = float(np.mean(ear))
    f["ear_std"] = float(np.std(ear))

    # ---- v2: speaking-conditioned contrasts ------------------------------- #
    # Within-video speaking/silent ratios self-normalize recording conditions
    # (fps, resolution, camera) — the confound that sank absolute features.
    # Core tell: reading WHILE speaking = reading the answer aloud (Cluely),
    # vs a clean candidate who reads silently and speaks facing the camera.
    f.update(_conditioned_features(win, gx, gy, yaw, blink, dt, cfg, vthr))

    # ---- optional appearance-gaze channels ----
    if has_l2cs:
        yl = _fill_window(win["gaze_yaw_l2cs"])
        pl = _fill_window(win["gaze_pitch_l2cs"])
        f["l2cs_yaw_mean"] = float(np.mean(yl))
        f["l2cs_pitch_mean"] = float(np.mean(pl))
        f["l2cs_yaw_var"] = float(np.var(yl))
        f["l2cs_pitch_var"] = float(np.var(pl))

    return f


def _conditioned_features(win: dict, gx, gy, yaw, blink, dt, cfg: Config, vthr) -> dict:
    """Speaking-vs-silent contrasts for one window.

    Ratios are (a+eps)/(b+eps); when either side has < min_side_s of samples the
    contrast is unmeasurable — return neutral values (ratio 1, diff 0) so the
    model leans on `speak_frac` to know the contrast was absent.
    """
    from common import detect_return_sweeps

    n = len(gx)
    out = dict(
        speak_frac=0.0,
        sweeps_pm_speak=0.0, sweeps_pm_silent=0.0, sweeps_speak_ratio=1.0,
        gazex_var_speak_ratio=1.0, off_center_speak_minus_silent=0.0,
        yaw_var_speak_ratio=1.0, blink_rate_speak_ratio=1.0,
        speak_gaze_modal_frac=0.0,
    )
    speaking = win.get("speaking")
    if speaking is None or n < 5:
        return out
    sp = np.asarray(speaking[:n]) > 0.5
    out["speak_frac"] = float(sp.mean())

    min_side = max(3, int(round(2.0 / dt)))  # need >=2s on each side to contrast
    n_sp, n_si = int(sp.sum()), int((~sp).sum())
    eps = 1e-6

    # Sweep events attributed by the time they fire.
    times, _ = detect_return_sweeps(gx, dt, cfg, vthr)
    idx = [min(n - 1, int(round(t / dt))) for t in times]
    sw_sp = sum(1 for i in idx if sp[i])
    sw_si = len(idx) - sw_sp
    dur_sp, dur_si = n_sp * dt, n_si * dt
    if dur_sp >= 2.0:
        out["sweeps_pm_speak"] = 60.0 * sw_sp / dur_sp
    if dur_si >= 2.0:
        out["sweeps_pm_silent"] = 60.0 * sw_si / dur_si

    if n_sp >= min_side and n_si >= min_side:
        out["sweeps_speak_ratio"] = (out["sweeps_pm_speak"] + eps) / (out["sweeps_pm_silent"] + eps)
        out["gazex_var_speak_ratio"] = float((np.var(gx[sp]) + eps) / (np.var(gx[~sp]) + eps))
        out["yaw_var_speak_ratio"] = float((np.var(yaw[sp]) + eps) / (np.var(yaw[~sp]) + eps))
        off = (np.abs(gx) > cfg.off_center_thresh) | (np.abs(gy) > cfg.off_center_thresh)
        out["off_center_speak_minus_silent"] = float(off[sp].mean() - off[~sp].mean())
        # blink onsets attributed by time
        onsets = np.where(np.diff((blink > 0.5).astype(int)) == 1)[0] + 1
        b_sp = sum(1 for i in onsets if sp[min(i, n - 1)])
        b_si = len(onsets) - b_sp
        out["blink_rate_speak_ratio"] = ((60.0 * b_sp / dur_sp) + eps) / ((60.0 * b_si / dur_si) + eps)

    # Speaking with gaze parked in one grid cell = reading a fixed overlay aloud.
    if n_sp >= min_side:
        xs = np.clip(((gx[sp] + 1) / 2 * cfg.grid_n).astype(int), 0, cfg.grid_n - 1)
        ys = np.clip(((gy[sp] + 1) / 2 * cfg.grid_n).astype(int), 0, cfg.grid_n - 1)
        cells = ys * cfg.grid_n + xs
        counts = np.bincount(cells, minlength=cfg.grid_n * cfg.grid_n)
        out["speak_gaze_modal_frac"] = float(counts.max() / max(counts.sum(), 1))

    # Clamp ratios: tiny denominators can explode; trees don't need the tail.
    for k in ("sweeps_speak_ratio", "gazex_var_speak_ratio",
              "yaw_var_speak_ratio", "blink_rate_speak_ratio"):
        out[k] = float(np.clip(out[k], 0.0, 50.0))
    return out


def _fixation_stats(gx, gy, dt, cfg: Config, vel_thresh=None):
    """Fixations = low-speed segments >= min_fixation_seconds between saccades."""
    from common import velocity_threshold
    if vel_thresh is None:
        vel_thresh = velocity_threshold(gx, gy, dt, cfg)
    speed = np.hypot(np.gradient(gx, dt), np.gradient(gy, dt))
    slow = speed <= vel_thresh
    min_len = max(1, int(round(cfg.min_fixation_seconds / dt)))
    durations, disps = [], []
    i, n = 0, len(gx)
    while i < n:
        if slow[i]:
            j = i
            while j < n and slow[j]:
                j += 1
            if (j - i) >= min_len:
                xs, ys = gx[i:j], gy[i:j]
                cx, cy = xs.mean(), ys.mean()
                disps.append(float(np.mean(np.hypot(xs - cx, ys - cy))))
                durations.append((j - i) * dt)
            i = j
        else:
            i += 1
    count = len(durations)
    return count, (float(np.mean(durations)) if count else 0.0), (float(np.mean(disps)) if count else 0.0)


def _grid_entropy(gx, gy, grid_n):
    """Occupancy entropy over a grid_n x grid_n grid on [-1,1]^2 + #cells used."""
    xs = np.clip(((gx + 1) / 2 * grid_n).astype(int), 0, grid_n - 1)
    ys = np.clip(((gy + 1) / 2 * grid_n).astype(int), 0, grid_n - 1)
    flat = ys * grid_n + xs
    counts = np.bincount(flat, minlength=grid_n * grid_n).astype(float)
    p = counts[counts > 0] / counts.sum()
    ent = float(-(p * np.log2(p)).sum()) if p.size else 0.0
    return ent, int((counts > 0).sum())


def _safe_corr(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def video_turn_features(clean: dict, cfg: Config) -> dict:
    """Video-level answer-turn structure (joined onto every window of the video).

    Turns come from lip motion, so gap_s is the silence before each answer burst
    (it includes the unheard question); gap_cv measures how flat that rhythm is.
    """
    turns = clean.get("_turns") or []
    dt = clean["dt"]
    dur_min = max(clean["duration"] / 60.0, 1e-6)
    out = dict(
        bursts_per_min=len(turns) / dur_min,
        burst_dur_mean=0.0, burst_dur_std=0.0,
        gap_dur_mean=0.0, gap_dur_cv=0.0,
        pause_rate=0.0, prespeech_gaze_shift=0.0,
    )
    if not turns:
        return out
    durs = np.array([t["duration_s"] for t in turns], dtype=float)
    gaps = np.array([t["gap_s"] for t in turns], dtype=float)
    out["burst_dur_mean"] = float(durs.mean())
    out["burst_dur_std"] = float(durs.std())
    # First gap is "since video start" — drop it from gap stats when possible.
    g = gaps[1:] if len(gaps) > 1 else gaps
    g = g[g > 0.05]
    if g.size:
        out["gap_dur_mean"] = float(g.mean())
        out["gap_dur_cv"] = float(g.std() / (g.mean() + 1e-6))  # LOW = lag-loop flat
        # mid-answer thinking pauses: short silences (<2s) between bursts
        speak_min = max(durs.sum() / 60.0, 1e-6)
        out["pause_rate"] = float((g < 2.0).sum() / speak_min)

    # Gaze flick toward a fixed region just before each answer starts.
    gx, gy = clean["gaze_x"], clean["gaze_y"]
    n = len(gx)
    w = max(2, int(round(cfg.prespeech_window_s / dt)))
    shifts = []
    for t in turns:
        i = int(round(t["onset_s"] / dt))
        pre0, pre1 = max(0, i - w), i                   # 1s before onset
        base0, base1 = max(0, i - 3 * w), max(0, i - w)  # the 2s before that
        if pre1 - pre0 >= 2 and base1 - base0 >= 2 and pre1 <= n:
            dx = np.nanmean(gx[pre0:pre1]) - np.nanmean(gx[base0:base1])
            dy = np.nanmean(gy[pre0:pre1]) - np.nanmean(gy[base0:base1])
            if np.isfinite(dx) and np.isfinite(dy):
                shifts.append(float(np.hypot(dx, dy)))
    if shifts:
        out["prespeech_gaze_shift"] = float(np.mean(shifts))
    return out


def _window_sequence(win: dict, start: int, end: int, seq_len: int) -> np.ndarray:
    """Resample SEQ_CHANNELS in [start:end] to a fixed length -> (seq_len, C)."""
    chans = []
    src_idx = np.linspace(0, end - start - 1, seq_len)
    for ch in SEQ_CHANNELS:
        a = _fill_window(win[ch][start:end])
        chans.append(np.interp(src_idx, np.arange(end - start), a))
    return np.stack(chans, axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def build(cfg: Config):
    parquets = sorted(FEATURES_DIR.glob("*.parquet"))
    if not parquets:
        print(f"No feature parquets in {FEATURES_DIR}. Run extract_features.py first.")
        return 1

    win_len = int(round(cfg.window_seconds * cfg.target_fps))
    stride = max(1, int(round(cfg.stride_seconds * cfg.target_fps)))
    seq_len = int(round(cfg.window_seconds * cfg.seq_resample_fps))

    rows, seqs = [], []
    n_videos = 0
    for pq in parquets:
        if pq.name.startswith("_"):
            continue
        parts = pq.stem.split("__")
        label_str = parts[0]
        if label_str not in ("cheating", "clean"):
            # ignore any foreign parquet (e.g. a stray prediction-cache file)
            print(f"  skip (unknown label prefix '{label_str}'): {pq.name}")
            continue
        source = parts[1] if len(parts) >= 3 else "default"
        label = 1 if label_str == "cheating" else 0
        df = pd.read_parquet(pq)
        has_l2cs = all(c in df.columns for c in L2CS_CHANNELS)

        win = clean_video(df, cfg, has_l2cs)
        n = len(win["gaze_x"])
        if n < win_len:
            print(f"  skip (too short): {pq.name} ({n} < {win_len} samples)")
            continue
        n_videos += 1
        group = pq.stem  # video id -> CV group
        turn_feats = video_turn_features(win, cfg)

        for start in range(0, n - win_len + 1, stride):
            end = start + win_len
            sub = {k: (v[start:end] if isinstance(v, np.ndarray) else v) for k, v in win.items()}
            sub["dt"] = win["dt"]
            feats = compute_window_features(sub, cfg, has_l2cs)
            if feats["valid_ratio"] < cfg.min_valid_ratio:
                continue
            feats.update(turn_feats)
            feats.update(dict(
                video=pq.stem, group=group, label=label, label_str=label_str,
                source=source,
                win_start_s=round(start * win["dt"], 2), win_end_s=round(end * win["dt"], 2),
            ))
            rows.append(feats)
            seqs.append(_window_sequence(win, start, end, seq_len))

    if not rows:
        print("No valid windows produced (check valid_ratio / video length).")
        return 1

    windows = pd.DataFrame(rows)
    meta_cols = ["video", "group", "label", "label_str", "source",
                 "win_start_s", "win_end_s"]
    feature_cols = [c for c in windows.columns if c not in meta_cols]
    windows = windows[meta_cols + feature_cols]

    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(WINDOWS_DIR / "windows.parquet", index=False)
    X_seq = np.stack(seqs, axis=0)
    np.savez_compressed(
        WINDOWS_DIR / "sequences.npz",
        X=X_seq,
        label=windows["label"].to_numpy(),
        group=windows["group"].to_numpy(),
        source=windows["source"].to_numpy(),
        channels=np.array(SEQ_CHANNELS),
    )
    (WINDOWS_DIR / "_feature_columns.json").write_text(json.dumps({
        "feature_cols": feature_cols,
        "seq_channels": SEQ_CHANNELS,
        "seq_len": seq_len,
        "window_seconds": cfg.window_seconds,
        "stride_seconds": cfg.stride_seconds,
        "has_l2cs": bool(all(c in feature_cols for c in ["l2cs_yaw_mean"])),
    }, indent=2))

    n_pos = int(windows["label"].sum())
    print(f"\n=== Stage 2 summary ===")
    print(f"videos: {n_videos}  windows: {len(windows)}  "
          f"(cheating={n_pos}, clean={len(windows)-n_pos})")
    print(f"features/window: {len(feature_cols)}   sequence: {X_seq.shape}")
    print(f"-> {WINDOWS_DIR / 'windows.parquet'}")
    print(f"-> {WINDOWS_DIR / 'sequences.npz'}")
    # class/group/source sanity for CV
    print(f"videos/class: cheating={windows[windows.label==1]['group'].nunique()}, "
          f"clean={windows[windows.label==0]['group'].nunique()}")
    src_tbl = windows.groupby(["label_str", "source"])["group"].nunique()
    print("videos/source:")
    for (lab, src), cnt in src_tbl.items():
        print(f"    {lab:8s} {src:14s} {cnt}")
    return 0


def _selftest(cfg: Config) -> bool:
    """Signal-level selftest: feed a synthetic reading window and a random-walk
    window through the real feature computer and assert they separate."""
    from common import synth_reading_signal, synth_random_walk
    ok = run_selftest()

    def make_win(t, gx, gy, dt):
        n = len(gx)
        return dict(gaze_x=gx, gaze_y=gy, yaw=np.zeros(n), pitch=np.zeros(n),
                    roll=np.zeros(n), ear=np.full(n, 0.3), blink=np.zeros(n),
                    valid=np.ones(n), dt=dt, duration=n * dt)

    t, gx, gy, dt = synth_reading_signal(duration_s=cfg.window_seconds, seed=0)
    reading = compute_window_features(make_win(t, gx, gy, dt), cfg, has_l2cs=False)
    t2, gx2, gy2, dt2 = synth_random_walk(duration_s=cfg.window_seconds, seed=1)
    walk = compute_window_features(make_win(t2, gx2, gy2, dt2), cfg, has_l2cs=False)

    checks = [
        ("reading lines_per_minute > walk",
         reading["lines_per_minute"] > walk["lines_per_minute"] + 5),
        ("reading sacc_hv_ratio high (horizontal)", reading["sacc_hv_ratio"] > 3.0),
        ("reading sweep_regularity > walk", reading["sweep_regularity"] >= walk["sweep_regularity"]),
        ("reading autocorr_peak >= 0.4", reading["autocorr_peak"] >= 0.4),
    ]

    # v2: conditioned features — a window whose SPEAKING half contains the
    # reading pattern must show sweeps_pm_speak >> sweeps_pm_silent.
    n = len(gx)
    half = n // 2
    gx_mix = np.concatenate([gx[:half], gx2[:n - half]])   # read then wander
    gy_mix = np.concatenate([gy[:half], gy2[:n - half]])
    win_mix = make_win(t, gx_mix, gy_mix, dt)
    speaking = np.zeros(n)
    speaking[:half] = 1.0                                   # speaking while reading
    win_mix["speaking"] = speaking
    mix = compute_window_features(win_mix, cfg, has_l2cs=False)
    checks += [
        ("speak_frac ~ 0.5", abs(mix["speak_frac"] - 0.5) < 0.05),
        ("sweeps_pm_speak > sweeps_pm_silent",
         mix["sweeps_pm_speak"] > mix["sweeps_pm_silent"] + 5),
        ("sweeps_speak_ratio > 2", mix["sweeps_speak_ratio"] > 2.0),
    ]

    print("\n[selftest] window features:")
    print("  reading lpm=%.1f hv=%.1f reg=%.2f ac=%.2f | walk lpm=%.1f hv=%.1f" % (
        reading["lines_per_minute"], reading["sacc_hv_ratio"], reading["sweep_regularity"],
        reading["autocorr_peak"], walk["lines_per_minute"], walk["sacc_hv_ratio"]))
    print("  mix speak_frac=%.2f sweeps_speak=%.1f sweeps_silent=%.1f ratio=%.1f" % (
        mix["speak_frac"], mix["sweeps_pm_speak"], mix["sweeps_pm_silent"],
        mix["sweeps_speak_ratio"]))
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("[selftest]", "ALL PASS ✅" if ok else "FAILURES ❌")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 2: windowed behavioral features.")
    ap.add_argument("--window", type=float, default=None, help="window length seconds")
    ap.add_argument("--stride", type=float, default=None, help="window stride seconds")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    cfg = DEFAULT_CONFIG
    cfg_path = MODELS_DIR / "config.json"
    if cfg_path.exists():
        cfg = Config.from_json(cfg_path)
    if args.window:
        cfg.window_seconds = args.window
    if args.stride:
        cfg.stride_seconds = args.stride

    if args.selftest:
        return 0 if _selftest(cfg) else 1

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.to_json(cfg_path)
    return build(cfg)


if __name__ == "__main__":
    sys.exit(main())
