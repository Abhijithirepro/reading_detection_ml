"""
build_windows.py (v2) — Stage 2: per-frame series -> short READING windows.

v1 cut 20 s windows every 10 s and gave every window of a cheating video the label
"cheating", so its CNN learned *who* cheats, not *when* they read (per-second AUC
0.48-0.71 against the manual reviews). v2:

  * 8 s windows every 1 s -> one reading probability per second of video.
  * STRONG labels from the manual reviews (labels.py): a window of a reviewed video
    is reading (1) when >= label_pos_frac of it lies inside a reviewer's span, not
    reading (0) when <= label_neg_frac does, and ignored in between. Every window
    of a clean video is 0. Windows of unreviewed cheating videos get -1 (unknown):
    they are never trained on, which makes them an honest video-level test set.
  * Features describe the EYES and the MOUTH together, because the behaviour the
    reviewers mark is "reading and answering from the screen":
      - speaking: fused lip-gap + jawOpen motion (speaking.visual_speaking_v2)
      - horizontal eye sawtooth: slow line-following drift + fast return, counted
        in the video's own dominant polarity (mirroring-proof), velocity skew,
        drift fraction, line-rhythm autocorrelation
      - where the eyes are: vertical gaze vs the candidate's own LISTENING baseline,
        gaze parked in a small region vs looking away (thinking aversion)
      - head stillness, eye-head decoupling, blinking, squint, leaning in
      - each key feature also relative to the video's own median (_vrel), so a
        person's baseline (camera, face, habits) is not mistaken for behaviour

Outputs (all under v2/data/windows/):
    windows.parquet   one row per window: meta + label + ~90 features
    videos.parquet    one row per video: t0, duration, polarity, speaking share
    _feature_columns.json

Usage:
    python src/build_windows.py                 # all parquets in v2/data/features
    python src/build_windows.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    Config, DEFAULT_CONFIG, FEATURES_DIR, WINDOWS_DIR, MODELS_DIR,
    BASE_CHANNELS, BLENDSHAPE_CHANNELS, L2CS_CHANNELS,
    interpolate_gaps, median_filter_1d, resample_uniform, detect_return_sweeps,
)
import labels as lab

BS_BLINK_THRESHOLD = 0.40     # eyeBlink blendshape score that counts as eyes closed

# Features that also get a video-relative copy (feature - the video's median).
VREL_FEATURES = [
    "h_std", "h_range", "h_sweeps_dom", "h_sweeps_max", "h_ac_peak", "h_skew_dom",
    "gx_std", "v_mean_listen", "gy_mean_listen", "v_std", "disp", "aversion_frac",
    "parked_frac", "yaw_std", "pitch_std", "head_speed", "blink_rate", "jaw_act",
    "mouth_act", "squint_mean", "brow_down_mean", "face_scale_rel", "pitch_mean_listen",
    "ear_mean", "big_jump_rate", "h_prog_frac",
]


# --------------------------------------------------------------------------- #
# Per-video cleaning -> uniform channel arrays
# --------------------------------------------------------------------------- #
def _median_dt(t: np.ndarray) -> float:
    if len(t) < 2:
        return 1.0 / DEFAULT_CONFIG.target_fps
    d = np.diff(t)
    d = d[d > 0]
    return float(np.median(d)) if d.size else 1.0 / DEFAULT_CONFIG.target_fps


def _col(df, name):
    return df[name].to_numpy(dtype=float) if name in df else np.full(len(df), np.nan)


def clean_video(df: pd.DataFrame, cfg: Config, has_l2cs: bool = False):
    """Uniform-grid arrays at cfg.target_fps + speaking analysis.

    Keys: every BASE channel, derived blendshape channels (eye_h, eye_v, jaw, smile,
    squint, brow_down, face_scale), blink, valid, speaking, t0 (video time of
    sample 0), dt, duration. Grid sample i sits at video time t0 + i*dt.
    """
    df = df.sort_values("frame_idx").reset_index(drop=True)
    t = df["t"].to_numpy(dtype=float)
    face = df["face_found"].to_numpy() if "face_found" in df else np.ones(len(df))
    has_bs = all(c in df.columns for c in BLENDSHAPE_CHANNELS)

    raw = {ch: _col(df, ch) for ch in BASE_CHANNELS}
    if has_bs:
        L, R = "Left", "Right"
        raw["eye_h"] = ((_col(df, "eyeLookOut" + L) + _col(df, "eyeLookIn" + R))
                        - (_col(df, "eyeLookIn" + L) + _col(df, "eyeLookOut" + R))) / 2
        raw["eye_v"] = ((_col(df, "eyeLookUp" + L) + _col(df, "eyeLookUp" + R))
                        - (_col(df, "eyeLookDown" + L) + _col(df, "eyeLookDown" + R))) / 2
        raw["jaw"] = _col(df, "jawOpen")
        raw["smile"] = (_col(df, "mouthSmile" + L) + _col(df, "mouthSmile" + R)) / 2
        raw["squint"] = (_col(df, "eyeSquint" + L) + _col(df, "eyeSquint" + R)) / 2
        raw["brow_down"] = (_col(df, "browDown" + L) + _col(df, "browDown" + R)) / 2
        raw["face_scale"] = _col(df, "face_scale")
        blink_score = (_col(df, "eyeBlink" + L) + _col(df, "eyeBlink" + R)) / 2
        blink = ((blink_score > BS_BLINK_THRESHOLD) & (face > 0)).astype(float)
    else:
        blink = df["blink"].to_numpy(dtype=float) if "blink" in df else np.zeros(len(df))

    gaze_valid = (face > 0) & (blink < 0.5)
    pose_valid = face > 0
    eye_chs = {"gaze_x", "gaze_y", "eye_h", "eye_v"}
    mdt = _median_dt(t)

    out = {}
    for ch, x in raw.items():
        valid = (gaze_valid if ch in eye_chs else pose_valid) & np.isfinite(x)
        filled = interpolate_gaps(x, valid, mdt, cfg.max_gap_seconds)
        filled = median_filter_1d(filled, cfg.median_filter_k)
        _, uni, dt = resample_uniform(t, filled, cfg.target_fps)
        out[ch] = uni
    _, blink_u, dt = resample_uniform(t, blink.astype(float), cfg.target_fps)
    _, facev_u, _ = resample_uniform(t, pose_valid.astype(float), cfg.target_fps)
    out["blink"] = (blink_u > 0.5).astype(float)
    out["valid"] = facev_u

    # Per-candidate calibration (as v1): eye position is a deviation from the
    # person's own median, not an absolute number.
    if cfg.calibrate_gaze:
        for ch in eye_chs & set(out):
            med = np.nanmedian(out[ch])
            if np.isfinite(med):
                out[ch] = out[ch] - med

    from speaking import visual_speaking_v2
    n = len(out["gaze_x"])
    res = visual_speaking_v2(out["mouth"], out.get("jaw"), out["valid"] > 0.5, dt, cfg)
    out["speaking"] = res.speaking_mask.astype(float)
    out["speak_act"] = res.activity
    out["_turns"] = res.turns
    out["_speak_source"] = res.source
    out["has_blendshapes"] = has_bs
    out["t0"] = float(t[0]) if len(t) else 0.0
    out["dt"] = dt
    out["duration"] = float(n * dt)
    return out


# --------------------------------------------------------------------------- #
# Window features (vectorised over all windows of one video)
# --------------------------------------------------------------------------- #
def _fill(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).copy()
    fin = np.isfinite(a)
    if fin.all():
        return a
    if not fin.any():
        return np.zeros_like(a)
    idx = np.arange(len(a))
    a[~fin] = np.interp(idx[~fin], idx[fin], a[fin])
    return a


def _roll(x, W, fn, **kw):
    """Trailing rolling stat: value at i summarises x[i-W+1 .. i]."""
    r = getattr(pd.Series(x).rolling(W, min_periods=max(2, W // 2)), fn)(**kw)
    return r.to_numpy()


def _robust_scale(x):
    x = x[np.isfinite(x)]
    if x.size < 10:
        return 1.0
    return float(max(1.4826 * np.median(np.abs(x - np.median(x))), 1e-4))


def _ac_peak(x, dt, lo_s=1.5, hi_s=8.0):
    """Peak normalised autocorrelation within the reading-line cadence band."""
    x = x - x.mean()
    var = float(np.dot(x, x))
    n = len(x)
    if var < 1e-12 or n < 8:
        return 0.0
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n] / var
    lo, hi = max(1, int(lo_s / dt)), min(n - 1, int(hi_s / dt))
    return float(ac[lo:hi + 1].max()) if hi > lo else 0.0


def video_polarity(h: np.ndarray, speaking: np.ndarray, dt: float, cfg: Config) -> int:
    """Direction (+1/-1) of the fast return jump in eye_h for THIS video, decided
    from its speaking stretches (reading aloud is where the sawtooth lives)."""
    vthr = _sweep_vthr(h, dt)
    W = int(round(cfg.read_window_seconds / dt))
    counts = {+1: 0, -1: 0}
    for s in range(0, len(h) - W + 1, W):
        if speaking[s:s + W].mean() < 0.5:
            continue
        seg = h[s:s + W]
        for d in (+1, -1):
            counts[d] += len(detect_return_sweeps(seg, dt, cfg, vthr, d)[0])
    return +1 if counts[+1] >= counts[-1] else -1


def _sweep_vthr(h, dt):
    v = np.abs(np.gradient(h, dt))
    v = v[np.isfinite(v)]
    return float(np.median(v) + 2.5 * 1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else 1.0


def window_features(c: dict, cfg: Config):
    """All 8 s / 1 s-stride windows of one cleaned video -> (DataFrame, starts)."""
    dt = c["dt"]
    n = len(c["gaze_x"])
    W = int(round(cfg.read_window_seconds / dt))
    S = max(1, int(round(cfg.read_stride_seconds / dt)))
    if n < W:
        return pd.DataFrame(), np.array([], int)
    starts = np.arange(0, n - W + 1, S)
    ends = starts + W
    at = ends - 1                       # trailing-roll index that covers [start, end)
    has_bs = c.get("has_blendshapes", False)

    h = _fill(c["eye_h"] if has_bs else c["gaze_x"])
    v = _fill(c["eye_v"] if has_bs else c["gaze_y"])
    gx, gy = _fill(c["gaze_x"]), _fill(c["gaze_y"])
    yaw, pitch, roll = _fill(c["yaw"]), _fill(c["pitch"]), _fill(c["roll"])
    ear, mouth = _fill(c["ear"]), _fill(c["mouth"])
    sp = c["speaking"].astype(float)
    blink = c["blink"].astype(float)
    valid = c["valid"].astype(float)
    f = {}

    # ---- speaking ----
    f["speak_frac"] = _roll(sp, W, "mean")[at]
    one_s = max(2, int(round(1.0 / dt)))
    mouth_act = pd.Series(mouth).rolling(one_s, min_periods=2, center=True).std().fillna(0).to_numpy()
    f["mouth_act"] = _roll(mouth_act, W, "mean")[at]
    f["speak_act"] = _roll(_fill(c["speak_act"]), W, "mean")[at]
    if has_bs:
        jaw = _fill(c["jaw"])
        jaw_act = pd.Series(jaw).rolling(one_s, min_periods=2, center=True).std().fillna(0).to_numpy()
        f["jaw_act"] = _roll(jaw_act, W, "mean")[at]
        f["jaw_mean"] = _roll(jaw, W, "mean")[at]
        f["smile_mean"] = _roll(_fill(c["smile"]), W, "mean")[at]
    # speaking context: was the candidate silent just before (question) / talking around?
    ctx = int(round(30.0 / dt))
    sp_ctx = pd.Series(sp).rolling(ctx, min_periods=1, center=True).mean().to_numpy()
    f["speak_ctx30"] = sp_ctx[np.clip(starts + W // 2, 0, n - 1)]
    pre = int(round(10.0 / dt))
    sp_pre = _roll(sp, pre, "mean")
    f["speak_prev10"] = np.where(starts > 0, sp_pre[np.clip(starts - 1, 0, n - 1)], np.nan)

    # ---- horizontal eye movement: the reading sawtooth ----
    hs = median_filter_1d(h, 3)
    vel = np.gradient(pd.Series(hs).rolling(3, min_periods=1, center=True).mean().to_numpy(), dt)
    pol = video_polarity(hs, sp, dt, cfg)
    vthr = _sweep_vthr(hs, dt)
    f["h_std"] = _roll(hs, W, "std")[at]
    f["h_range"] = (_roll(hs, W, "quantile", q=0.95) - _roll(hs, W, "quantile", q=0.05))[at]
    f["gx_std"] = _roll(gx, W, "std")[at]
    skew = _roll(vel, W, "skew")[at]
    f["h_skew_dom"] = skew * pol            # >0 = fast jumps in this video's return direction
    f["h_skew_abs"] = np.abs(skew)
    vslow = pd.Series(vel).rolling(max(2, int(0.4 / dt)), min_periods=1, center=True).mean().to_numpy()
    drift = (vslow * (-pol) > 0.05 * vthr).astype(float)    # slow line-following drift
    f["h_prog_frac"] = _roll(drift, W, "mean")[at]
    dom, mx, ac = np.zeros(len(starts)), np.zeros(len(starts)), np.zeros(len(starts))
    gx_dom = np.zeros(len(starts))
    gthr = _sweep_vthr(gx, dt)
    for k, (s, e) in enumerate(zip(starts, ends)):
        seg = hs[s:e]
        a = len(detect_return_sweeps(seg, dt, cfg, vthr, pol)[0])
        b = len(detect_return_sweeps(seg, dt, cfg, vthr, -pol)[0])
        dom[k], mx[k] = a, max(a, b)
        ac[k] = _ac_peak(seg, dt)
        gx_dom[k] = len(detect_return_sweeps(gx[s:e], dt, cfg, gthr, pol)[0])
    per_min = 60.0 / cfg.read_window_seconds
    f["h_sweeps_dom"] = dom * per_min
    f["h_sweeps_max"] = mx * per_min
    f["gx_sweeps_dom"] = gx_dom * per_min
    f["h_ac_peak"] = ac

    # ---- where the eyes are: listening baseline, parking, aversion ----
    listen = (sp < 0.5) & (valid > 0.5)
    if listen.sum() < 25:
        listen = valid > 0.5
    base_v = float(np.median(v[listen])) if listen.any() else 0.0
    base_gy = float(np.median(gy[listen])) if listen.any() else 0.0
    base_pitch = float(np.median(pitch[listen])) if listen.any() else 0.0
    f["v_mean_listen"] = _roll(v - base_v, W, "mean")[at]
    f["gy_mean_listen"] = _roll(gy - base_gy, W, "mean")[at]
    f["pitch_mean_listen"] = _roll(pitch - base_pitch, W, "mean")[at]
    f["v_std"] = _roll(v, W, "std")[at]
    sh, sv = _robust_scale(h), _robust_scale(v)
    hz, vz = (h - np.median(h)) / sh, (v - np.median(v)) / sv
    f["disp"] = np.sqrt(_roll(hz, W, "var")[at] + _roll(vz, W, "var")[at])
    f["aversion_frac"] = _roll(((np.abs(hz) > 3) | (np.abs(vz) > 3)).astype(float), W, "mean")[at]
    # parked: share of samples within 1.5 robust-sd of the window's own median
    mh = _roll(hz, W, "median")[at]
    mv = _roll(vz, W, "median")[at]
    parked = np.zeros(len(starts))
    for k, (s, e) in enumerate(zip(starts, ends)):
        parked[k] = np.mean(np.hypot(hz[s:e] - mh[k], vz[s:e] - mv[k]) < 1.5)
    f["parked_frac"] = parked
    # big gaze jumps: displacement over 0.2 s (a saccade spans several 25 fps samples)
    lag = max(1, int(round(0.2 / dt)))
    dh, dv = np.zeros(n), np.zeros(n)
    dh[lag:], dv[lag:] = hz[lag:] - hz[:-lag], vz[lag:] - vz[:-lag]
    jump = np.hypot(dh, dv)
    big = (jump > 3) & (np.roll(jump, 1) <= 3)          # count onsets, not samples
    f["big_jump_rate"] = _roll(big.astype(float), W, "sum")[at] * per_min

    # ---- head ----
    f["yaw_std"] = _roll(yaw, W, "std")[at]
    f["pitch_std"] = _roll(pitch, W, "std")[at]
    f["roll_std"] = _roll(roll, W, "std")[at]
    hsp = np.abs(np.gradient(yaw, dt)) + np.abs(np.gradient(pitch, dt))
    f["head_speed"] = _roll(hsp, W, "mean")[at]
    corr = pd.Series(hs).rolling(W, min_periods=W // 2).corr(pd.Series(yaw)).to_numpy()[at]
    f["corr_h_yaw"] = np.nan_to_num(corr)

    # ---- blinking / eye opening / effort ----
    onsets = (np.diff(blink, prepend=0) > 0.5).astype(float)
    f["blink_rate"] = _roll(onsets, W, "sum")[at] * per_min
    f["ear_mean"] = _roll(ear, W, "mean")[at] - np.nanmedian(ear)
    if has_bs:
        f["squint_mean"] = _roll(_fill(c["squint"]), W, "mean")[at]
        f["brow_down_mean"] = _roll(_fill(c["brow_down"]), W, "mean")[at]
        fs = _fill(c["face_scale"])
        f["face_scale_rel"] = _roll(fs, W, "mean")[at] / max(np.nanmedian(fs), 1e-6) - 1
    f["valid_ratio"] = _roll(valid, W, "mean")[at]

    # ---- answer context (group "ctx"): +-15 s, SPEAKING samples only ----
    # Reviewers judge whole answers; an 8 s window is a noisy slice of one. These
    # describe how the eyes/head behave while the candidate talks around this
    # moment, and contrast it with the same person's listening baseline.
    f.update(_context_features(c, cfg, starts, W, dict(
        hs=hs, v=v - base_v, aversion=((np.abs(hz) > 3) | (np.abs(vz) > 3)).astype(float),
        onsets=onsets, head_speed=hsp, jump=big.astype(float)), f["h_sweeps_dom"]))

    feats = pd.DataFrame(f)
    # video-relative copies
    for k in VREL_FEATURES:
        if k in feats:
            feats[k + "_vrel"] = feats[k] - np.nanmedian(feats[k])
    return feats, starts


CTX_SECONDS = 30.0


def _context_features(c: dict, cfg: Config, starts, W, sig: dict, sweeps_dom) -> dict:
    dt = c["dt"]
    n = len(c["gaze_x"])
    sp = c["speaking"] > 0.5
    lis = (~sp) & (c["valid"] > 0.5)
    Wc = int(round(CTX_SECONDS / dt))
    need = max(2, int(round(2.0 / dt)))           # >= 2 s of speech in the context
    centers = np.clip(starts + W // 2, 0, n - 1)

    def ctx(x, fn, mask=sp):
        s = pd.Series(np.where(mask, x, np.nan))
        return getattr(s.rolling(Wc, center=True, min_periods=need), fn)().to_numpy()[centers]

    def listen_level(x, fn):
        xs = x[lis]
        if xs.size < need:
            return np.nan
        return float(np.std(xs) if fn == "std" else np.mean(xs))

    per_min = 60.0 / dt
    out = {
        "ctx_speak_frac": pd.Series(sp.astype(float)).rolling(Wc, center=True, min_periods=1)
                            .mean().to_numpy()[centers],
        "ctx_h_std_speak": ctx(sig["hs"], "std"),
        "ctx_v_std_speak": ctx(sig["v"], "std"),
        "ctx_v_mean_speak": ctx(sig["v"], "mean"),
        "ctx_aversion_speak": ctx(sig["aversion"], "mean"),
        "ctx_blink_speak": ctx(sig["onsets"], "mean") * per_min,
        "ctx_head_speed_speak": ctx(sig["head_speed"], "mean"),
        "ctx_jump_speak": ctx(sig["jump"], "mean") * per_min,
    }
    # same person's listening baseline -> speaking-minus-listening contrasts
    out["ctx_h_std_vs_listen"] = out["ctx_h_std_speak"] - listen_level(sig["hs"], "std")
    out["ctx_v_std_vs_listen"] = out["ctx_v_std_speak"] - listen_level(sig["v"], "std")
    out["ctx_aversion_vs_listen"] = out["ctx_aversion_speak"] - listen_level(sig["aversion"], "mean")
    out["ctx_blink_vs_listen"] = out["ctx_blink_speak"] - listen_level(sig["onsets"], "mean") * per_min
    out["ctx_head_vs_listen"] = out["ctx_head_speed_speak"] - listen_level(sig["head_speed"], "mean")

    # sweep rate while speaking, over the context (window-level counts, 1 s stride)
    sw = pd.Series(np.asarray(sweeps_dom, dtype=float))
    spw = pd.Series(sp[centers].astype(float))
    k = int(round(CTX_SECONDS / cfg.read_stride_seconds))
    num = (sw * spw).rolling(k, center=True, min_periods=1).sum()
    den = spw.rolling(k, center=True, min_periods=1).sum()
    out["ctx_sweeps_speak"] = (num / den.where(den > 0)).to_numpy()

    # speech rhythm: short pauses inside speech (reading stalls / waiting for text)
    edges = np.diff(sp.astype(int), prepend=0)
    off_i, on_i = np.where(edges == -1)[0], np.where(edges == 1)[0]
    pause = np.zeros(n)
    for o in off_i:
        nxt = on_i[on_i > o]
        if nxt.size and 0.3 <= (nxt[0] - o) * dt <= 2.5:
            pause[o] = 1.0
    speak_min = pd.Series(sp.astype(float)).rolling(Wc, center=True, min_periods=1).sum().to_numpy()[centers] * dt / 60
    pauses = pd.Series(pause).rolling(Wc, center=True, min_periods=1).sum().to_numpy()[centers]
    out["ctx_pause_rate"] = np.where(speak_min > 2.0 / 60, pauses / np.maximum(speak_min, 1e-6), np.nan)
    act = _fill(c["speak_act"])
    m = ctx(act, "mean")
    out["ctx_artic_cv"] = ctx(act, "std") / np.where(m > 1e-6, m, np.nan)
    return out


# --------------------------------------------------------------------------- #
# One video -> labelled window table
# --------------------------------------------------------------------------- #
def parse_name(stem: str):
    parts = stem.split("__")
    label_str = parts[0]
    source = parts[1] if len(parts) >= 3 else "default"
    return label_str, source


def video_windows(pq: Path, cfg: Config, reviews: dict):
    df = pd.read_parquet(pq)
    c = clean_video(df, cfg)
    feats, starts = window_features(c, cfg)
    if feats.empty:
        return None, None
    label_str, source = parse_name(pq.stem)
    dt, t0 = c["dt"], c["t0"]
    W = int(round(cfg.read_window_seconds / dt))
    ws = t0 + starts * dt
    we = ws + W * dt
    review = lab.review_for(pq.stem, reviews) if label_str == "cheating" else None

    if label_str == "clean":
        frac = np.zeros(len(ws))
        wlabel = np.zeros(len(ws), int)
    elif review is not None:
        frac = np.array([lab.window_label_fraction(review.segments, a, b) for a, b in zip(ws, we)])
        wlabel = np.where(frac >= cfg.label_pos_frac, 1, np.where(frac <= cfg.label_neg_frac, 0, -2))
    else:
        frac = np.full(len(ws), np.nan)
        wlabel = np.full(len(ws), -1)

    meta = pd.DataFrame(dict(
        video=pq.stem, label_str=label_str, label=int(label_str == "cheating"), source=source,
        reviewed=review is not None, win_start_s=np.round(ws, 3), win_end_s=np.round(we, 3),
        label_frac=frac, wlabel=wlabel))
    vinfo = dict(video=pq.stem, label_str=label_str, source=source, reviewed=review is not None,
                 t0=t0, dt=dt, duration=c["duration"], n_windows=len(ws),
                 speak_share=float(np.mean(c["speaking"])),
                 face_ratio=float(df["face_found"].mean()), has_blendshapes=c["has_blendshapes"])
    return pd.concat([meta, feats], axis=1), vinfo


def _worker(job):
    pq, cfg_dict, reviews = job
    cfg = Config(**cfg_dict)
    try:
        return video_windows(Path(pq), cfg, reviews)
    except Exception as e:  # keep the batch going; report at the end
        return None, dict(video=Path(pq).stem, error=f"{type(e).__name__}: {e}")


def build(cfg: Config, workers: int = None, features_dir: Path = FEATURES_DIR):
    parquets = [p for p in sorted(features_dir.glob("*.parquet"))
                if not p.name.startswith("_") and parse_name(p.stem)[0] in ("cheating", "clean")]
    if not parquets:
        print(f"No feature parquets in {features_dir}. Run extract_features.py first.")
        return 1
    reviews = lab.load_reviews()
    from dataclasses import asdict
    jobs = [(str(p), asdict(cfg), reviews) for p in parquets]
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    tables, vinfos, errors = [], [], []
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (tbl, vinfo) in enumerate(ex.map(_worker, jobs), 1):
            if tbl is None:
                if vinfo and "error" in vinfo:
                    errors.append(vinfo)
                continue
            tables.append(tbl)
            vinfos.append(vinfo)
            print(f"[{i}/{len(jobs)}] {vinfo['video'][:70]:70s} windows={vinfo['n_windows']}"
                  f"{'  REVIEWED' if vinfo['reviewed'] else ''}", flush=True)
    if not tables:
        print("No windows produced.")
        return 1
    windows = pd.concat(tables, ignore_index=True)
    meta_cols = ["video", "label_str", "label", "source", "reviewed", "win_start_s",
                 "win_end_s", "label_frac", "wlabel"]
    feature_cols = [c for c in windows.columns if c not in meta_cols]
    float_cols = windows[feature_cols].select_dtypes("float64").columns
    windows[float_cols] = windows[float_cols].astype(np.float32)
    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(WINDOWS_DIR / "windows.parquet", index=False)
    pd.DataFrame(vinfos).to_parquet(WINDOWS_DIR / "videos.parquet", index=False)
    (WINDOWS_DIR / "_feature_columns.json").write_text(json.dumps(dict(
        feature_cols=feature_cols, read_window_seconds=cfg.read_window_seconds,
        read_stride_seconds=cfg.read_stride_seconds, target_fps=cfg.target_fps), indent=2))

    print("\n=== Stage 2 (v2) summary ===")
    print(f"videos: {len(vinfos)}  windows: {len(windows)}  features/window: {len(feature_cols)}")
    print(windows.groupby(["label_str", "reviewed"])["video"].nunique().to_string())
    print("window labels:", windows["wlabel"].value_counts().sort_index().to_dict(),
          "(1 reading, 0 not, -1 unknown, -2 boundary)")
    for e in errors:
        print("  ERROR", e)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 2 (v2): reading windows + strong labels.")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = DEFAULT_CONFIG
    cfg_path = MODELS_DIR / "config.json"
    if cfg_path.exists():
        cfg = Config.from_json(cfg_path)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.to_json(cfg_path)
    return build(cfg, args.workers)


if __name__ == "__main__":
    sys.exit(main())
