"""
speaking.py — speaking/turn engine (visual).

Answers ONE question: *when is the candidate speaking?* — from lip motion alone.
The rolling std of the inner-lip mouth-aspect-ratio channel goes through adaptive
hysteresis thresholding to give speaking bursts; the gap before each burst is the
silence since the previous one.

Only the engineered window features read this (speaking-conditioned contrasts and
turn structure). The CNN reads the raw sequence channels and the reason tags read
speaking-independent fields, so verdicts never depend on it. There is no audio path.

Contract (SpeakingResult): per-sample `speaking_mask` on the uniform time grid,
continuous `activity` (for plots), and `turns` — [{onset_s, end_s, duration_s,
gap_s}] with `source` = "visual" | "none".

CLI:
    python src/speaking.py --selftest              # synthetic-signal validation
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

import numpy as np

from common import Config, DEFAULT_CONFIG


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
@dataclass
class SpeakingResult:
    source: str                              # "visual" | "none"
    speaking_mask: np.ndarray                # bool, per uniform sample
    activity: np.ndarray                     # continuous mouth activity (plots)
    turns: list = field(default_factory=list)          # candidate answer turns
    meta: dict = field(default_factory=dict)

    def candidate_segments(self, dt: float) -> list:
        return segments_from_mask(self.speaking_mask, dt)


# --------------------------------------------------------------------------- #
# Signal helpers
# --------------------------------------------------------------------------- #
def _fill_nan(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).copy()
    fin = np.isfinite(a)
    if not fin.any():
        return np.zeros_like(a)
    if fin.all():
        return a
    idx = np.arange(len(a))
    a[~fin] = np.interp(idx[~fin], idx[fin], a[fin])
    return a


def rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    """Centered rolling standard deviation (NaN-safe via pre-fill)."""
    import pandas as pd
    win = max(2, int(win))
    return (pd.Series(_fill_nan(x))
            .rolling(win, min_periods=max(2, win // 2), center=True)
            .std()
            .to_numpy(copy=True))


def mouth_activity(mouth: np.ndarray, dt: float, cfg: Config) -> np.ndarray:
    """Continuous speaking-likelihood signal: windowed variability of the MAR.
    Even when syllable motion (~4-7 Hz) aliases at 7.5-15 fps sampling, samples
    scatter across the open-close range, so windowed std stays high in speech."""
    win = max(2, int(round(cfg.vad_act_window_s / max(dt, 1e-6))))
    act = rolling_std(mouth, win)
    act[~np.isfinite(act)] = 0.0
    return act


# --------------------------------------------------------------------------- #
# Segments <-> mask
# --------------------------------------------------------------------------- #
def segments_from_mask(mask: np.ndarray, dt: float) -> list:
    """Boolean mask -> [(start_s, end_s)]."""
    segs = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            segs.append((i * dt, j * dt))
            i = j
        else:
            i += 1
    return segs


def mask_from_segments(segs: list, n: int, dt: float) -> np.ndarray:
    """Inverse of segments_from_mask (round-to-sample so round-trips are exact
    even after segment endpoints were rounded for JSON)."""
    mask = np.zeros(n, dtype=bool)
    for s, e in segs:
        i0 = max(0, int(round(s / dt)))
        i1 = min(n, max(i0 + 1, int(round(e / dt))))
        mask[i0:i1] = True
    return mask


def postprocess_segments(segs: list, min_burst_s: float, merge_gap_s: float) -> list:
    """Merge close segments, then drop too-short bursts (in that order, so two
    short chunks separated by a micro-pause survive as one real burst)."""
    if not segs:
        return []
    segs = sorted(segs)
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if (e - s) >= min_burst_s]


# --------------------------------------------------------------------------- #
# Visual VAD
# --------------------------------------------------------------------------- #
def visual_speaking(mouth: np.ndarray, valid: np.ndarray, dt: float,
                    cfg: Config = DEFAULT_CONFIG) -> SpeakingResult:
    """Speaking bursts from mouth-motion energy with per-video adaptive
    hysteresis. `valid` gates on face-found frames."""
    act = mouth_activity(mouth, dt, cfg)
    v = np.asarray(valid, dtype=bool)
    usable = act[v & np.isfinite(act)]
    if usable.size < 10:
        return SpeakingResult("none", np.zeros(len(mouth), bool), act,
                              meta={"reason": "insufficient_valid_frames"})

    floor = float(np.percentile(usable, cfg.vad_floor_pctile))
    p90 = float(np.percentile(usable, 90))
    # No real dynamic range = no detectable speech (flat noise or frozen mouth).
    if p90 < cfg.vad_min_dynamic * max(floor, 1e-9):
        return SpeakingResult("visual", np.zeros(len(mouth), bool), act,
                              meta={"reason": "no_speech_dynamic",
                                    "floor": floor, "p90": p90, "n_bursts": 0})
    rng = p90 - floor
    # Absolute floor: on a still mouth the whole adaptive range is tracking
    # noise — never let the ON threshold scale down into it.
    thr_on = max(floor + cfg.vad_on_frac * rng, cfg.vad_abs_min_act)
    thr_off = max(floor + cfg.vad_off_frac * rng, 0.6 * cfg.vad_abs_min_act)

    # hysteresis state machine
    mask = np.zeros(len(act), dtype=bool)
    on = False
    for i, a in enumerate(act):
        if not v[i]:
            on = False
            continue
        if on:
            on = a >= thr_off
        else:
            on = a >= thr_on
        mask[i] = on

    segs = postprocess_segments(segments_from_mask(mask, dt),
                                cfg.vad_min_burst_s, cfg.vad_merge_gap_s)
    mask = mask_from_segments(segs, len(act), dt)
    turns = turns_from_segments(segs)
    return SpeakingResult("visual", mask, act, turns=turns,
                          meta={"thr_on": thr_on, "thr_off": thr_off,
                                "floor": floor, "p90": p90,
                                "n_bursts": len(segs)})


def visual_speaking_v2(mouth: np.ndarray, jaw, valid: np.ndarray, dt: float,
                       cfg: Config = DEFAULT_CONFIG) -> SpeakingResult:
    """v2: fuse lip-gap (MAR) and blendshape jawOpen motion.

    Each channel's 1 s rolling std is scaled by its own per-video P90, the two are
    averaged, and the v1 hysteresis runs on the fused activity. A channel whose raw
    activity never clears its absolute floor contributes nothing (a still mouth
    can't be "scaled up" into speech). Falls back to visual_speaking() when the
    video has no blendshapes.
    """
    if jaw is None or not np.isfinite(jaw).any():
        return visual_speaking(mouth, valid, dt, cfg)
    v = np.asarray(valid, dtype=bool)
    parts = []
    for x, floor in ((mouth, cfg.vad_abs_min_act), (jaw, 0.02)):
        act = mouth_activity(x, dt, cfg)
        u = act[v & np.isfinite(act)]
        if u.size < 10:
            continue
        p90 = float(np.percentile(u, 90))
        if p90 < floor:            # no articulation on this channel at all
            parts.append(np.zeros_like(act))
            continue
        parts.append(np.clip(act / p90, 0, 3) * (p90 >= floor))
    if not parts:
        return SpeakingResult("none", np.zeros(len(mouth), bool), np.zeros(len(mouth)),
                              meta={"reason": "insufficient_valid_frames"})
    act = np.mean(parts, axis=0)
    usable = act[v]
    floor = float(np.percentile(usable, cfg.vad_floor_pctile))
    p90 = float(np.percentile(usable, 90))
    if p90 <= 1e-6 or p90 < cfg.vad_min_dynamic * max(floor, 1e-9):
        return SpeakingResult("visual", np.zeros(len(mouth), bool), act,
                              meta={"reason": "no_speech_dynamic", "n_bursts": 0})
    rng = p90 - floor
    thr_on, thr_off = floor + cfg.vad_on_frac * rng, floor + cfg.vad_off_frac * rng
    mask = np.zeros(len(act), dtype=bool)
    on = False
    for i, a in enumerate(act):
        if not v[i]:
            on = False
            continue
        on = a >= (thr_off if on else thr_on)
        mask[i] = on
    segs = postprocess_segments(segments_from_mask(mask, dt), cfg.vad_min_burst_s, cfg.vad_merge_gap_s)
    mask = mask_from_segments(segs, len(act), dt)
    return SpeakingResult("visual", mask, act, turns=turns_from_segments(segs),
                          meta={"thr_on": thr_on, "thr_off": thr_off, "n_bursts": len(segs)})


# --------------------------------------------------------------------------- #
# Turns
# --------------------------------------------------------------------------- #
def turns_from_segments(cand_segs: list) -> list:
    """Candidate answer turns with the preceding gap: the silence since the
    candidate's previous burst (it includes the unheard question)."""
    turns = []
    prev_end = 0.0
    for s, e in cand_segs:
        gap = max(0.0, s - prev_end)
        turns.append(dict(onset_s=round(s, 3), end_s=round(e, 3),
                          duration_s=round(e - s, 3), gap_s=round(gap, 3)))
        prev_end = e
    return turns


# --------------------------------------------------------------------------- #
# Selftest
# --------------------------------------------------------------------------- #
def _synth_mouth(duration_s=120.0, fps=15.0, bursts=None, seed=0):
    """Synthetic MAR: quiet floor + oscillating 'speech' bursts."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fps)
    dt = 1.0 / fps
    t = np.arange(n) * dt
    mar = 0.05 + rng.normal(0, 0.004, n)          # closed-mouth floor + jitter
    bursts = bursts or []
    for s, e in bursts:
        seg = (t >= s) & (t < e)
        # 4 Hz articulation (aliases at 15fps — exactly the real-world case)
        mar[seg] += 0.10 + 0.08 * np.abs(np.sin(2 * np.pi * 4.0 * t[seg])) \
            + rng.normal(0, 0.01, seg.sum())
    return t, mar, dt


def _iou(pred: list, truth: list, n: int, dt: float) -> float:
    pm = mask_from_segments(pred, n, dt)
    tm = mask_from_segments(truth, n, dt)
    inter = (pm & tm).sum()
    union = (pm | tm).sum()
    return inter / union if union else 1.0


def run_selftest() -> bool:
    cfg = DEFAULT_CONFIG
    ok = True
    checks = []

    # 1. bursts recovered
    truth = [(10, 25), (40, 70), (90, 110)]
    t, mar, dt = _synth_mouth(120, 15, truth, seed=1)
    valid = np.ones(len(mar), bool)
    res = visual_speaking(mar, valid, dt, cfg)
    iou = _iou(res.candidate_segments(dt), truth, len(mar), dt)
    checks.append((f"burst recovery IoU={iou:.2f} >= 0.85", iou >= 0.85))
    checks.append((f"burst count {res.meta.get('n_bursts')} == 3", res.meta.get("n_bursts") == 3))

    # 2. flat noise -> no speech
    t2, mar2, dt2 = _synth_mouth(120, 15, [], seed=2)
    res2 = visual_speaking(mar2, np.ones(len(mar2), bool), dt2, cfg)
    frac = res2.speaking_mask.mean()
    checks.append((f"flat noise speaking frac={frac:.3f} <= 0.05", frac <= 0.05))

    # 3. micro-pause merged
    t3, mar3, dt3 = _synth_mouth(60, 15, [(10, 20), (20.3, 30)], seed=3)
    res3 = visual_speaking(mar3, np.ones(len(mar3), bool), dt3, cfg)
    nseg = len(res3.candidate_segments(dt3))
    checks.append((f"0.3s micro-pause merged: {nseg} segment(s) == 1", nseg == 1))

    # 4. turn gaps (visual): burst2 onset 40 - burst1 end 25 = 15s
    gaps = [tr["gap_s"] for tr in res.turns]
    gap_ok = len(gaps) == 3 and abs(gaps[1] - 15.0) < 1.5
    checks.append((f"visual turn gaps {['%.1f' % g for g in gaps]} (expect ~[10,15,20])", gap_ok))

    print("[speaking selftest]")
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("[speaking selftest]", "ALL PASS ✅" if ok else "FAILURES ❌")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if run_selftest() else 1)
    print(__doc__)
