"""
speaking.py — unified speaking/turn engine (v2).

Answers ONE question either way the data allows: *when is the candidate speaking?*
Everything anti-Cluely hangs on it — reading-while-answering, lag-loop timing,
answer burst structure.

Two sources behind one contract:

  * AUDIO path (when the recording has an audio track): ffmpeg-decode to 16 kHz
    mono → silero-vad speech segments → lip-sync attribution (segments where the
    candidate's mouth moves in sync = candidate speech; speech with a still mouth
    = interviewer). Yields REAL turns: interviewer-question-end → answer-start gap.
  * VISUAL path (video-only fallback): rolling std of the inner-lip mouth-aspect-
    ratio channel → adaptive hysteresis thresholding → speaking bursts. Gaps are
    silence-before-burst (weaker: silence includes the unheard question).

Contract (SpeakingResult): per-sample `speaking_mask` on the uniform time grid,
continuous `activity` (for plots), and `turns` — [{onset_s, end_s, duration_s,
gap_s}] with `source` = "audio" | "visual" | "none".

CLI:
    python src/speaking.py --selftest              # synthetic-signal validation
    python src/speaking.py --test-audio <video>    # audio path on a real file
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from common import Config, DEFAULT_CONFIG, _mad


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
@dataclass
class SpeakingResult:
    source: str                              # "audio" | "visual" | "none"
    speaking_mask: np.ndarray                # bool, per uniform sample
    activity: np.ndarray                     # continuous mouth activity (plots)
    turns: list = field(default_factory=list)          # candidate answer turns
    interviewer_segments: list = field(default_factory=list)  # [(s,e)] audio only
    meta: dict = field(default_factory=dict)

    def candidate_segments(self, dt: float) -> list:
        return segments_from_mask(self.speaking_mask, dt)

    def to_sidecar(self, dt: float) -> dict:
        """JSON-serializable summary; the mask is reconstructed from segments."""
        return dict(
            source=self.source,
            candidate_segments=[[round(s, 3), round(e, 3)] for s, e in self.candidate_segments(dt)],
            interviewer_segments=[[round(s, 3), round(e, 3)] for s, e in self.interviewer_segments],
            turns=self.turns,
            meta=self.meta,
        )


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
    turns = turns_from_segments(segs, None)
    return SpeakingResult("visual", mask, act, turns=turns,
                          meta={"thr_on": thr_on, "thr_off": thr_off,
                                "floor": floor, "p90": p90,
                                "n_bursts": len(segs)})


# --------------------------------------------------------------------------- #
# Turns
# --------------------------------------------------------------------------- #
def turns_from_segments(cand_segs: list, interviewer_segs: Optional[list]) -> list:
    """Candidate answer turns with the preceding gap.

    With interviewer segments (audio path): gap = answer onset - end of the most
    recent interviewer speech before it — the true Cluely lag-loop quantity.
    Without (visual path): gap = silence since the candidate's previous burst.
    """
    turns = []
    prev_end = 0.0
    for s, e in cand_segs:
        if interviewer_segs:
            prior = [ie for _, ie in interviewer_segs if ie <= s + 1e-9]
            ref = max(prior) if prior else prev_end
            # question may end after our previous answer started overlapping talk;
            # never report a negative gap
            gap = max(0.0, s - max(ref, prev_end)) if prior else max(0.0, s - prev_end)
        else:
            gap = max(0.0, s - prev_end)
        turns.append(dict(onset_s=round(s, 3), end_s=round(e, 3),
                          duration_s=round(e - s, 3), gap_s=round(gap, 3)))
        prev_end = e
    return turns


# --------------------------------------------------------------------------- #
# Audio path
# --------------------------------------------------------------------------- #
def has_audio_track(video: Path) -> bool:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, timeout=30)
        return bool(out.stdout.strip())
    except Exception:
        return False


def decode_audio(video: Path, sr: int) -> Optional[np.ndarray]:
    """Decode the audio track to float32 mono at `sr` via ffmpeg (piped)."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-vn",
             "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"],
            capture_output=True, timeout=600)
        if out.returncode != 0 or not out.stdout:
            return None
        return np.frombuffer(out.stdout, dtype=np.float32)
    except Exception:
        return None


def _silero_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("silero_vad") is not None


def audio_speech_segments(wav: np.ndarray, sr: int, cfg: Config) -> list:
    """Speech segments [(s,e)] from silero-vad."""
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps

    model = load_silero_vad()
    ts = get_speech_timestamps(
        torch.tensor(wav), model,
        sampling_rate=sr,
        threshold=cfg.audio_vad_threshold,
        min_speech_duration_ms=int(cfg.audio_min_speech_s * 1000),
        min_silence_duration_ms=int(cfg.audio_min_silence_s * 1000),
    )
    return [(t["start"] / sr, t["end"] / sr) for t in ts]


def _otsu_split(scores: np.ndarray, min_sep: float = 0.3):
    """Best 2-cluster threshold on 1-D scores (Otsu). Returns the split value,
    or None when the clusters aren't separated by at least `min_sep`."""
    x = np.sort(scores[np.isfinite(scores)])
    if len(x) < 4:
        return None
    best, best_thr = -1.0, None
    for i in range(2, len(x) - 1):
        lo, hi = x[:i], x[i:]
        between = len(lo) * len(hi) * (hi.mean() - lo.mean()) ** 2
        if between > best:
            best, best_thr = between, (lo.max() + hi.min()) / 2.0
            best_sep = hi.mean() - lo.mean()
    if best_thr is None or best_sep < min_sep:
        return None
    return float(best_thr)


def attribute_segments(segs: list, activity: np.ndarray, dt: float, cfg: Config):
    """Split audio speech segments into candidate vs interviewer via lip sync:
    a segment where the candidate's mouth moves in sync is the candidate talking;
    speech over a still mouth is the interviewer.

    Per-segment score = upper-quartile mouth-activity z inside the segment
    (robust to intra-segment pauses). The candidate/interviewer split threshold
    adapts to each recording (Otsu 2-cluster) and falls back to the fixed
    cfg.lipsync_z_threshold when the score distribution has no clear gap."""
    act = np.asarray(activity, dtype=float)
    fin = act[np.isfinite(act)]
    med = float(np.median(fin)) if fin.size else 0.0
    spread = _mad(fin) or 1e-6
    z = (act - med) / spread

    n = len(act)
    scored = []
    for s, e in segs:
        i0, i1 = max(0, int(s / dt)), min(n, max(int(s / dt) + 1, int(e / dt)))
        seg_z = z[i0:i1]
        seg_z = seg_z[np.isfinite(seg_z)]
        score = float(np.percentile(seg_z, 75)) if seg_z.size else 0.0
        scored.append((s, e, score))

    thr = _otsu_split(np.array([sc for _, _, sc in scored]))
    if thr is None:
        thr = cfg.lipsync_z_threshold
    cand = [(s, e) for s, e, sc in scored if sc >= thr]
    inter = [(s, e) for s, e, sc in scored if sc < thr]
    return cand, inter


def audio_speaking(video: Path, mouth: np.ndarray, valid: np.ndarray, dt: float,
                   cfg: Config = DEFAULT_CONFIG) -> Optional[SpeakingResult]:
    """Full audio path. Returns None when anything is missing (no track, no
    silero, decode failure) so the caller falls back to the visual path."""
    if not has_audio_track(video) or not _silero_available():
        return None
    wav = decode_audio(video, cfg.audio_sr)
    if wav is None or len(wav) < cfg.audio_sr:
        return None
    try:
        segs = audio_speech_segments(wav, cfg.audio_sr, cfg)
    except Exception:
        return None
    if not segs:
        return None

    act = mouth_activity(mouth, dt, cfg)
    cand, inter = attribute_segments(segs, act, dt, cfg)
    cand = postprocess_segments(cand, cfg.vad_min_burst_s, cfg.vad_merge_gap_s)
    mask = mask_from_segments(cand, len(mouth), dt)
    turns = turns_from_segments(cand, inter)
    return SpeakingResult("audio", mask, act, turns=turns, interviewer_segments=inter,
                          meta={"n_speech_segments": len(segs),
                                "n_candidate": len(cand), "n_interviewer": len(inter)})


# --------------------------------------------------------------------------- #
# Entry point used by the pipeline
# --------------------------------------------------------------------------- #
def compute_speaking(video: Optional[Path], mouth: np.ndarray, valid: np.ndarray,
                     dt: float, cfg: Config = DEFAULT_CONFIG) -> SpeakingResult:
    """Best available speaking analysis: audio when possible, else visual."""
    if video is not None:
        res = audio_speaking(Path(video), mouth, valid, dt, cfg)
        if res is not None:
            return res
    return visual_speaking(mouth, valid, dt, cfg)


def speaking_mask_from_sidecar(sidecar: dict, n: int, dt: float) -> np.ndarray:
    return mask_from_segments([tuple(s) for s in sidecar.get("candidate_segments", [])], n, dt)


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

    # 5. audio attribution + audio-turn logic on mocked segments (no silero needed)
    #    candidate answers at 30-50 & 70-90 (mouth moving); interviewer asks at
    #    20-27 & 60-66 (mouth still).
    cand_truth = [(30, 50), (70, 90)]
    t5, mar5, dt5 = _synth_mouth(100, 15, cand_truth, seed=5)
    act5 = mouth_activity(mar5, dt5, cfg)
    segs = [(20, 27), (30, 50), (60, 66), (70, 90)]
    cand, inter = attribute_segments(segs, act5, dt5, cfg)
    checks.append((f"lip-sync attribution: cand={cand} inter={inter}",
                   cand == [(30, 50), (70, 90)] and inter == [(20, 27), (60, 66)]))
    turns = turns_from_segments(cand, inter)
    # answer1 gap = 30 - 27 = 3s (after question 1); answer2 gap = 70 - 66 = 4s
    tg = [tr["gap_s"] for tr in turns]
    checks.append((f"audio turn gaps {tg} == [3.0, 4.0]",
                   len(tg) == 2 and abs(tg[0] - 3.0) < 0.01 and abs(tg[1] - 4.0) < 0.01))

    # 6. sidecar round-trip
    sc = res.to_sidecar(dt)
    mask_rt = speaking_mask_from_sidecar(sc, len(mar), dt)
    checks.append(("sidecar mask round-trip", bool((mask_rt == res.speaking_mask).all())))

    print("[speaking selftest]")
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("[speaking selftest]", "ALL PASS ✅" if ok else "FAILURES ❌")
    return ok


def test_audio_file(video: str) -> int:
    """Integration check of the audio path on a real recording."""
    video = Path(video)
    print(f"audio track: {has_audio_track(video)}   silero installed: {_silero_available()}")
    wav = decode_audio(video, DEFAULT_CONFIG.audio_sr)
    print(f"decoded: {None if wav is None else f'{len(wav)/DEFAULT_CONFIG.audio_sr:.1f}s'}")
    if wav is None or not _silero_available():
        return 1
    segs = audio_speech_segments(wav, DEFAULT_CONFIG.audio_sr, DEFAULT_CONFIG)
    dur = sum(e - s for s, e in segs)
    print(f"speech segments: {len(segs)}  total speech {dur:.0f}s "
          f"({dur / (len(wav)/DEFAULT_CONFIG.audio_sr):.0%} of audio)")
    for s, e in segs[:8]:
        print(f"   {s:7.1f} - {e:7.1f}")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if run_selftest() else 1)
    if "--test-audio" in sys.argv:
        idx = sys.argv.index("--test-audio")
        sys.exit(test_audio_file(sys.argv[idx + 1]))
    print(__doc__)
