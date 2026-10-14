"""
predict.py (v2) — Stage 4: score any video -> per-second reading probability,
reading segments (with reasons), and a verdict based on total reading time.

Pipeline: extract (15 fps + blendshapes) -> clean -> 8 s windows every 1 s ->
XGBoost reading probability per window -> per-second score (reading.per_second)
-> segments above the tuned second threshold (short gaps merged, blips dropped)
-> verdict: total reading seconds >= the tuned video threshold => CHEATING.

Output JSON keeps v1's keys so older consumers still work, with v2 meanings:
    verdict             "CHEATING" | "clean"
    probability         evidence in [0, 1]: reading_s / (reading_s + video_threshold),
                        so 0.5 is exactly the decision boundary
    threshold           0.5 (the evidence boundary)
    flagged_segments    reading segments on the VIDEO clock, each with reasons
plus v2 fields: reading_seconds, reading_share_of_speech, speaking_seconds,
video_threshold_seconds, second_threshold, seconds (per-second probabilities).

Outputs:
    reports/<video>__prediction.json
    reports/<video>__timeline.npz          (consumed by report.py)

Usage:
    python src/predict.py path/to/video.mp4
    python src/predict.py path/to/video.mp4 --json-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import Config, DEFAULT_CONFIG, MODELS_DIR, REPORTS_DIR
import build_windows as bw
import reading


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_winner(device="cpu"):
    """(artifact, scorer). `device` is accepted for API compatibility; trees run on CPU."""
    mpath = MODELS_DIR / "model.json"
    if not mpath.exists():
        print(f"No trained model at {mpath}. Run src/train.py first.")
        sys.exit(1)
    art = json.loads(mpath.read_text())
    if art.get("kind") != "reading_xgb":
        print(f"{mpath} holds a '{art.get('kind')}' model; v2 needs 'reading_xgb'. Re-run train.py.")
        sys.exit(1)
    # the model only fits features sampled at the rate it was trained on (15 fps)
    cfg_path = MODELS_DIR / "config.json"
    cfg_fps = json.loads(cfg_path.read_text()).get("target_fps") if cfg_path.exists() else None
    if art.get("target_fps") and cfg_fps and abs(art["target_fps"] - cfg_fps) > 1e-6:
        print(f"{mpath} was trained on {art['target_fps']:g} fps features but {cfg_path.name} "
              f"samples {cfg_fps:g} fps. Re-run train.py, or restore the matching config.json.")
        sys.exit(1)
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.load_model(str(MODELS_DIR / "reading_xgb.json"))
    feats = art["features"]

    def scorer(win_df, _unused=None):
        X = win_df.reindex(columns=feats).to_numpy(dtype=np.float32)
        return model.predict_proba(X)[:, 1]

    return art, scorer


# --------------------------------------------------------------------------- #
# Feature extraction for one video (in-memory)
# --------------------------------------------------------------------------- #
def video_to_windows(video: Path, cfg: Config, max_seconds=None, progress=None, workers=1):
    """Extract -> clean -> reading windows. Returns (win_df, clean, feature_parquet)."""
    from extract_features import extract_one

    # Cache OUTSIDE data/features/ so scored videos never leak into the training set.
    cache_dir = REPORTS_DIR / "_predict_cache"
    out = extract_one(video, "scored", cfg, "mediapipe", args_max_seconds=max_seconds,
                      out_dir=cache_dir, progress=progress, workers=workers)
    if out.get("status") != "ok":
        raise RuntimeError(f"extraction failed: {out.get('status')}")
    if progress is not None:
        progress("windows", None)
    df = pd.read_parquet(out["out"])
    clean = bw.clean_video(df, cfg)
    feats, starts = bw.window_features(clean, cfg)
    if feats.empty:
        raise RuntimeError(f"video too short: needs at least {cfg.read_window_seconds:.0f}s "
                           f"of usable signal")
    dt, t0 = clean["dt"], clean["t0"]
    W = int(round(cfg.read_window_seconds / dt))
    feats["win_start_s"] = np.round(t0 + starts * dt, 3)
    feats["win_end_s"] = np.round(t0 + (starts + W) * dt, 3)
    return feats, clean, str(out["out"])


# --------------------------------------------------------------------------- #
# Reasons
# --------------------------------------------------------------------------- #
def segment_reasons(win_df: pd.DataFrame, start_s: float, end_s: float, art: dict, top: int = 3):
    """Human-readable signals for one segment: features whose mean over the segment's
    windows lies beyond the clean-speech reference (stored by train.py)."""
    refs = art.get("reason_reference", [])
    mid = (win_df["win_start_s"] + win_df["win_end_s"]) / 2
    sub = win_df[(mid >= start_s) & (mid < end_s)]
    if sub.empty or not refs:
        return []
    scored = []
    for r in refs:
        if r["feature"] not in sub:
            continue
        val = float(sub[r["feature"]].mean())
        z = r["direction"] * (val - r["clean_median"]) / max(r["scale"], 1e-9)
        beyond = val >= r["ref"] if r["direction"] > 0 else val <= r["ref"]
        if beyond and np.isfinite(z):
            scored.append((z, r["label"]))
    scored.sort(reverse=True)
    return [lab for _, lab in scored[:top]]


def window_reasons(row, cfg: Config = None, art: dict = None):
    """Reasons for a single window (kept for API compatibility with v1 callers)."""
    if art is None:
        return []
    df = pd.DataFrame([row])
    return segment_reasons(df, float(row["win_start_s"]), float(row["win_end_s"]) + 1e-6, art)


# --------------------------------------------------------------------------- #
# Main scoring
# --------------------------------------------------------------------------- #
def mmss(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m:02d}:{s:02d}"


STAT_TEXT = {
    "reading_seconds": "reading time",
    "reading_share": "share of the video spent reading",
    "reading_share_of_speech": "share of speaking time spent reading",
    "speech_prob_mean": "average reading probability while speaking",
    "speech_prob_p90": "90th-pct reading probability while speaking",
    "top10_mean": "mean of the top 10% reading probabilities",
    "p95": "95th-pct reading probability",
}


def _rule_text(stat: str, thr: float) -> str:
    if stat == "reading_seconds":
        return f"CHEATING when reading time >= {reading.mmss(thr)}"
    return f"CHEATING when {STAT_TEXT.get(stat, stat)} >= {thr:.2f}"


def score_video(video: Path, cfg: Config, artifact, scorer, flag_threshold=None,
                max_seconds=None, progress=None, workers=1):
    win_df, clean, feat_path = video_to_windows(video, cfg, max_seconds, progress, workers)
    win_df = win_df.copy()
    win_df["prob"] = np.asarray(scorer(win_df), dtype=float)

    art = artifact
    sec_thr = float(flag_threshold if flag_threshold is not None else art["second_threshold"])
    n_sec = int(np.ceil(win_df["win_end_s"].max()))
    score = reading.per_second(win_df["win_start_s"], win_df["win_end_s"], win_df["prob"],
                               n_sec, art.get("smooth_s", 3))
    result = result_from_scores(video, score, win_df, clean, art, sec_thr)
    return result, win_df, clean, feat_path


def result_from_scores(video: Path, score: np.ndarray, win_df: pd.DataFrame, clean: dict,
                       art: dict, sec_thr: float, stat_speaking=None) -> dict:
    """Per-second reading score -> segments with reasons, speaking, verdict.
    Shared by score_video and the web app's pre-scored library (which passes
    out-of-fold scores), so both build their results one way only.
    `stat_speaking` (per second, 0..1) replaces the frame-based speaking in the
    video statistic; the library passes train.py's window-based speaking so its
    out-of-fold verdicts match the cross-validation tables exactly."""
    n_sec = len(score)
    segs = reading.segments(score, sec_thr, art["min_segment_seconds"], art["merge_gap_seconds"])

    # speaking per second on the video clock
    dt, t0 = clean["dt"], clean["t0"]
    sp = clean["speaking"]
    speak_sec = np.zeros(n_sec)
    for i in range(n_sec):
        a, b = int(round((i - t0) / dt)), int(round((i + 1 - t0) / dt))
        a, b = max(0, a), min(len(sp), b)
        speak_sec[i] = float(sp[a:b].mean()) if b > a else 0.0
    speaking_s = float(np.sum(sp) * dt)

    stats = reading.video_stats(score, speaking_s, sec_thr, art["min_segment_seconds"],
                                art["merge_gap_seconds"],
                                speak_sec if stat_speaking is None else stat_speaking)
    vthr = float(art["video_threshold"])
    vstat = art.get("video_stat", "reading_seconds")
    value = stats[vstat]
    verdict = "CHEATING" if value >= vthr else "clean"
    rs = stats["reading_seconds"]
    if vstat == "reading_seconds":
        evidence = rs / (rs + vthr) if (rs + vthr) > 0 else 0.0
    else:
        evidence = value / (value + vthr) if (value + vthr) > 0 else 0.0

    segments = []
    for s, e, peak, mean in segs:
        reasons = segment_reasons(win_df, s, e, art)
        sp_frac = float(speak_sec[int(s):int(e)].mean()) if e > s else 0.0
        lead = "reading while answering" if sp_frac >= 0.5 else "reading on screen"
        segments.append(dict(
            start_s=s, end_s=e, range=f"{mmss(s)}–{mmss(e)}", duration_s=e - s,
            peak_prob=round(peak, 4), mean_prob=round(mean, 4), speaking_frac=round(sp_frac, 3),
            reason=lead + (" · " + ", ".join(reasons[:2]) if reasons else ""), reasons=reasons))

    result = dict(
        version=2, video=str(video), verdict=verdict,
        probability=round(float(evidence), 4), threshold=0.5,
        video_stat=vstat, video_stat_value=round(float(value), 4),
        video_threshold_seconds=round(vthr, 2) if vstat == "reading_seconds" else None,
        video_threshold=round(vthr, 4),
        verdict_rule=_rule_text(vstat, vthr),
        reading_seconds=round(rs, 1), reading_segments=len(segments),
        reading_share_of_speech=round(stats["reading_share_of_speech"], 4),
        speaking_seconds=round(speaking_s, 1),
        second_threshold=round(sec_thr, 4), flag_threshold=round(sec_thr, 4),
        n_windows=int(np.isfinite(score).sum()), n_flagged_windows=int(round(rs)),
        model=dict(kind=art["kind"], feature_set="mediapipe+blendshapes", experiment=art.get("experiment")),
        flagged_segments=segments,
        seconds=[None if not np.isfinite(x) else round(float(x), 3) for x in score],
        speaking=[round(float(x), 2) for x in speak_sec],
    )
    return result


def save_timeline(video: Path, result, win_df, clean):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = video.stem
    (REPORTS_DIR / f"{stem}__prediction.json").write_text(json.dumps(result, indent=2))
    sec = np.array([np.nan if x is None else x for x in result["seconds"]], dtype=float)
    np.savez_compressed(
        REPORTS_DIR / f"{stem}__timeline.npz",
        gaze_x=clean["gaze_x"], gaze_y=clean["gaze_y"],
        eye_h=clean.get("eye_h", clean["gaze_x"]), eye_v=clean.get("eye_v", clean["gaze_y"]),
        yaw=clean["yaw"], pitch=clean["pitch"], ear=clean["ear"], blink=clean["blink"],
        mouth=clean["mouth"], jaw=clean.get("jaw", np.zeros_like(clean["mouth"])),
        speaking=clean["speaking"], valid=clean["valid"],
        dt=np.array([clean["dt"]]), t0=np.array([clean["t0"]]),
        second_prob=sec, win_start_s=win_df["win_start_s"].to_numpy(),
        win_end_s=win_df["win_end_s"].to_numpy(), prob=win_df["prob"].to_numpy(),
        result_json=np.array([json.dumps(result)]),
    )
    return REPORTS_DIR / f"{stem}__prediction.json"


def print_summary(result):
    v = result["verdict"]
    banner = "🚩 CHEATING (reading detected)" if v == "CHEATING" else "✅ clean"
    print("\n" + "=" * 64)
    print(f"  {banner}")
    print(f"  reading {reading.mmss(result['reading_seconds'])} in {result['reading_segments']} segment(s)"
          f"  ·  {result['reading_share_of_speech']:.0%} of speaking time")
    print(f"  verdict rule: {result['verdict_rule']}  (this video: {result['video_stat_value']:.3f})")
    print("=" * 64)
    for s in result["flagged_segments"]:
        print(f"    {s['range']}  ({s['duration_s']:.0f}s, peak {s['peak_prob']:.2f})  {s['reason']}")
    if not result["flagged_segments"]:
        print("  no reading segments above the threshold.")
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 4 (v2): reading detection for one video.")
    ap.add_argument("video", help="path to a video file")
    ap.add_argument("--flag-threshold", type=float, default=None,
                    help="per-second reading probability to flag (default: tuned)")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--workers", type=int, default=1, help="MediaPipe threads")
    ap.add_argument("--json-only", action="store_true")
    args = ap.parse_args(argv)

    cfg = DEFAULT_CONFIG
    if (MODELS_DIR / "config.json").exists():
        cfg = Config.from_json(MODELS_DIR / "config.json")
    video = Path(args.video)
    if not video.exists():
        print(f"Video not found: {video}")
        return 1
    artifact, scorer = load_winner()
    result, win_df, clean, _ = score_video(video, cfg, artifact, scorer, args.flag_threshold,
                                           args.max_seconds, workers=args.workers)
    path = save_timeline(video, result, win_df, clean)
    if args.json_only:
        print(json.dumps({k: v for k, v in result.items() if k not in ("seconds", "speaking")}, indent=2))
    else:
        print_summary(result)
        print(f"  saved: {path}")
        print(f"  report: python src/report.py \"{video}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
