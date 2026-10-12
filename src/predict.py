"""
predict.py — Stage 4: score any video -> verdict + probability + flagged segments.

Runs the same pipeline as training on a single video: extract per-frame gaze ->
clean + window -> score each window with the shipped winner model -> aggregate to
a video-level probability (top-k mean) -> verdict via the tuned threshold. Suspicious
windows are merged into readable time ranges, each labeled with its dominant reason
(reading sweeps / off-center gaze / eyes-move-head-still / suppressed blinking).

Outputs:
    console summary + JSON  -> reports/<video>__prediction.json
    timeline artifact        -> reports/<video>__timeline.npz   (consumed by report.py)

Usage:
    python src/predict.py path/to/video.mp4
    python src/predict.py path/to/video.mp4 --flag-threshold 0.5 --json-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    Config, DEFAULT_CONFIG, MODELS_DIR, REPORTS_DIR, L2CS_CHANNELS, SEQ_CHANNELS,
)
import build_windows as bw


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_winner(device="cpu"):
    mpath = MODELS_DIR / "model.json"
    if not mpath.exists():
        print(f"No trained model at {mpath}. Run train.py first.")
        sys.exit(1)
    artifact = json.loads(mpath.read_text())
    if artifact["kind"] != "cnn":
        print(f"{mpath} holds a '{artifact['kind']}' model; only the CNN is supported. "
              "Re-run train.py.")
        sys.exit(1)

    import torch
    ckpt = torch.load(MODELS_DIR / "model_cnn.pt", map_location="cpu", weights_only=False)
    net = bw_build_seq_net(len(artifact["seq_channels"]))
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    mean, std = ckpt["mean"], ckpt["std"]

    def scorer(win_df, X_seq):
        Xn = (X_seq - mean) / std
        with torch.no_grad():
            logits = net(torch.tensor(Xn, dtype=torch.float32, device=device))
            return torch.sigmoid(logits).cpu().numpy()

    return artifact, scorer


def bw_build_seq_net(n_channels):
    from train import _build_seq_net
    return _build_seq_net(n_channels)


# --------------------------------------------------------------------------- #
# Feature + sequence extraction for one video (in-memory)
# --------------------------------------------------------------------------- #
def video_to_windows(video: Path, cfg: Config, backend: str, max_seconds=None, progress=None, workers=1):
    """Extract -> clean -> window a single video. Returns (win_df, X_seq, clean)."""
    from extract_features import extract_one

    # Reuse the batch extractor but write to a prediction cache dir OUTSIDE
    # data/features/, so scored videos never leak into the training set.
    cache_dir = REPORTS_DIR / "_predict_cache"
    out = extract_one(video, "scored", cfg, backend, args_max_seconds=max_seconds,
                      out_dir=cache_dir, progress=progress, workers=workers)
    if out.get("status") != "ok":
        raise RuntimeError(f"extraction failed: {out.get('status')}")
    if progress is not None:
        progress("windows", None)
    df = pd.read_parquet(out["out"])

    has_l2cs = all(c in df.columns for c in L2CS_CHANNELS)
    clean = bw.clean_video(df, cfg, has_l2cs)
    n = len(clean["gaze_x"])
    win_len = int(round(cfg.window_seconds * cfg.target_fps))
    stride = max(1, int(round(cfg.stride_seconds * cfg.target_fps)))
    seq_len = int(round(cfg.window_seconds * cfg.seq_resample_fps))

    rows, seqs = [], []
    if n < win_len:
        raise RuntimeError(f"video too short: {n} < {win_len} samples "
                           f"({cfg.window_seconds}s at {cfg.target_fps}fps)")
    turn_feats = bw.video_turn_features(clean, cfg)
    for start in range(0, n - win_len + 1, stride):
        end = start + win_len
        sub = {k: (v[start:end] if isinstance(v, np.ndarray) else v) for k, v in clean.items()}
        sub["dt"] = clean["dt"]
        feats = bw.compute_window_features(sub, cfg, has_l2cs)
        feats.update(turn_feats)
        feats["win_start_s"] = round(start * clean["dt"], 2)
        feats["win_end_s"] = round(end * clean["dt"], 2)
        rows.append(feats)
        seqs.append(bw._window_sequence(clean, start, end, seq_len))
    return pd.DataFrame(rows), np.stack(seqs, axis=0), clean, str(out["out"])


# --------------------------------------------------------------------------- #
# Aggregation + segment reasons
# --------------------------------------------------------------------------- #
def topk_video_prob(probs: np.ndarray, cfg: Config) -> float:
    p = np.sort(probs)[::-1]
    k = max(cfg.topk_min, int(np.ceil(cfg.topk_fraction * len(p))))
    return float(np.mean(p[:min(k, len(p))]))


REASON_ORDER = ["reading sweeps", "off-center gaze", "eyes move, head still", "suppressed blinking"]


def window_reasons(row, cfg: Config):
    reasons = []
    if row.get("lines_per_minute", 0) >= 12 and row.get("autocorr_peak", 0) >= 0.4:
        reasons.append("reading sweeps")
    if row.get("pct_off_center", 0) >= 0.5 and row.get("gaze_grid_cells", 9) <= 3:
        reasons.append("off-center gaze")
    if (row.get("saccade_rate", 0) >= 20 and abs(row.get("corr_yaw_gazex", 1)) < 0.25
            and row.get("yaw_var", 1e9) < 25):
        reasons.append("eyes move, head still")
    if row.get("blink_rate", 99) <= 4:
        reasons.append("suppressed blinking")
    return reasons


def mmss(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m:02d}:{s:02d}"


def merge_segments(win_df, probs, flag_threshold, cfg: Config):
    """Merge consecutive flagged (overlapping) windows into readable ranges."""
    flagged = probs >= flag_threshold
    segments = []
    i, n = 0, len(win_df)
    while i < n:
        if flagged[i]:
            j = i
            while j < n and flagged[j]:
                j += 1
            start_s = float(win_df.iloc[i]["win_start_s"])
            end_s = float(win_df.iloc[j - 1]["win_end_s"])
            # dominant reason across the merged windows
            reason_counts = {}
            for k in range(i, j):
                for r in window_reasons(win_df.iloc[k], cfg):
                    reason_counts[r] = reason_counts.get(r, 0) + 1
            if reason_counts:
                reason = sorted(reason_counts.items(),
                                key=lambda kv: (-kv[1], REASON_ORDER.index(kv[0])))[0][0]
            else:
                reason = "elevated gaze anomaly"
            segments.append(dict(
                start_s=start_s, end_s=end_s, range=f"{mmss(start_s)}–{mmss(end_s)}",
                peak_prob=float(np.max(probs[i:j])), reason=reason))
            i = j
        else:
            i += 1
    return segments


# --------------------------------------------------------------------------- #
# Main scoring
# --------------------------------------------------------------------------- #
def score_video(video: Path, cfg: Config, artifact, scorer, flag_threshold=None,
                max_seconds=None, progress=None, workers=1):
    # Extract with the backend the winner was trained on (so L2CS channels exist
    # when the winning feature set needs them).
    backend = "mediapipe+l2cs" if "l2cs" in artifact["feature_set"] else "mediapipe"
    win_df, X_seq, clean, feat_path = video_to_windows(video, cfg, backend, max_seconds, progress, workers)
    probs = np.asarray(scorer(win_df, X_seq), dtype=float)
    win_df = win_df.copy()
    win_df["prob"] = probs

    threshold = artifact["threshold"]
    flag_threshold = flag_threshold if flag_threshold is not None else threshold
    video_prob = topk_video_prob(probs, cfg)
    verdict = "CHEATING" if video_prob >= threshold else "clean"
    segments = merge_segments(win_df, probs, flag_threshold, cfg)

    result = dict(
        video=str(video), verdict=verdict, probability=round(video_prob, 4),
        threshold=round(threshold, 4), flag_threshold=round(flag_threshold, 4),
        n_windows=len(win_df), n_flagged_windows=int(np.sum(probs >= flag_threshold)),
        model=dict(kind=artifact["kind"], feature_set=artifact["feature_set"]),
        flagged_segments=segments,
    )
    return result, win_df, clean, feat_path


def save_timeline(video: Path, result, win_df, clean):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = video.stem
    (REPORTS_DIR / f"{stem}__prediction.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        REPORTS_DIR / f"{stem}__timeline.npz",
        gaze_x=clean["gaze_x"], gaze_y=clean["gaze_y"], yaw=clean["yaw"],
        pitch=clean["pitch"], ear=clean["ear"], blink=clean["blink"],
        speaking=clean.get("speaking", np.zeros_like(clean["gaze_x"])),
        valid=clean["valid"], dt=np.array([clean["dt"]]),
        win_start_s=win_df["win_start_s"].to_numpy(),
        win_end_s=win_df["win_end_s"].to_numpy(),
        prob=win_df["prob"].to_numpy(),
        result_json=np.array([json.dumps(result)]),
    )
    return REPORTS_DIR / f"{stem}__prediction.json"


def print_summary(result):
    v = result["verdict"]
    banner = "🚩 CHEATING" if v == "CHEATING" else "✅ clean"
    print("\n" + "=" * 56)
    print(f"  {banner}   probability={result['probability']:.3f}  "
          f"(threshold {result['threshold']:.3f})")
    print("=" * 56)
    print(f"  model: {result['model']['kind']} on {result['model']['feature_set']}")
    print(f"  windows: {result['n_windows']}  flagged: {result['n_flagged_windows']}")
    if result["flagged_segments"]:
        print("  suspicious segments:")
        for s in result["flagged_segments"]:
            print(f"    {s['range']}  {s['reason']:22s} (peak p={s['peak_prob']:.2f})")
    else:
        print("  no suspicious segments above the flag threshold.")
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 4: score a video for cheating.")
    ap.add_argument("video", help="path to a video file")
    ap.add_argument("--flag-threshold", type=float, default=None,
                    help="window prob to flag a segment (default: model threshold)")
    ap.add_argument("--json-only", action="store_true", help="print only the JSON result")
    args = ap.parse_args(argv)

    cfg = DEFAULT_CONFIG
    cfg_path = MODELS_DIR / "config.json"
    if cfg_path.exists():
        cfg = Config.from_json(cfg_path)

    video = Path(args.video)
    if not video.exists():
        print(f"Video not found: {video}")
        return 1

    artifact, scorer = load_winner()
    result, win_df, clean, _ = score_video(video, cfg, artifact, scorer, args.flag_threshold)
    path = save_timeline(video, result, win_df, clean)

    if args.json_only:
        print(json.dumps(result, indent=2))
    else:
        print_summary(result)
        print(f"  saved: {path}")
        print(f"  report: python src/report.py \"{video}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
