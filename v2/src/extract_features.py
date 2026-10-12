"""
extract_features.py (v2) — Stage 1: video -> per-frame gaze/head-pose/blendshape time series.

Batch, resumable, idempotent. For each video under data/raw/{cheating,clean}/ it
writes one parquet to data/features/<label>__<videoname>.parquet with the per-frame
schema documented in common.py.

Usage:
    python src/extract_features.py                       # extract everything (skip done)
    python src/extract_features.py --backend mediapipe+l2cs
    python src/extract_features.py --fps 15 --force
    python src/extract_features.py --video path/to/one.mp4 --label cheating
    python src/extract_features.py --selftest            # signal-only self test (no video)

A per-video "% frames with a detected face" is logged; videos below
cfg.min_face_ratio are flagged for manual review in the summary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    Config, DEFAULT_CONFIG, RAW_DIR, FEATURES_DIR, MODELS_DIR,
    make_backends, channels_for_backend, run_selftest,
)

VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v"}


def discover_videos(explicit_video: str = None, explicit_label: str = None):
    """Yield (video_path, label, source) triples. label in {'cheating','clean'}.

    Layout: data/raw/<label>/<source>/videos... — the source subfolder names the
    recording batch/setup (used by the leave-one-source-pair-out evaluation).
    Videos placed directly under data/raw/<label>/ get source='default'.
    """
    if explicit_video:
        label = explicit_label or "unknown"
        yield Path(explicit_video), label, "default"
        return
    for label in ("cheating", "clean"):
        d = RAW_DIR / label
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in VIDEO_EXTS:
                yield p, label, "default"
            elif p.is_dir():
                source = p.name.strip().replace("__", "_")
                for v in sorted(p.rglob("*")):
                    if v.suffix.lower() in VIDEO_EXTS and v.is_file():
                        yield v, label, source


def output_path(video: Path, label: str, source: str = "default",
                out_dir: Path = None) -> Path:
    safe = video.stem.replace("/", "_").replace("__", "_")
    src = (source or "default").replace("/", "_").replace("__", "_")
    return (out_dir or FEATURES_DIR) / f"{label}__{src}__{safe}.parquet"


def extract_one(video: Path, label: str, cfg: Config, backend: str,
                args_max_seconds: float = None, out_dir: Path = None,
                source: str = "default", progress=None, workers: int = 1) -> dict:
    """Extract one video to parquet. Returns a summary dict.

    `progress(stage, seconds)`, if given, gets ("frames", t) every 15 kept frames;
    it may raise to abort the extraction (the web app uses this to cancel a job).
    `workers` > 1 runs MediaPipe on that many threads — identical output, faster."""
    import cv2

    mp_backend, app_backend = make_backends(backend, cfg, workers)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        mp_backend.close()
        return dict(video=str(video), label=label, status="error_open", n_frames=0, face_ratio=0.0)

    # Sanitize the reported fps. Many webm recordings carry bogus headers
    # (e.g. 1000 fps) or are variable-frame-rate, so we sample by *timestamp*
    # (CAP_PROP_POS_MSEC) and fall back to a frame counter only if timestamps
    # are missing/non-monotonic. This keeps time-based features correct.
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps_sane = bool(src_fps and 1.0 < src_fps <= 120.0 and not np.isnan(src_fps))
    fallback_fps = src_fps if fps_sane else 30.0
    target_dt = 1.0 / cfg.target_fps

    rows = []
    frame_idx = 0
    kept = 0
    next_t = 0.0
    last_msec = -1.0
    used_timestamps = False
    t0 = time.time()
    while True:
        ok = cap.grab()
        if not ok:
            break
        msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if msec and msec > 0 and msec + 1e-6 >= last_msec:
            t = msec / 1000.0
            last_msec = msec
            used_timestamps = True
        else:
            t = frame_idx / fallback_fps
        if args_max_seconds is not None and t > args_max_seconds:
            break
        # v2: grid-anchored keep rule with a quarter-frame tolerance. v1's
        # `t >= next_t, next_t = t + dt` dropped every other frame of a 25 fps source
        # whenever a timestamp landed a hair before the grid (12.5 fps effective).
        if t + 0.25 * target_dt >= next_t:
            ok, frame = cap.retrieve()
            if not ok:
                break
            rec = mp_backend.process(frame)
            rec["t"] = t
            rec["frame_idx"] = frame_idx
            if app_backend is not None:
                # Only run the (expensive) appearance model when a face was found.
                if rec.get("face_found", 0):
                    rec.update(app_backend.process(frame))
                else:
                    rec.update({c: np.nan for c in app_backend.channels})
            rows.append(rec)
            kept += 1
            next_t = max(next_t + target_dt, t + 0.75 * target_dt)
            if progress is not None and kept % 15 == 0:
                progress("frames", t)
        frame_idx += 1

    cap.release()
    mp_backend.close()

    if not rows:
        return dict(video=str(video), label=label, status="no_frames", n_frames=0, face_ratio=0.0)

    df = pd.DataFrame(rows)
    cols = ["t", "frame_idx", "face_found"] + channels_for_backend(backend, cfg) + ["blink"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    # float32 keeps the 25 fps + blendshape parquets small; `t` stays float64
    f32 = [c for c in cols if c not in ("t", "frame_idx", "face_found", "blink")]
    df[f32] = df[f32].astype(np.float32)

    face_ratio = float(df["face_found"].mean())
    out = output_path(video, label, source, out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.attrs = {}
    df.to_parquet(out, index=False)

    elapsed = time.time() - t0
    span = max(float(df["t"].iloc[-1]), 1e-6)
    return dict(
        video=str(video), label=label, source=source, status="ok", out=str(out),
        n_frames=len(df), face_ratio=face_ratio,
        src_fps=round(float(src_fps or 0), 2), timestamps=used_timestamps,
        video_seconds=round(span, 1), sampled_fps=round(len(df) / span, 2),
        seconds=round(elapsed, 1),
    )


def _extract_worker(job):
    """Top-level (picklable) worker for parallel extraction across videos."""
    video, label, cfg, backend, max_seconds, source = job
    try:
        return extract_one(video, label, cfg, backend, args_max_seconds=max_seconds,
                           source=source)
    except Exception as e:
        return dict(video=str(video), label=label, source=source,
                    status=f"error:{type(e).__name__}:{e}", n_frames=0, face_ratio=0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 1: extract per-frame gaze time series from videos.")
    ap.add_argument("--backend", default=None,
                    help="mediapipe | mediapipe+l2cs | mediapipe+openface3 (default: config)")
    ap.add_argument("--fps", type=float, default=None, help="target sampling fps (default: config)")
    ap.add_argument("--video", default=None, help="extract a single video instead of the raw/ tree")
    ap.add_argument("--label", default=None, help="label for --video (cheating|clean|unknown)")
    ap.add_argument("--force", action="store_true", help="re-extract even if parquet exists")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="only process the first N seconds of each video (quick tests)")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel videos to extract at once (default: ~90%% of cores)")
    ap.add_argument("--selftest", action="store_true", help="run signal-only self test and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return 0 if run_selftest() else 1

    cfg = DEFAULT_CONFIG
    if args.fps:
        cfg.target_fps = args.fps
    backend = (args.backend or cfg.backend).lower()
    cfg.backend = backend

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Persist the config used for extraction (feeds later stages + JS port).
    cfg.to_json(MODELS_DIR / "config.json")

    videos = list(discover_videos(args.video, args.label))
    if not videos:
        print(f"No videos found. Drop files into {RAW_DIR}/cheating and {RAW_DIR}/clean.")
        return 1

    # Build the to-do list (skip already-extracted videos unless --force).
    todo = []
    for video, label, source in videos:
        out = output_path(video, label, source)
        if out.exists() and not args.force:
            print(f"SKIP (done): {video.name}")
        else:
            todo.append((video, label, cfg, backend, args.max_seconds, source))

    # v2: videos that have a manual reading review go first (they drive training).
    import re
    from common import LABELS_DIR
    reviewed = {m.group(1) for f in LABELS_DIR.glob("*.csv")
                for m in [re.search(r"ir(\d+)", f.name)] if m}
    todo.sort(key=lambda j: not any(f"ir{ir}" in j[0].name for ir in reviewed))

    # Workers: each MediaPipe process uses ~1.3 cores, so cap to keep ~1 core
    # free (target ~90% CPU) and avoid oversubscription.
    import os
    n_cpu = os.cpu_count() or 4
    default_workers = max(1, min(len(todo), int((n_cpu - 1) / 1.3) or 1))
    workers = args.workers if args.workers else default_workers
    workers = max(1, min(workers, len(todo))) if todo else 1

    print(f"Backend={backend}  target_fps={cfg.target_fps}  videos={len(videos)}  "
          f"to-extract={len(todo)}  workers={workers}")
    summaries = []
    flagged = []

    def _record(s, i):
        summaries.append(s)
        status = s.get("status")
        fr = s.get("face_ratio", 0.0)
        print(f"[{i}/{len(todo)}] {status}  face={fr:.0%}  frames={s.get('n_frames')}"
              f"  ({s.get('seconds','?')}s)  [{s.get('source','?')}] "
              f"{Path(s['video']).name}", flush=True)
        if status == "ok" and fr < cfg.min_face_ratio:
            flagged.append((Path(s["video"]).name, fr))

    if workers <= 1:
        for i, job in enumerate(todo, 1):
            _record(_extract_worker(job), i)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_extract_worker, job): job for job in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                _record(fut.result(), i)

    print("\n=== Stage 1 summary ===")
    ok = [s for s in summaries if s.get("status") == "ok"]
    print(f"extracted: {len(ok)}  |  errors/empty: {len(summaries) - len(ok)}")
    if flagged:
        print(f"\n⚠️  {len(flagged)} video(s) below {cfg.min_face_ratio:.0%} face-found — review manually:")
        for name, fr in flagged:
            print(f"    {fr:5.0%}  {name}")
    if summaries:
        pd.DataFrame(summaries).to_csv(FEATURES_DIR / "_extract_summary.csv", index=False)
        print(f"\nsummary -> {FEATURES_DIR / '_extract_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
