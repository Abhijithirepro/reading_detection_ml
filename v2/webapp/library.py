"""
library.py (v2) — the pre-scored candidate library: every recording in the three
on-disk folders scored ahead of time, so the page opens a candidate instantly
(video, reading timeline, reasons while it plays, detailed report) without
re-running anything.

Scores are OUT-OF-FOLD: each video's per-second reading probability comes from the
cross-validation run that shipped the model (v2/results/oof_seconds__<exp>.npz),
i.e. from a model that never trained on that video, so the verdicts match
v2/results/ALL_VIDEOS_RESULTS.md. The shipped model's own verdict (it trained on
these videos, so it is in-sample) is kept alongside for reference.

Nothing is re-extracted: the per-frame features cached in v2/data/features and the
window features in v2/data/windows are reused, and the result is built by
predict.result_from_scores, the same code that scores an upload. WebM recordings
are remuxed (stream copy, no re-encode) into library/media/ with a seek index —
MediaRecorder files carry none, and browsers can't seek in them without it.

Usage (from the project root):
    venv/bin/python v2/webapp/library.py                   # build everything not built yet
    venv/bin/python v2/webapp/library.py --force           # rebuild results + reports
    venv/bin/python v2/webapp/library.py --only ir8266049  # one candidate
    venv/bin/python v2/webapp/library.py --no-media        # skip the WebM remux

Output (git-ignored):
    library/index.json                 one row per candidate (the page's list)
    library/<id>/payload.json          the page's result payload (same shape as a job's)
    library/<id>/traces.json           eye / head / mouth traces for the live chart
    library/<id>/*__report.html        the detailed report (report.py)
    library/media/<id>.webm            seekable copy of a WebM recording
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parent
V2_DIR = WEBAPP_DIR.parent
PROJECT_DIR = V2_DIR.parent
LIB_DIR = WEBAPP_DIR / "library"
MEDIA_DIR = LIB_DIR / "media"

# (folder under the project root, label, source name used by v2/data/features)
FOLDERS = [
    ("cheating_folder", "cheating", "interviews"),
    ("not_cheating_folder", "clean", "interviews"),
    ("new_not_cheating", "clean", "interviews_new"),
]
VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v"}
ID_RE = re.compile(r"[a-z0-9_-]{1,64}")


def candidate_id(stem: str) -> str:
    m = re.search(r"ir(\d{5,})", stem, re.IGNORECASE)
    return f"ir{m.group(1)}" if m else re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")[:64]


def candidate_name(stem: str) -> str:
    """'combined_ir8266049_Sai_Koti_Reddy_Sai_Bhavanam (1)' -> 'Sai Koti Reddy Sai Bhavanam'."""
    name = re.sub(r"^combined_", "", stem)
    name = re.sub(r"ir\d{5,}_?", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s*\(\d+\)$", "", name)
    return name.replace("_", " ").strip() or stem


def discover() -> list:
    """[(video, folder, label, source)] for every recording in the three folders."""
    out = []
    for folder, label, source in FOLDERS:
        d = PROJECT_DIR / folder
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                out.append((p, folder, label, source))
    return out


# --------------------------------------------------------------------------- #
# Build (heavy imports live inside the workers)
# --------------------------------------------------------------------------- #
_W = {}


def _init_worker():
    sys.path.insert(0, str(V2_DIR / "src"))
    sys.path.insert(0, str(WEBAPP_DIR))
    import warnings
    warnings.filterwarnings("ignore")
    import engine as eng          # payload helpers (bins, review comparison, traces)
    import predict
    import report
    import labels as lab
    from common import Config, DEFAULT_CONFIG, MODELS_DIR
    cfg = DEFAULT_CONFIG
    if (MODELS_DIR / "config.json").exists():
        cfg = Config.from_json(MODELS_DIR / "config.json")
    art, scorer = predict.load_winner()
    _W.update(eng=eng, predict=predict, report=report, lab=lab, cfg=cfg, art=art, scorer=scorer,
              reviews=lab.load_reviews())


def _build_one(job: dict) -> dict:
    import numpy as np
    import pandas as pd
    if not _W:
        _init_worker()
    eng, predict, report, lab = _W["eng"], _W["predict"], _W["report"], _W["lab"]
    art, cfg = _W["art"], _W["cfg"]
    import build_windows as bw
    import reading
    from common import FEATURES_DIR

    t_start = time.time()
    video = Path(job["video"])
    cid, label = job["id"], job["label"]
    item_dir = LIB_DIR / cid
    item_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(FEATURES_DIR / f"{job['key']}.parquet")
    clean = bw.clean_video(df, cfg)
    win_df = job["windows"].sort_values("win_start_s").reset_index(drop=True)
    sec_thr = float(art["second_threshold"])

    # Out-of-fold result (what the page shows), with the verdict statistic on the
    # window-based speaking the threshold was tuned on (train.per_second_tables) ...
    oof = np.asarray(job["oof"], dtype=float)
    speak = reading.per_second(win_df["win_start_s"], win_df["win_end_s"], win_df["speak_frac"],
                               len(oof), smooth_s=1, center_radius_s=0.5)
    result = predict.result_from_scores(video, oof, win_df, clean, art, sec_thr, stat_speaking=speak)
    result["video"] = video.name
    result["scoring"] = "out-of-fold"
    result["model"]["experiment"] = f"{art.get('experiment')} · out-of-fold"

    # ... and the shipped model's in-sample result, for reference.
    probs = np.asarray(_W["scorer"](win_df), dtype=float)
    n_sec = int(np.ceil(win_df["win_end_s"].max()))
    ins = predict.result_from_scores(
        video, reading.per_second(win_df["win_start_s"], win_df["win_end_s"], probs, n_sec,
                                  art.get("smooth_s", 3)), win_df, clean, art, sec_thr,
        stat_speaking=reading.per_second(win_df["win_start_s"], win_df["win_end_s"],
                                         win_df["speak_frac"], n_sec, smooth_s=1, center_radius_s=0.5))
    shipped = dict(verdict=ins["verdict"], video_stat_value=ins["video_stat_value"],
                   reading_seconds=ins["reading_seconds"], reading_segments=ins["reading_segments"])

    # Detailed report: report.py reads <stem>__timeline.npz from its REPORTS_DIR.
    # The per-window probabilities in the timeline are unused by the report, and
    # only per-second out-of-fold scores exist, so they are left empty.
    predict.REPORTS_DIR = report.REPORTS_DIR = item_dir
    win_df["prob"] = np.nan
    predict.save_timeline(video, result, win_df, clean)
    review = lab.review_for(video.name, _W["reviews"])
    report.build_video_report(video, review=review)
    (item_dir / f"{video.stem}__timeline.npz").unlink(missing_ok=True)    # only the report needs it
    report_file = f"{video.stem}__report.html"

    span = float(df["t"].iloc[-1]) if len(df) else 0.0
    correct = (result["verdict"] == "CHEATING") == (label == "cheating")
    cand = dict(
        id=cid, name=candidate_name(video.stem), ir=lab.ir_of(video.stem), file=video.name,
        folder=job["folder"], actual=label, reviewed=review is not None, correct=correct,
        scoring="out-of-fold", experiment=art.get("experiment"), shipped=shipped)
    payload = dict(
        version=2, job_id=cid, source=video.name, result=result, candidate=cand,
        windows=eng.bins(result, result["flagged_segments"]),
        stats=dict(
            n_frames=int(len(df)), face_ratio=round(float(df["face_found"].mean()), 4),
            video_seconds=round(span, 1), analysed_seconds=round(float(win_df["win_end_s"].iloc[-1]), 1),
            sampled_fps=round(len(df) / span, 2) if span > 0 else None,
            processing_seconds=None, speed_x_realtime=None, max_seconds=None, t0=0.0,
            stride_seconds=eng.BIN_SECONDS, bin_seconds=eng.BIN_SECONDS),
        runtime=dict(mediapipe="cpu", cnn="cpu (xgboost)", mediapipe_threads=None),
        report_url=f"/api/library/{cid}/report",
        segments_csv_url=f"/api/library/{cid}/segments.csv",
        traces_url=f"/api/library/{cid}/traces",
        video_url=f"/api/library/{cid}/video",
        created=time.time(),
    )
    if review is not None:
        payload["review"] = eng.compare_with_review(result, review)
        payload["review"]["origin"] = "manual_review_for_candidate/ (matched by IR number)"
    (item_dir / "payload.json").write_text(json.dumps(payload))
    (item_dir / "traces.json").write_text(json.dumps(eng.traces(clean)))

    return dict(
        **{k: cand[k] for k in ("id", "name", "ir", "file", "folder", "actual", "reviewed", "correct")},
        verdict=result["verdict"], score=result["video_stat_value"],
        reading_seconds=result["reading_seconds"], segments=result["reading_segments"],
        speaking_seconds=result["speaking_seconds"], duration=round(span, 1),
        face_ratio=payload["stats"]["face_ratio"], shipped_verdict=shipped["verdict"],
        report=report_file, source_path=str(video), build_seconds=round(time.time() - t_start, 1))


# --------------------------------------------------------------------------- #
# Media
# --------------------------------------------------------------------------- #
def remux(src: Path, dst: Path) -> str:
    """Stream-copy a WebM into a seekable one (duration + cues up front)."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return "kept"
    tmp = dst.with_suffix(".part.webm")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-map", "0", "-c", "copy",
           "-cues_to_front", "1", "-f", "webm", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return "failed: " + (r.stderr.strip().splitlines() or ["ffmpeg error"])[-1]
    tmp.replace(dst)
    return "remuxed"


def media_seconds(path: Path):
    """Container duration (what the browser's player shows), or None."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                        str(path)], capture_output=True, text=True)
    try:
        return round(float(r.stdout.strip()), 1)
    except ValueError:
        return None


def media_for(row: dict, use_media: bool) -> Path:
    src = Path(row["source_path"])
    if use_media and src.suffix.lower() == ".webm":
        dst = MEDIA_DIR / f"{row['id']}.webm"
        if dst.exists():
            return dst
    return src


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build(force=False, only=None, workers=None, media=True) -> int:
    sys.path.insert(0, str(V2_DIR / "src"))
    import numpy as np
    import pandas as pd
    from common import WINDOWS_DIR, RESULTS_DIR, MODELS_DIR

    art = json.loads((MODELS_DIR / "model.json").read_text())
    exp = art.get("experiment")
    oof_path = RESULTS_DIR / f"oof_seconds__{exp}.npz"
    if not oof_path.exists():
        print(f"No out-of-fold scores for the shipped model ({oof_path.name}). "
              f"Re-run v2/src/train.py so it writes them.")
        return 1
    oof = np.load(oof_path)
    oof_csv = pd.read_csv(RESULTS_DIR / f"oof_video_scores__{exp}.csv", float_precision="round_trip").set_index("video")

    found = discover()
    if only:
        found = [f for f in found if candidate_id(f[0].stem) in only or f[0].stem in only]
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    index_path = LIB_DIR / "index.json"
    old = {}
    if index_path.exists():
        old = {r["id"]: r for r in json.loads(index_path.read_text()).get("candidates", [])}

    print(f"{len(found)} recordings in {', '.join(f for f, *_ in FOLDERS)}")
    windows = pd.read_parquet(WINDOWS_DIR / "windows.parquet")
    by_video = dict(tuple(windows.groupby("video", sort=False)))

    jobs, rows, skipped = [], dict(old), []
    for video, folder, label, source in found:
        cid = candidate_id(video.stem)
        key = f"{label}__{source}__{video.stem}"
        if key not in by_video or video.stem not in oof.files:
            skipped.append(f"{video.name}: no cached features/out-of-fold scores (run extract + train)")
            continue
        if not force and cid in old and (LIB_DIR / cid / "payload.json").exists():
            continue
        jobs.append(dict(id=cid, key=key, video=str(video), folder=folder, label=label,
                         windows=by_video[key], oof=oof[video.stem]))
    del windows

    t0 = time.time()
    if jobs:
        n = workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
        print(f"scoring {len(jobs)} candidate(s) on {n} process(es)...")
        with ProcessPoolExecutor(max_workers=n, initializer=_init_worker) as pool:
            futs = {pool.submit(_build_one, j): j for j in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                j = futs[fut]
                try:
                    row = fut.result()
                except Exception as e:      # keep the batch going; report at the end
                    skipped.append(f"{Path(j['video']).name}: {type(e).__name__}: {e}")
                    continue
                rows[row["id"]] = row
                print(f"  [{i}/{len(jobs)}] {row['id']:<11} {row['verdict']:<8} "
                      f"reading {row['reading_seconds']:6.0f}s  {row['name']}", flush=True)

    if media:
        webms = [r for r in rows.values() if r["source_path"].lower().endswith(".webm")
                 and (not only or r["id"] in only or Path(r["source_path"]).stem in only)]
        if webms:
            MEDIA_DIR.mkdir(parents=True, exist_ok=True)
            print(f"making {len(webms)} WebM recording(s) seekable (stream copy)...")
            with ThreadPoolExecutor(max_workers=4) as pool:
                futs = {pool.submit(remux, Path(r["source_path"]), MEDIA_DIR / f"{r['id']}.webm"): r
                        for r in webms}
                for fut in as_completed(futs):
                    status = fut.result()
                    if status.startswith("failed"):
                        skipped.append(f"{futs[fut]['file']}: remux {status} (the original is served)")

    # index: rows of recordings still on disk, cheating first, highest score first
    on_disk = {candidate_id(v.stem) for v, *_ in discover()}
    cands = sorted((r for r in rows.values() if r["id"] in on_disk),
                   key=lambda r: (r["actual"] != "cheating", -r["score"]))
    for r in cands:
        key = next((k for k in oof_csv.index if k.endswith("__" + Path(r["source_path"]).stem)), None)
        r["oof_csv_score"] = round(float(oof_csv.loc[key, "score"]), 4) if key else None
    # The player's duration vs the analysed span: decoding stopped early on some
    # recordings, and the page marks the un-analysed tail.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for r, secs in zip(cands, pool.map(lambda r: media_seconds(media_for(r, media)), cands)):
            r["media_seconds"] = secs
    cheat = [r for r in cands if r["actual"] == "cheating"]
    clean = [r for r in cands if r["actual"] == "clean"]
    index = dict(
        built=time.time(), experiment=exp, scoring="out-of-fold",
        video_stat=art.get("video_stat"), video_threshold=art["video_threshold"],
        second_threshold=art["second_threshold"],
        summary=dict(n=len(cands), cheating=len(cheat), clean=len(clean),
                     caught=sum(r["verdict"] == "CHEATING" for r in cheat),
                     cleared=sum(r["verdict"] == "clean" for r in clean)),
        folders=[dict(folder=f, actual=lab_, n=sum(r["folder"] == f for r in cands))
                 for f, lab_, _ in FOLDERS],
        candidates=cands)
    index_path.write_text(json.dumps(index, indent=1))

    s = index["summary"]
    print(f"\nlibrary: {s['n']} candidates  ·  cheating caught {s['caught']}/{s['cheating']}  ·  "
          f"clean cleared {s['cleared']}/{s['clean']}  ({time.time() - t0:.0f}s)")
    drift = [r for r in cands if r["oof_csv_score"] is not None
             and (r["score"] >= art["video_threshold"]) != (r["oof_csv_score"] >= art["video_threshold"])]
    for r in drift:
        print(f"  note: {r['name']} verdict differs from ALL_VIDEOS_RESULTS "
              f"(score {r['score']:.3f} here vs {r['oof_csv_score']:.3f})")
    for msg in skipped:
        print(f"  skipped: {msg}")
    print(f"-> {index_path}")
    return 0


# --------------------------------------------------------------------------- #
# Read side (used by server.py)
# --------------------------------------------------------------------------- #
class Library:
    """Serves the built library. Re-reads index.json when it changes on disk."""

    PUBLIC = ("id", "name", "ir", "file", "folder", "actual", "reviewed", "correct", "verdict",
              "score", "reading_seconds", "segments", "speaking_seconds", "duration",
              "media_seconds", "face_ratio", "shipped_verdict")

    def __init__(self, root: Path = LIB_DIR):
        self.root = root
        self._index, self._mtime = None, None

    def _load(self):
        path = self.root / "index.json"
        if not path.exists():
            self._index, self._mtime = None, None
            return None
        mtime = path.stat().st_mtime
        if mtime != self._mtime:
            self._index, self._mtime = json.loads(path.read_text()), mtime
        return self._index

    def index(self) -> dict:
        idx = self._load()
        if idx is None:
            return dict(built=None, candidates=[], summary=None,
                        hint="Not built yet: venv/bin/python v2/webapp/library.py")
        out = {k: v for k, v in idx.items() if k != "candidates"}
        out["candidates"] = [{k: r.get(k) for k in self.PUBLIC} for r in idx["candidates"]]
        return out

    def _row(self, cid: str):
        if not ID_RE.fullmatch(cid or ""):
            return None
        idx = self._load()
        return next((r for r in idx["candidates"] if r["id"] == cid), None) if idx else None

    def payload(self, cid: str):
        row = self._row(cid)
        path = self.root / cid / "payload.json" if row else None
        return json.loads(path.read_text()) if path and path.exists() else None

    def report_path(self, cid: str):
        row = self._row(cid)
        path = self.root / cid / row["report"] if row else None
        return path if path and path.exists() else None

    def traces_path(self, cid: str):
        row = self._row(cid)
        path = self.root / cid / "traces.json" if row else None
        return path if path and path.exists() else None

    def video_path(self, cid: str):
        row = self._row(cid)
        if row is None:
            return None
        path = media_for(row, use_media=True)
        return path if path.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pre-score every on-disk recording for the web app.")
    ap.add_argument("--force", action="store_true", help="rebuild candidates already in the library")
    ap.add_argument("--only", nargs="+", default=None, help="candidate ids (ir8266049) or file stems")
    ap.add_argument("--workers", type=int, default=None, help="scoring processes (default: cores - 1)")
    ap.add_argument("--no-media", action="store_true", help="skip making WebM recordings seekable")
    args = ap.parse_args(argv)
    return build(force=args.force, only=set(args.only) if args.only else None,
                 workers=args.workers, media=not args.no_media)


if __name__ == "__main__":
    sys.exit(main())
