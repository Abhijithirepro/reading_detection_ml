#!/usr/bin/env python3
"""
build_static.py — fills this folder with the pre-scored candidate library so the page
runs on any plain web server (nginx, Apache, `python3 serve.py`): no model, no
MediaPipe, no pip install.

It READS the library built by v2/webapp/library.py (read-only) and writes only
inside this folder:

    data/info.json               model facts for the header      (was GET /api/info)
    data/index.json              the candidate list              (was GET /api/library)
    data/<id>/payload.json       one candidate's result          (was GET /api/library/<id>)
    data/<id>/traces.json        eye / head / mouth traces       (was .../traces)
    data/<id>/report.html        the detailed report             (was .../report)
    data/<id>/segments.csv       reading segments, review format (was .../segments.csv)
    videos/<id>.mp4|.webm        the recording                   (was .../video)

Every URL written into the data is relative, so the folder also works from a
sub-path (https://host/preprocess_report_webapp/). Local file paths are left out.
Videos are copied once (APFS clones on macOS: no extra disk until changed); a video
already here with the same size is skipped.

Only needed again if the library is rebuilt — NOT needed on the server:
    python3 v2/preprocess_report_webapp/build_static.py
    python3 v2/preprocess_report_webapp/build_static.py --no-videos     # data only
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIBRARY = HERE.parent / "webapp" / "library"          # built by v2/webapp/library.py
MODEL = HERE.parent / "models" / "model.json"         # window / stride for the header

# the fields the original server publishes for each candidate (webapp/library.py Library.PUBLIC)
PUBLIC = ("id", "name", "ir", "file", "folder", "actual", "reviewed", "correct", "verdict",
          "score", "reading_seconds", "segments", "speaking_seconds", "duration",
          "media_seconds", "face_ratio", "shipped_verdict")


def mmss(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def segments_csv(result: dict) -> str:
    """Reading segments in the reviewers' sheet format (same as webapp/engine.py)."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["start time (min:sec)", "End time (min:sec)", "supsicious/not suspsicious ", "reason "])
    for s in result["flagged_segments"]:
        w.writerow([mmss(s["start_s"]), mmss(s["end_s"]), "reading",
                    f"{s['reason']} (model peak {s['peak_prob']:.2f})"])
    return buf.getvalue()


def video_source(row: dict) -> Path:
    """The file the original server streams: the seekable WebM copy if there is one,
    else the recording itself (webapp/library.py media_for)."""
    src = Path(row["source_path"])
    media = LIBRARY / "media" / f"{row['id']}.webm"
    return media if src.suffix.lower() == ".webm" and media.exists() else src


def copy_video(src: Path, dst: Path) -> str:
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return "kept"
    tmp = dst.with_name(dst.name + ".part")
    tmp.unlink(missing_ok=True)
    if sys.platform == "darwin" and subprocess.run(["cp", "-c", str(src), str(tmp)],
                                                   capture_output=True).returncode == 0:
        how = "cloned"
    else:
        shutil.copyfile(src, tmp)
        how = "copied"
    tmp.replace(dst)
    return how


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def build(videos: bool = True) -> int:
    index_path = LIBRARY / "index.json"
    if not index_path.exists():
        sys.exit(f"No library at {LIBRARY} — build it first: venv/bin/python v2/webapp/library.py")
    idx = json.loads(index_path.read_text())
    art = json.loads(MODEL.read_text()) if MODEL.exists() else {}
    data, vids = HERE / "data", HERE / "videos"
    data.mkdir(exist_ok=True)
    vids.mkdir(exist_ok=True)

    write_json(data / "info.json", dict(
        static=True, built=idx.get("built"),
        model=dict(kind=art.get("kind", "reading_xgb"), experiment=idx.get("experiment"),
                   second_threshold=idx.get("second_threshold"), video_stat=idx.get("video_stat"),
                   video_threshold=idx.get("video_threshold"),
                   window_seconds=art.get("read_window_seconds", 8.0),
                   stride_seconds=art.get("read_stride_seconds", 1.0)),
        runtime=dict(mediapipe="pre-computed", mediapipe_threads=0, cpu_count=0,
                     cnn="xgboost · out-of-fold", workers=1, cpu="pre-scored"),
        accepted=[]))

    rows, problems, counts, t0 = [], [], {}, time.time()
    for k, row in enumerate(idx["candidates"], 1):
        cid, src_dir = row["id"], LIBRARY / row["id"]
        out = data / cid
        out.mkdir(exist_ok=True)
        payload = json.loads((src_dir / "payload.json").read_text())
        video = video_source(row)
        if not video.exists():
            problems.append(f"{cid}: recording missing ({video.name})")
        payload.update(
            video_url=f"videos/{cid}{video.suffix.lower()}" if video.exists() else None,
            traces_url=f"data/{cid}/traces.json" if (src_dir / "traces.json").exists() else None,
            report_url=f"data/{cid}/report.html" if (src_dir / row["report"]).exists() else None,
            segments_csv_url=f"data/{cid}/segments.csv")
        write_json(out / "payload.json", payload)
        (out / "segments.csv").write_text(segments_csv(payload["result"]), encoding="utf-8")
        for name, dst in (("traces.json", "traces.json"), (row["report"], "report.html")):
            if (src_dir / name).exists():
                shutil.copyfile(src_dir / name, out / dst)
        if videos and video.exists():
            how = copy_video(video, vids / f"{cid}{video.suffix.lower()}")
            counts[how] = counts.get(how, 0) + 1
        rows.append({key: row.get(key) for key in PUBLIC})
        print(f"\r  {k}/{len(idx['candidates'])} {cid}", end="", flush=True)

    write_json(data / "index.json", {**{k: v for k, v in idx.items() if k != "candidates"},
                                     "candidates": rows})
    print(f"\n-> {len(rows)} candidates in {data}  videos: {counts or 'skipped'}  "
          f"({time.time() - t0:.0f} s)")
    for p in problems:
        print("  !", p)
    return 1 if problems else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--no-videos", action="store_true", help="refresh data/ only, leave videos/ as is")
    return build(videos=not ap.parse_args(argv).no_videos)


if __name__ == "__main__":
    sys.exit(main())
