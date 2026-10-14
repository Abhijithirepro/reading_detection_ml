"""
engine.py (v2) — the analysis side of the web app: a small in-process job queue
that runs uploaded videos through the v2 pipeline (v2/src/).

Nothing here re-implements the maths. Every job goes through predict.score_video,
the same function the CLI uses, so a video scored here gets exactly the result
`python v2/src/predict.py <video>` would give it.

v2 additions:
  * the timeline is PER SECOND (reading probability + speaking), not per 20 s window
  * when the uploaded filename carries an IR number that has a manual review in
    manual_review_for_candidate/, the result is compared against it automatically;
    a review CSV can also be uploaded for any finished job (compare_review)
  * detected segments export as CSV in the reviewers' own sheet format

Hardware:
  * MediaPipe face/iris/blendshape tracking runs on the CPU, fanned out over a
    pool of threads (90% of the cores by default) warmed once at start-up.
  * The reading model is a small XGBoost ensemble: CPU, milliseconds per video.
"""

from __future__ import annotations

import csv
import io
import json
import os
import platform
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import warnings
from pathlib import Path

import numpy as np

WEBAPP_DIR = Path(__file__).resolve().parent
ROOT = WEBAPP_DIR.parent                      # .../v2
RUNS_DIR = WEBAPP_DIR / "runs"
UPLOADS_DIR = RUNS_DIR / "uploads"

sys.path.insert(0, str(ROOT / "src"))

import predict  # noqa: E402  (v2/src must be on sys.path first)
import report  # noqa: E402
import labels as lab  # noqa: E402
from evaluate import per_second_metrics  # noqa: E402
from common import (Config, DEFAULT_CONFIG, MODELS_DIR, LABELS_DIR,  # noqa: E402
                    warm_mediapipe, close_mediapipe)

# Point src/'s output directory at runs/ so web jobs never write into v2/reports/
# (and, via predict's cache dir, never into the training set's data/features/).
predict.REPORTS_DIR = RUNS_DIR
report.REPORTS_DIR = RUNS_DIR

VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v"}
BIN_SECONDS = 5          # chart / table granularity (the strip stays per second)
TRACE_HZ = 5             # live signal chart sampling (frames are averaged down to this)

# matplotlib's pyplot (used by report.py) is not thread-safe.
_REPORT_LOCK = threading.Lock()


class JobCancelled(Exception):
    pass


def default_threads() -> int:
    """90% of the logical cores (9 on a 10-core M5), leaving one for the decoder/OS."""
    return max(1, round(0.9 * (os.cpu_count() or 2)))


def _cpu_name() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=2).stdout.strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def _estimate_duration(video: Path):
    """Container duration in seconds, or None when it can't be trusted. Only used
    for the progress bar — webm recordings often carry bogus frame counts/fps."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if not (fps and 1.0 < fps <= 120.0 and n and n > 0):
        return None
    d = n / fps
    return d if 1.0 < d < 6 * 3600 else None


def display_name(filename: str) -> str:
    """Original filename reduced to a safe, display-only form (it ends up in HTML)."""
    name = Path(filename or "video").name
    return re.sub(r"[^A-Za-z0-9 ._()\-]", "_", name)[:120] or "video"


# --------------------------------------------------------------------------- #
# Payload helpers (pure; also used by the review-upload route)
# --------------------------------------------------------------------------- #
def _seconds(result) -> np.ndarray:
    return np.array([np.nan if x is None else x for x in result["seconds"]], dtype=float)


def bins(result, segments, width=BIN_SECONDS):
    """Per-second scores -> fixed bins [{start_s, end_s, prob, reasons}] (video clock)."""
    sec = _seconds(result)
    out = []
    for a in range(0, len(sec), width):
        chunk = sec[a:a + width]
        if not np.isfinite(chunk).any():
            continue
        b = min(a + width, len(sec))
        mid = (a + b) / 2
        seg = next((s for s in segments if s["start_s"] <= mid < s["end_s"]), None)
        out.append(dict(start_s=float(a), end_s=float(b), prob=float(np.nanmax(chunk)),
                        reasons=(seg["reasons"] if seg else [])))
    return out


def compare_with_review(result, review) -> dict:
    """Agreement between the detected reading segments and a manual review."""
    sec = _seconds(result)
    n = len(sec)
    # only the reviewer spans inside the analysed range count ("analyse first N min")
    in_range = [(a, min(b, n), tag, reason) for a, b, tag, reason in review.segments if a < n - 1]
    truth = lab.truth_mask(in_range, n, 1.0)
    flag = np.zeros(n, dtype=bool)
    for s in result["flagged_segments"]:
        flag[int(s["start_s"]):int(s["end_s"])] = True
    m = per_second_metrics(sec, truth, flag, in_range)
    m["spans_outside_analysed_range"] = len(review.segments) - len(in_range)
    spans = []
    for a, b, tag, reason in in_range:
        i0, i1 = int(max(0, a)), int(min(n, b))
        cov = float(flag[i0:i1].mean()) if i1 > i0 else 0.0
        spans.append(dict(start_s=a, end_s=b, range=f"{lab.mmss(a)}–{lab.mmss(b)}",
                          coverage=round(cov, 3), caught=cov >= 0.5, reason=reason))
    extra = []
    for s in result["flagged_segments"]:
        i0, i1 = int(s["start_s"]), int(s["end_s"])
        if i1 > i0 and truth[i0:i1].mean() < 0.2:
            extra.append(dict(start_s=s["start_s"], end_s=s["end_s"], range=s["range"]))
    return dict(
        source=review.path.name, ir=review.ir, spans=spans, extra_segments=extra,
        metrics={k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})


def traces(clean, hz=TRACE_HZ) -> dict:
    """The eye, head and mouth channels averaged down to ~`hz` samples a second for
    the page's live signal chart. Sample i sits at video time t0 + i / hz."""
    dt = float(clean["dt"])
    k = max(1, int(round(1.0 / (hz * dt))))

    def down(x, how="mean"):
        x = np.asarray(x, dtype=float)
        n = len(x) // k * k
        if n == 0:
            return []
        blocks = x[:n].reshape(-1, k)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            v = np.nanmax(blocks, axis=1) if how == "max" else np.nanmean(blocks, axis=1)
        return [None if not np.isfinite(a) else round(float(a), 3) for a in v]

    return dict(
        hz=round(1.0 / (k * dt), 4), t0=round(float(clean["t0"]), 3),
        eye_h=down(clean.get("eye_h", clean["gaze_x"])), eye_v=down(clean.get("eye_v", clean["gaze_y"])),
        yaw=down(clean["yaw"]), mouth=down(clean["mouth"]),
        speaking=down(clean["speaking"]), blink=down(clean["blink"], "max"))


def segments_csv(result) -> str:
    """Detected segments in the reviewers' sheet format (min:sec, like their CSVs)."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["start time (min:sec)", "End time (min:sec)", "supsicious/not suspsicious ", "reason "])
    for s in result["flagged_segments"]:
        w.writerow([lab.mmss(s["start_s"]), lab.mmss(s["end_s"]), "reading",
                    f"{s['reason']} (model peak {s['peak_prob']:.2f})"])
    return buf.getvalue()


class Engine:
    def __init__(self, workers: int = 1, cnn_device: str = "auto", keep_uploads: bool = False,
                 threads: int = None):
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

        self.cfg = DEFAULT_CONFIG
        cfg_path = MODELS_DIR / "config.json"
        if cfg_path.exists():
            self.cfg = Config.from_json(cfg_path)

        self.device = "cpu"     # XGBoost trees; cnn_device kept for CLI compatibility
        self.artifact, self.scorer = predict.load_winner()
        self.keep_uploads = keep_uploads
        self.workers = workers
        # Split the thread budget across parallel jobs so they don't oversubscribe.
        self.threads = threads or default_threads()
        self.threads_per_job = max(1, self.threads // workers)
        if self.threads_per_job > 1:
            warm_mediapipe(self.cfg, self.threads_per_job)

        self._jobs: dict[str, dict] = {}
        self._order: list[str] = []          # queued job ids, for queue position
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        for i in range(workers):
            threading.Thread(target=self._worker, name=f"analysis-{i}", daemon=True).start()

    # ------------------------------------------------------------------ info
    def info(self) -> dict:
        a = self.artifact
        cv = a.get("cv", {})
        return dict(
            model=dict(kind=a["kind"], version=2, threshold=a["second_threshold"],
                       second_threshold=a["second_threshold"],
                       video_stat=a.get("video_stat"), video_threshold=a["video_threshold"],
                       window_seconds=a["read_window_seconds"], stride_seconds=a["read_stride_seconds"],
                       experiment=a.get("experiment"),
                       cv_metrics=dict(video=cv.get("video", {}), per_second=cv.get("per_second", []))),
            runtime=dict(mediapipe="cpu", cnn="cpu (xgboost)", workers=self.workers,
                         mediapipe_threads=self.threads_per_job, cpu_count=os.cpu_count(),
                         cpu=_cpu_name(), torch="-", mps_available=False, cuda_available=False),
            accepted=sorted(VIDEO_EXTS),
            reviews=sorted(r.path.name for r in lab.load_reviews().values()),
        )

    def shutdown(self):
        """Close the warm MediaPipe pool before the interpreter tears down."""
        close_mediapipe()

    # ------------------------------------------------------------------ jobs
    def submit(self, upload_path: Path, filename: str, max_seconds) -> str:
        job_id = upload_path.stem
        job = dict(
            id=job_id, source=display_name(filename), status="queued", stage="queued",
            seconds_done=0.0, duration=None, max_seconds=max_seconds,
            created=time.time(), started=None, finished=None,
            error=None, cancel=False, upload=str(upload_path),
        )
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
        self._queue.put(job_id)
        return job_id

    @staticmethod
    def new_upload_path(filename: str) -> Path:
        ext = Path(filename or "").suffix.lower()
        if ext not in VIDEO_EXTS:
            raise ValueError(f"Unsupported file type '{ext or '(none)'}'. "
                             f"Use one of: {', '.join(sorted(VIDEO_EXTS))}")
        return UPLOADS_DIR / f"{uuid.uuid4().hex[:16]}{ext}"

    def status(self, job_id: str):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                out = {k: v for k, v in job.items() if k not in ("upload", "cancel")}
                if job["status"] == "queued":
                    out["queue_position"] = self._order.index(job_id) + 1
                return out
        # Not in memory (e.g. the server restarted): fall back to a finished payload.
        payload = self.payload(job_id)
        if payload is None:
            return None
        return dict(id=job_id, source=payload["source"], status="done", stage="done",
                    result=payload)

    def payload(self, job_id: str):
        if not re.fullmatch(r"[0-9a-f]{16}", job_id or ""):
            return None
        path = RUNS_DIR / f"{job_id}__web.json"
        return json.loads(path.read_text()) if path.exists() else None

    def _save_payload(self, job_id: str, payload: dict):
        (RUNS_DIR / f"{job_id}__web.json").write_text(json.dumps(payload, indent=2))
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.get("status") == "done":
                job["result"] = payload

    def video_path(self, job_id: str):
        """The kept upload for a job (only exists with --keep-uploads)."""
        if not re.fullmatch(r"[0-9a-f]{16}", job_id or ""):
            return None
        return next((p for p in UPLOADS_DIR.glob(f"{job_id}.*") if p.suffix in VIDEO_EXTS), None)

    def report_path(self, job_id: str):
        if not re.fullmatch(r"[0-9a-f]{16}", job_id or ""):
            return None
        path = RUNS_DIR / f"{job_id}__report.html"
        return path if path.exists() else None

    def traces_path(self, job_id: str):
        if not re.fullmatch(r"[0-9a-f]{16}", job_id or ""):
            return None
        path = RUNS_DIR / f"{job_id}__traces.json"
        return path if path.exists() else None

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["status"] in ("done", "error", "cancelled"):
                return False
            job["cancel"] = True
            if job["status"] == "queued":   # never started: drop it right away
                job.update(status="cancelled", stage="cancelled", finished=time.time())
                if job_id in self._order:
                    self._order.remove(job_id)
        return True

    # ------------------------------------------------------ manual reviews
    def compare_review(self, job_id: str, csv_bytes: bytes, filename: str) -> dict:
        """Compare a finished job with an uploaded review CSV; stored in the payload."""
        payload = self.payload(job_id)
        if payload is None:
            raise KeyError("No result for this job")
        with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as fh:
            fh.write(csv_bytes)
            tmp = Path(fh.name)
        try:
            review = lab.load_review(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        if not review.segments:
            raise ValueError("No reading spans found. Expected columns: start (min:sec), "
                             "end (min:sec), suspicious/not suspicious, reason.")
        review.path = Path(display_name(filename))
        review.ir = lab.ir_of(filename) or lab.ir_of(payload["source"])
        payload["review"] = compare_with_review(payload["result"], review)
        payload["review"]["origin"] = "uploaded"
        self._save_payload(job_id, payload)
        return payload["review"]

    def segments_csv(self, job_id: str):
        payload = self.payload(job_id)
        return None if payload is None else segments_csv(payload["result"])

    # ---------------------------------------------------------------- worker
    def _update(self, job_id, **kw):
        with self._lock:
            self._jobs[job_id].update(kw)

    def _worker(self):
        while True:
            job_id = self._queue.get()
            with self._lock:
                job = self._jobs[job_id]
                if job["status"] == "cancelled":
                    self._cleanup(job)
                    continue
                if job_id in self._order:
                    self._order.remove(job_id)
                job.update(status="running", stage="frames", started=time.time())
            try:
                payload = self._analyse(job)
                self._update(job_id, status="done", stage="done", finished=time.time(),
                             result=payload)
            except JobCancelled:
                self._update(job_id, status="cancelled", stage="cancelled", finished=time.time())
            except Exception as e:
                msg = str(e)
                if "too short" in msg:
                    msg = (f"Video is too short to analyse: the model needs at least "
                           f"{self.cfg.read_window_seconds:.0f}s of usable signal.")
                self._update(job_id, status="error", stage="error", finished=time.time(), error=msg)
            finally:
                self._cleanup(job)

    def _analyse(self, job) -> dict:
        job_id = job["id"]
        video = Path(job["upload"])
        dur = _estimate_duration(video)
        if job["max_seconds"]:
            dur = min(dur, job["max_seconds"]) if dur else job["max_seconds"]
        self._update(job_id, duration=dur)

        def progress(stage, seconds):
            if job["cancel"]:
                raise JobCancelled()
            if seconds is not None:
                self._update(job_id, stage=stage, seconds_done=float(seconds))
            else:
                self._update(job_id, stage=stage)

        t0 = time.time()
        result, win_df, clean, feat_path = predict.score_video(
            video, self.cfg, self.artifact, self.scorer,
            max_seconds=job["max_seconds"], progress=progress, workers=self.threads_per_job)
        processing = time.time() - t0
        if job["cancel"]:
            raise JobCancelled()

        import pandas as pd
        df = pd.read_parquet(feat_path)
        span = float(df["t"].iloc[-1]) if len(df) else 0.0
        analysed = float(win_df["win_end_s"].iloc[-1]) if len(win_df) else 0.0

        # report.py renders the video's name from result["video"]; never expose the
        # server-side upload path, show the original (sanitised) filename instead.
        result["video"] = job["source"]
        self._update(job_id, stage="report")
        review = lab.review_for(job["source"])
        predict.save_timeline(video, result, win_df, clean)
        (RUNS_DIR / f"{job_id}__traces.json").write_text(json.dumps(traces(clean)))
        with _REPORT_LOCK:
            report.build_video_report(video, review=review)

        payload = dict(
            version=2, job_id=job_id, source=job["source"], result=result,
            windows=bins(result, result["flagged_segments"]),
            stats=dict(
                n_frames=int(len(df)), face_ratio=round(float(df["face_found"].mean()), 4),
                video_seconds=round(span, 1), analysed_seconds=round(analysed, 1),
                sampled_fps=round(len(df) / span, 2) if span > 0 else None,
                processing_seconds=round(processing, 1),
                speed_x_realtime=round(span / processing, 1) if processing > 0 else None,
                max_seconds=job["max_seconds"], t0=0.0,     # v2 times are on the video clock
                stride_seconds=BIN_SECONDS, bin_seconds=BIN_SECONDS,
            ),
            runtime=dict(mediapipe="cpu", cnn="cpu (xgboost)", mediapipe_threads=self.threads_per_job),
            report_url=f"/api/jobs/{job_id}/report",
            segments_csv_url=f"/api/jobs/{job_id}/segments.csv",
            traces_url=f"/api/jobs/{job_id}/traces",
            video_url=f"/api/jobs/{job_id}/video" if self.keep_uploads else None,
            created=job["created"],
        )
        if review is not None:
            payload["review"] = compare_with_review(result, review)
            payload["review"]["origin"] = f"{LABELS_DIR.name}/ (matched by IR number)"
        (RUNS_DIR / f"{job_id}__web.json").write_text(json.dumps(payload, indent=2))
        return payload

    def _cleanup(self, job):
        """Delete the uploaded video and the per-frame cache; keep the result,
        timeline and report."""
        job_id = job["id"]
        if not self.keep_uploads:
            Path(job["upload"]).unlink(missing_ok=True)
        cache = RUNS_DIR / "_predict_cache"
        for p in cache.glob(f"scored__*__{job_id}.*"):
            p.unlink(missing_ok=True)
