"""
engine.py — the analysis side of the web app: a small in-process job queue that
runs uploaded videos through the existing src/ pipeline.

Nothing here re-implements the maths. Every job goes through predict.score_video,
the same function the CLI uses, so a video scored here gets exactly the verdict
`python src/predict.py <video>` would give it.

Hardware:
  * MediaPipe face/iris tracking runs on the CPU (TFLite XNNPACK). The Python
    MediaPipe build has no GPU delegate on macOS, and the shipped threshold was
    tuned on CPU landmarks anyway. This is where nearly all the time goes, so it is
    fanned out over a pool of threads — 90% of the cores by default — that is
    warmed once at start-up. Frames are independent, so the output is identical to
    a single-threaded run.
  * The CNN runs on the best available torch device (CUDA > Apple MPS > CPU) unless
    overridden. It is tiny, so the device barely matters for speed; MPS and CPU
    agree to ~1e-8 on the probabilities.
"""

from __future__ import annotations

import json
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parent
ROOT = WEBAPP_DIR.parent
RUNS_DIR = WEBAPP_DIR / "runs"
UPLOADS_DIR = RUNS_DIR / "uploads"

sys.path.insert(0, str(ROOT / "src"))

import predict  # noqa: E402  (src/ must be on sys.path first)
import report  # noqa: E402
from common import Config, DEFAULT_CONFIG, MODELS_DIR, warm_mediapipe, close_mediapipe  # noqa: E402

# Point src/'s output directory at runs/ so web jobs never write into reports/
# (and, via predict's cache dir, never into the training set's data/features/).
predict.REPORTS_DIR = RUNS_DIR
report.REPORTS_DIR = RUNS_DIR

VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v"}

# matplotlib's pyplot (used by report.py) is not thread-safe.
_REPORT_LOCK = threading.Lock()


class JobCancelled(Exception):
    pass


def default_threads() -> int:
    """90% of the logical cores (9 on a 10-core M5), leaving one for the decoder/OS."""
    return max(1, round(0.9 * (os.cpu_count() or 2)))


def pick_device(requested: str = "auto") -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


class Engine:
    def __init__(self, workers: int = 1, cnn_device: str = "auto", keep_uploads: bool = False,
                 threads: int = None):
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

        self.cfg = DEFAULT_CONFIG
        cfg_path = MODELS_DIR / "config.json"
        if cfg_path.exists():
            self.cfg = Config.from_json(cfg_path)

        self.device = pick_device(cnn_device)
        self.artifact, self.scorer = predict.load_winner(device=self.device)
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
        import torch
        a = self.artifact
        return dict(
            model=dict(kind=a["kind"], threshold=a["threshold"],
                       window_seconds=a["window_seconds"], stride_seconds=a["stride_seconds"],
                       seq_channels=a["seq_channels"], seq_len=a["seq_len"],
                       cv_metrics={k: a["metrics"].get(k) for k in
                                   ("video_roc_auc", "precision", "recall")}),
            runtime=dict(mediapipe="cpu", cnn=self.device, workers=self.workers,
                         mediapipe_threads=self.threads_per_job, cpu_count=os.cpu_count(),
                         cpu=_cpu_name(), torch=torch.__version__,
                         mps_available=torch.backends.mps.is_available(),
                         cuda_available=torch.cuda.is_available()),
            accepted=sorted(VIDEO_EXTS),
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
                           f"{self.cfg.window_seconds:.0f}s of usable signal.")
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

        # Per-frame stats from the extracted parquet (before it is cleaned up).
        import pandas as pd
        df = pd.read_parquet(feat_path)
        span = float(df["t"].iloc[-1]) if len(df) else 0.0
        # Window times are relative to the first kept frame (resample_uniform anchors
        # its grid there); the player adds t0 to map them onto the video's own clock.
        t0 = float(df["t"].iloc[0]) if len(df) else 0.0
        analysed = float(win_df["win_end_s"].iloc[-1]) if len(win_df) else 0.0

        # report.py renders the video's name from result["video"]; never expose the
        # server-side upload path, show the original (sanitised) filename instead.
        result["video"] = job["source"]
        self._update(job_id, stage="report")
        predict.save_timeline(video, result, win_df, clean)
        with _REPORT_LOCK:
            report.build_video_report(video)

        windows = [dict(start_s=float(r["win_start_s"]), end_s=float(r["win_end_s"]),
                        prob=float(r["prob"]),
                        reasons=predict.window_reasons(r, self.cfg))
                   for _, r in win_df.iterrows()]
        payload = dict(
            job_id=job_id, source=job["source"], result=result, windows=windows,
            stats=dict(
                n_frames=int(len(df)), face_ratio=round(float(df["face_found"].mean()), 4),
                video_seconds=round(span, 1), analysed_seconds=round(analysed, 1),
                sampled_fps=round(len(df) / span, 2) if span > 0 else None,
                processing_seconds=round(processing, 1),
                speed_x_realtime=round(span / processing, 1) if processing > 0 else None,
                max_seconds=job["max_seconds"], t0=round(t0, 4),
                stride_seconds=self.cfg.stride_seconds,
            ),
            runtime=dict(mediapipe="cpu", cnn=self.device, mediapipe_threads=self.threads_per_job),
            report_url=f"/api/jobs/{job_id}/report",
            video_url=f"/api/jobs/{job_id}/video" if self.keep_uploads else None,
            created=job["created"],
        )
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
