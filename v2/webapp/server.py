"""
server.py (v2) — web server for server-side reading detection.

Upload a recording from the page; the backend runs the same Python pipeline as
`v2/src/predict.py` (MediaPipe + blendshapes on CPU, XGBoost reading model) and the
page shows the verdict, the per-second reading timeline over the video, reading
segments with reasons, agreement with a manual review, and a detailed report.

Usage (from the project root):
    venv/bin/python v2/webapp/server.py                    # http://127.0.0.1:8002
    venv/bin/python v2/webapp/server.py --threads 6        # MediaPipe threads (default: 90% of cores)
    venv/bin/python v2/webapp/server.py --port 8080 --workers 2
    venv/bin/python v2/webapp/server.py --host 0.0.0.0     # reachable on your LAN (no auth!)

API:
    GET    /api/info                     model + hardware the server is using
    POST   /api/jobs                     multipart: file, max_minutes (optional) -> {id}
    GET    /api/jobs/{id}                status / progress / result
    DELETE /api/jobs/{id}                cancel a queued or running job
    GET    /api/jobs/{id}/report         detailed HTML report (gaze plots)
    GET    /api/jobs/{id}/result.json    result payload as a download
    GET    /api/jobs/{id}/video          the uploaded video (only with --keep-uploads)
    POST   /api/jobs/{id}/review         v2: multipart review CSV -> agreement with the manual review
    GET    /api/jobs/{id}/segments.csv   v2: detected reading segments, reviewers' sheet format
    GET    /api/jobs/{id}/traces         eye / head / mouth traces for the live signal chart

    Pre-scored library (every recording in the three folders; build it with library.py):
    GET    /api/library                  candidate list + summary
    GET    /api/library/{cid}            result payload (same shape as a job's)
    GET    /api/library/{cid}/video      the recording (seekable; HTTP Range)
    GET    /api/library/{cid}/report     detailed HTML report
    GET    /api/library/{cid}/traces     eye / head / mouth traces
    GET    /api/library/{cid}/segments.csv, /result.json   downloads
"""

from __future__ import annotations

import argparse
import mimetypes
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

import engine as eng
import library as lib

STATIC_DIR = Path(__file__).resolve().parent / "static"

ENGINE: eng.Engine | None = None
LIBRARY = lib.Library()


@asynccontextmanager
async def lifespan(_app):
    yield
    if ENGINE is not None:
        ENGINE.shutdown()


app = FastAPI(title="Gaze Integrity Analyser — server", docs_url="/api/docs", redoc_url=None,
              lifespan=lifespan)


def _engine() -> eng.Engine:
    if ENGINE is None:
        raise HTTPException(503, "Engine not started")
    return ENGINE


@app.get("/api/info")
def info():
    return _engine().info()


@app.post("/api/jobs")
def create_job(file: UploadFile = File(...), max_minutes: str = Form("")):
    engine = _engine()
    try:
        dest = engine.new_upload_path(file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))

    max_seconds = None
    if max_minutes.strip():
        try:
            minutes = float(max_minutes)
        except ValueError:
            raise HTTPException(400, "max_minutes must be a number")
        if minutes > 0:
            max_seconds = minutes * 60.0

    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out, length=4 * 1024 * 1024)
    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "The uploaded file is empty.")

    job_id = engine.submit(dest, file.filename, max_seconds)
    return JSONResponse({"id": job_id}, status_code=202)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    status = _engine().status(job_id)
    if status is None:
        raise HTTPException(404, "No such job")
    return status


@app.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str):
    if not _engine().cancel(job_id):
        raise HTTPException(409, "Job is not running")
    return {"cancelled": True}


@app.get("/api/jobs/{job_id}/report")
def job_report(job_id: str):
    path = _engine().report_path(job_id)
    if path is None:
        raise HTTPException(404, "No report for this job")
    return FileResponse(path, media_type="text/html")


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str):
    # FileResponse answers HTTP Range requests, so the player can seek.
    path = _engine().video_path(job_id)
    if path is None:
        raise HTTPException(404, "Video not kept on the server (start it with --keep-uploads)")
    media = mimetypes.guess_type(path.name)[0] or "video/mp4"
    return FileResponse(path, media_type=media)


@app.post("/api/jobs/{job_id}/review")
def job_review(job_id: str, file: UploadFile = File(...)):
    """v2: compare a finished job with a manual review CSV (reviewers' sheet format)."""
    data = file.file.read(2 * 1024 * 1024)
    try:
        return _engine().compare_review(job_id, data, file.filename or "review.csv")
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/jobs/{job_id}/segments.csv")
def job_segments_csv(job_id: str):
    """v2: detected reading segments in the reviewers' CSV format."""
    text = _engine().segments_csv(job_id)
    if text is None:
        raise HTTPException(404, "No result for this job")
    stem = Path(_engine().payload(job_id)["source"]).stem or job_id
    return PlainTextResponse(text, media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="{stem}__reading_segments.csv"'})


@app.get("/api/jobs/{job_id}/result.json")
def job_result(job_id: str):
    payload = _engine().payload(job_id)
    if payload is None:
        raise HTTPException(404, "No result for this job")
    stem = Path(payload["source"]).stem or job_id
    return JSONResponse(payload, headers={
        "Content-Disposition": f'attachment; filename="{stem}__prediction.json"'})


@app.get("/api/jobs/{job_id}/traces")
def job_traces(job_id: str):
    path = _engine().traces_path(job_id)
    if path is None:
        raise HTTPException(404, "No traces for this job")
    return FileResponse(path, media_type="application/json")


# --------------------------------------------------------------------------- #
# Pre-scored library: every recording in the three folders, built by library.py
# --------------------------------------------------------------------------- #
@app.get("/api/library")
def library_index():
    return LIBRARY.index()


def _library_payload(cid: str) -> dict:
    payload = LIBRARY.payload(cid)
    if payload is None:
        raise HTTPException(404, "No such candidate in the library")
    return payload


@app.get("/api/library/{cid}")
def library_item(cid: str):
    return _library_payload(cid)


@app.get("/api/library/{cid}/video")
def library_video(cid: str):
    path = LIBRARY.video_path(cid)
    if path is None:
        raise HTTPException(404, "Recording not found on disk")
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "video/mp4")


@app.get("/api/library/{cid}/report")
def library_report(cid: str):
    path = LIBRARY.report_path(cid)
    if path is None:
        raise HTTPException(404, "No report for this candidate")
    return FileResponse(path, media_type="text/html")


@app.get("/api/library/{cid}/traces")
def library_traces(cid: str):
    path = LIBRARY.traces_path(cid)
    if path is None:
        raise HTTPException(404, "No traces for this candidate")
    return FileResponse(path, media_type="application/json")


@app.get("/api/library/{cid}/segments.csv")
def library_segments_csv(cid: str):
    payload = _library_payload(cid)
    stem = Path(payload["source"]).stem
    return PlainTextResponse(eng.segments_csv(payload["result"]), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="{stem}__reading_segments.csv"'})


@app.get("/api/library/{cid}/result.json")
def library_result(cid: str):
    payload = _library_payload(cid)
    stem = Path(payload["source"]).stem
    return JSONResponse(payload, headers={
        "Content-Disposition": f'attachment; filename="{stem}__prediction.json"'})


# The page itself. Mounted last so the /api routes above take precedence.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main(argv=None):
    global ENGINE
    ap = argparse.ArgumentParser(description="Gaze Integrity Analyser — web server.")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (0.0.0.0 exposes it on your network — there is no auth)")
    ap.add_argument("--port", type=int, default=8002)   # v1 uses 8001; both can run side by side
    ap.add_argument("--workers", type=int, default=1,
                    help="videos analysed in parallel (they share the --threads budget)")
    ap.add_argument("--threads", type=int, default=None,
                    help=f"MediaPipe threads in total (default: 90%% of cores = {eng.default_threads()})")
    ap.add_argument("--cnn-device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                    help="ignored in v2 (the reading model is XGBoost on CPU); kept for v1 compatibility")
    ap.add_argument("--keep-uploads", action="store_true",
                    help="keep uploaded videos in webapp/runs/uploads/ (default: delete after scoring)")
    args = ap.parse_args(argv)

    ENGINE = eng.Engine(workers=max(1, args.workers), cnn_device=args.cnn_device,
                        keep_uploads=args.keep_uploads, threads=args.threads)
    rt = ENGINE.info()["runtime"]
    print(f"Gaze Integrity Analyser v2 — http://{args.host}:{args.port}")
    print(f"  MediaPipe: {rt['mediapipe']} x {rt['mediapipe_threads']} threads (of {rt['cpu_count']} cores)   "
          f"reading model: {rt['cnn']}   workers: {rt['workers']}   ({rt['cpu']})")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
