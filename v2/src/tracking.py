"""
tracking.py (v2, new) — mirrors every results_log run into MLflow.

results_log.append() calls log_run() right after it writes the JSON and the
RESULTS_LOG.md section, so train.py, train_seq.py and evaluate.py are all tracked
without code of their own. Tracking is best-effort: if MLflow isn't installed, or
V2_TRACKING=0, or logging fails, the run is still in RESULTS_LOG.md + results/ and
the script carries on.

Local store (nothing leaves the machine; both git-ignored):
  v2/mlflow.db       runs, params, metrics, tags
  v2/mlartifacts/    per-run files: result JSON, log section, out-of-fold scores,
                     and the shipped model when train.py ships

What a run holds
  params   the script's argparse args
  metrics  per_second/*  pooled over the reviewed videos (same pooling as RESULTS_LOG.md)
           per_video/<ir>/*  each reviewed video
           video/*  video-level verdicts;  stat_auc/*  each candidate video statistic
  notes    the run summary.  Head-to-heads (train_seq --compare-xgb) get one child
           run per contender, so all contenders share metric names.

  venv/bin/python v2/src/tracking.py ui          # http://127.0.0.1:5050
  venv/bin/python v2/src/tracking.py backfill    # import results/*.json not yet tracked
  venv/bin/python v2/src/tracking.py repair      # fill columns / metrics older runs lack
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from common import PROJECT_ROOT, RESULTS_DIR, V1_ROOT

EXPERIMENT = "reading-detection-v2"
DB_PATH = PROJECT_ROOT / "mlflow.db"
ARTIFACT_DIR = PROJECT_ROOT / "mlartifacts"
UI_PORT = 5050                     # 5000 is taken by AirPlay Receiver on macOS
LOG_PATH = PROJECT_ROOT / "RESULTS_LOG.md"

_STARTED_MS = int(time.time() * 1000)   # imported at script start -> run duration
_last_run_id = None


def _client():
    if os.environ.get("V2_TRACKING", "1") == "0":
        return None
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("tracking: mlflow not installed, run not tracked (pip install mlflow)")
        return None
    try:                                   # opens (and migrates) the store right away
        return MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{DB_PATH}")
    except Exception as e:
        print(f"tracking: cannot open the MLflow store ({type(e).__name__}: {e}), run not tracked")
        return None


def _experiment_id(client) -> str:
    # without this tag the MLflow 3 UI opens the experiment in its (empty) GenAI traces view
    kind = {"mlflow.experimentKind": "custom_model_development"}
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is not None:
        if "mlflow.experimentKind" not in exp.tags:
            client.set_experiment_tag(exp.experiment_id, *next(iter(kind.items())))
        return exp.experiment_id
    local = not os.environ.get("MLFLOW_TRACKING_URI")
    return client.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_DIR.as_uri() if local else None,
                                    tags=kind)


# ---------------------------------------------------------------- payload -> MLflow

def _num(v) -> bool:
    if v is None or isinstance(v, (bool, str)):
        return False
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _ratio(s) -> tuple:
    a, b = str(s).split("/")
    return int(a), int(b)


def _ir(video: str) -> str:
    m = re.search(r"ir\d+", video)
    return m.group(0) if m else video


def metrics_of(res: dict) -> dict:
    """Flat metric dict from one evaluation result (per_second / video / stat_auc)."""
    m = {}
    ps = res.get("per_second") or []
    for r in ps:
        for k in ("auc", "ap", "precision", "recall", "iou", "f1", "span_hit_rate"):
            if _num(r.get(k)):
                m[f"per_video/{_ir(r['video'])}/{k}"] = float(r[k])
    if ps:
        reading = sum(r["reading_s"] for r in ps)
        flagged = sum(r["flagged_s"] for r in ps)
        aucs = [r["auc"] for r in ps if _num(r["auc"])]
        caught = [_ratio(r["spans_caught"]) for r in ps]
        m["per_second/auc_mean"] = sum(aucs) / len(aucs) if aucs else float("nan")
        m["per_second/iou_mean"] = sum(r["iou"] for r in ps) / len(ps)
        m["per_second/precision_pooled"] = sum(r["precision"] * r["flagged_s"] for r in ps) / max(flagged, 1)
        m["per_second/recall_pooled"] = sum(r["recall"] * r["reading_s"] for r in ps) / max(reading, 1)
        m["per_second/spans_caught"] = sum(a for a, _ in caught)
        m["per_second/span_hit_rate"] = sum(a for a, _ in caught) / max(sum(b for _, b in caught), 1)
        m["per_second/flagged_s"] = flagged
    for k, v in (res.get("video") or {}).items():
        if _num(v):
            m[f"video/{k}"] = float(v)
        elif k == "unreviewed_caught":
            a, b = _ratio(v)
            m["video/unreviewed_caught"] = a
            m["video/unreviewed_caught_rate"] = a / max(b, 1)
    for k, v in (res.get("stat_auc") or {}).items():
        if _num(v):
            m[f"stat_auc/{k}"] = float(v)
    if _num(res.get("second_threshold")):
        m["second_threshold"] = float(res["second_threshold"])
    return {k: v for k, v in m.items() if _num(v)}


def _params(args: dict) -> dict:
    out = {}
    for k, v in args.items():
        if isinstance(v, (list, tuple)):
            v = ", ".join(map(str, v)) or "none"
        out[k] = str(v)[:6000]
    return out


# the options of train.py (XGBoost) and train_seq.py (CNN): every v2 run gets all of
# them, so runs line up column for column in the UI
XGB_OPTS = {"exp", "folds", "seed", "reviewed_neg_weight", "clean_budget", "video_stat", "drop_group",
            "pseudo_frac", "pseudo_weight", "clean_speech_weight", "no_ship", "no_log", "note"}
CNN_OPTS = {"exp", "epochs", "folds", "seed", "reviewed_neg_weight", "pseudo_frac", "pseudo_weight",
            "clean_speech_weight", "clean_budget", "video_stat", "compare_xgb", "workers", "note"}
MODEL_NAMES = {"CNN": "1D-CNN", "XGB+CNN average": "XGBoost + 1D-CNN average"}


def _is_cnn(payload: dict) -> bool:
    """train_seq.py run (the first one predates the per-contender layout)."""
    return "contenders" in payload or "epochs" in (payload.get("args") or {})


def _complete_params(args: dict, payload: dict, contender: str = None) -> dict:
    """The run's options plus the ones it didn't have (marked), the model, the
    contender and the sampling rate."""
    cnn = _is_cnn(payload)
    own = CNN_OPTS if cnn else XGB_OPTS
    out = _params(args)
    for k in sorted((XGB_OPTS | CNN_OPTS) - set(out)):
        out[k] = "not an option yet" if k in own else f"n/a ({'CNN' if cnn else 'XGBoost'} run)"
    contender = contender or ("CNN" if cnn else "XGBoost")
    out["contender"] = contender
    out["model"] = MODEL_NAMES.get(contender, contender)
    # every v2 run before the switch to 15 fps sampled at 25 fps and logged no rate
    out["target_fps"] = str(float(payload.get("target_fps") or 25.0))
    return out


def _run_metrics(payload: dict, contender: str = None) -> dict:
    """A run's metrics: the payload's own, or (train_seq) the named contender's —
    the CNN for the run itself."""
    contenders = payload.get("contenders") or {}
    if not contenders:
        return metrics_of(payload)
    return metrics_of(contenders[contender or ("CNN" if "CNN" in contenders else next(iter(contenders)))])


def _git_tags() -> dict:
    def git(*a):
        return subprocess.run(["git", "-C", str(V1_ROOT), *a], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except Exception:
        return {}
    return {"mlflow.source.git.commit": commit, "git_dirty": str(dirty).lower()} if commit else {}


def _batch(client, run_id, metrics: dict, params: dict, ts: int):
    from mlflow.entities import Metric, Param
    ms = [Metric(k, v, ts, 0) for k, v in metrics.items()]
    ps = [Param(k, v) for k, v in params.items()]
    for i in range(0, max(len(ms), len(ps), 1), 100):
        client.log_batch(run_id, metrics=ms[i:i + 100], params=ps[i:i + 100])


def _log(client, name, summary, payload, json_path, section_md, start_ms, end_ms, backfilled):
    eid = _experiment_id(client)
    args = payload.get("args") or {}
    params = _complete_params(args, payload) if args else {}
    tags = {"mlflow.note.content": summary, "results_json": json_path.name,
            "mlflow.source.type": "LOCAL"}
    if backfilled:
        tags["backfilled"] = "true"
    else:
        tags["mlflow.source.name"] = Path(sys.argv[0]).name
        tags.update(_git_tags())
    if args.get("no_ship") is False:
        tags["shipped"] = "true"

    contenders = payload.get("contenders") or {}
    metrics = _run_metrics(payload)
    if len(contenders) > 1:
        tags["metrics_from"] = "CNN contender (each contender also has a child run)"

    run = client.create_run(eid, start_time=start_ms, tags=tags, run_name=name)
    rid = run.info.run_id
    _batch(client, rid, metrics, params, end_ms)
    client.log_artifact(rid, str(json_path))
    if section_md:
        client.log_text(rid, section_md, "results_log_section.md")
    for p in sorted(RESULTS_DIR.glob(f"oof_*__{name}.*")):
        client.log_artifact(rid, str(p), "oof")

    if len(contenders) > 1:
        for cname, res in contenders.items():
            child = client.create_run(eid, start_time=start_ms, run_name=f"{name} / {cname}",
                                      tags={"mlflow.parentRunId": rid, "contender": cname,
                                            "mlflow.source.type": "LOCAL"})
            _batch(client, child.info.run_id, metrics_of(res), _complete_params(args, payload, cname), end_ms)
            client.set_terminated(child.info.run_id, "FINISHED", end_time=end_ms)
    client.set_terminated(rid, "FINISHED", end_time=end_ms)
    return rid


def log_run(name: str, summary: str, payload: dict, json_path: Path, section_md: str = "",
            start_ms: int = None, end_ms: int = None, backfilled: bool = False):
    """Mirror one results_log run into MLflow. Never raises; returns the run id or None."""
    global _last_run_id
    client = _client()
    if client is None:
        return None
    end_ms = end_ms or int(time.time() * 1000)
    try:
        rid = _log(client, name, summary, payload, Path(json_path), section_md,
                   start_ms or _STARTED_MS, end_ms, backfilled)
    except Exception as e:
        print(f"tracking: MLflow logging failed ({type(e).__name__}: {e}); "
              "the run is still in RESULTS_LOG.md")
        return None
    if not backfilled:
        _last_run_id = rid
        print(f"tracking: MLflow run {rid[:8]} ({name})")
    return rid


def attach(*paths, artifact_path: str = "model", shipped: bool = True):
    """Add files (e.g. the shipped model) to the run results_log.append() just logged."""
    if _last_run_id is None:
        return
    client = _client()
    if client is None:
        return
    try:
        for p in paths:
            if Path(p).exists():
                client.log_artifact(_last_run_id, str(p), artifact_path)
        if shipped:
            client.set_tag(_last_run_id, "shipped", "true")
    except Exception as e:
        print(f"tracking: could not attach {artifact_path} ({type(e).__name__}: {e})")


# ---------------------------------------------------------------- CLI

def _log_section(fname: str) -> str:
    if not LOG_PATH.exists():
        return ""
    for chunk in LOG_PATH.read_text().split("\n---\n"):
        if f"results/{fname}" in chunk:
            return chunk.strip() + "\n"
    return ""


def backfill() -> int:
    """Import results/*.json runs that MLflow doesn't have yet (safe to re-run)."""
    client = _client()
    if client is None:
        return 1
    eid = _experiment_id(client)
    done = {r.data.tags.get("results_json") for r in client.search_runs([eid], max_results=50000)}
    added = skipped = 0
    for p in sorted(RESULTS_DIR.glob("[0-9]*__*.json")):
        if p.name in done:
            skipped += 1
            continue
        d = json.loads(p.read_text())
        ms = int(datetime.strptime(p.name.split("__")[0], "%Y%m%d-%H%M%S").timestamp() * 1000)
        rid = log_run(d.get("name", p.stem.split("__", 1)[1]), d.get("summary", ""), d, p,
                      _log_section(p.name), start_ms=ms, end_ms=ms, backfilled=True)
        print(f"  {'added ' if rid else 'FAILED'} {p.name}")
        added += bool(rid)
    print(f"backfill: {added} run(s) added, {skipped} already tracked")
    return 0


def repair() -> int:
    """Fill what older v2 runs lack — the full option columns, model, contender,
    target_fps, and metrics never logged (a head-to-head's own run) — from their
    results JSON. Values already logged are never changed (safe to re-run). v1
    baseline runs have no options and are left alone."""
    client = _client()
    if client is None:
        return 1
    eid = _experiment_id(client)
    n_p = n_m = 0
    for run in client.search_runs([eid], max_results=50000):
        tags = run.data.tags
        parent = tags.get("mlflow.parentRunId")
        jname = tags.get("results_json") or (client.get_run(parent).data.tags.get("results_json")
                                             if parent else None)
        if not jname or not (RESULTS_DIR / jname).exists():
            continue
        payload = json.loads((RESULTS_DIR / jname).read_text())
        if not payload.get("args"):
            continue
        contender = tags.get("contender")
        params = {k: v for k, v in _complete_params(payload["args"], payload, contender).items()
                  if k not in run.data.params}
        metrics = {k: v for k, v in _run_metrics(payload, contender).items() if k not in run.data.metrics}
        if params or metrics:
            _batch(client, run.info.run_id, metrics, params, run.info.end_time or run.info.start_time)
            print(f"  {run.info.run_name}: +{len(params)} params, +{len(metrics)} metrics")
        if not parent and len(payload.get("contenders") or {}) > 1 and "metrics_from" not in tags:
            client.set_tag(run.info.run_id, "metrics_from", "CNN contender (each contender also has a child run)")
        n_p, n_m = n_p + len(params), n_m + len(metrics)
    print(f"repair: {n_p} params and {n_m} metrics added")
    return 0


def ui(port: int) -> int:
    client = _client()
    page = f"#/experiments/{_experiment_id(client)}/runs" if client else ""
    print(f"MLflow UI on http://127.0.0.1:{port}/{page}  (Ctrl+C to stop)")
    return subprocess.call([sys.executable, "-m", "mlflow", "ui", "--backend-store-uri",
                            f"sqlite:///{DB_PATH}", "--host", "127.0.0.1", "--port", str(port)])


def main(argv=None):
    ap = argparse.ArgumentParser(description="v2 experiment tracking (MLflow).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("backfill", help="import results/*.json not yet in MLflow")
    sub.add_parser("repair", help="fill params / metrics older v2 runs lack (never overwrites)")
    u = sub.add_parser("ui", help="start the MLflow UI on the local store")
    u.add_argument("--port", type=int, default=UI_PORT)
    args = ap.parse_args(argv)
    return {"backfill": backfill, "repair": repair}.get(args.cmd, lambda: ui(args.port))()


if __name__ == "__main__":
    sys.exit(main())
