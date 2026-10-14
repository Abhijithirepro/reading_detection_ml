"""
evaluate.py (v2, new) — score models against the manual reading reviews.

Per-second metrics (reviewed videos) and video-level verdict metrics, shared by
train.py (v2) and the v1 baseline below. Results go to RESULTS_LOG.md + results/.

Usage:
    python src/evaluate.py --v1-baseline     # log how the ORIGINAL model does
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from common import PROJECT_ROOT, V1_ROOT
import results_log


# --------------------------------------------------------------------------- #
# Per-second metrics vs a review
# --------------------------------------------------------------------------- #
def per_second_metrics(score: np.ndarray, truth: np.ndarray, flagged: np.ndarray,
                       segments=None, t0: float = 0.0) -> dict:
    """score/truth/flagged are per-second arrays (NaN score = not analysed)."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    m = np.isfinite(score)
    s, y, f = score[m], truth[m].astype(bool), flagged[m].astype(bool)
    tp, fp, fn = int((f & y).sum()), int((f & ~y).sum()), int((~f & y).sum())
    out = dict(
        seconds=int(m.sum()), reading_s=int(y.sum()), flagged_s=int(f.sum()),
        auc=float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan"),
        ap=float(average_precision_score(y, s)) if y.sum() else float("nan"),
        precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
        iou=tp / max(tp + fp + fn, 1),
    )
    out["f1"] = 2 * out["precision"] * out["recall"] / max(out["precision"] + out["recall"], 1e-9)
    if segments:
        hits = 0
        for a, b, *_ in segments:
            i0, i1 = int(max(0, a - t0)), int(min(len(flagged), b - t0))
            if i1 > i0 and flagged[i0:i1].mean() >= 0.5:
                hits += 1
        out["spans_caught"] = f"{hits}/{len(segments)}"
        out["span_hit_rate"] = hits / len(segments)
    return out


def video_metrics(scores: pd.DataFrame, threshold: float) -> dict:
    """scores: columns label (1 cheating), score, matched (bool: same-platform clean)."""
    from sklearn.metrics import roc_auc_score
    y, s = scores["label"].to_numpy(), scores["score"].to_numpy()
    pred = s >= threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    out = dict(n=len(y), auc=float(roc_auc_score(y, s)), tp=tp, fp=fp, fn=fn, tn=tn,
               precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
               accuracy=(tp + tn) / max(len(y), 1), threshold=float(threshold))
    if "matched" in scores:
        mm = (scores["label"] == 1) | scores["matched"]
        out["matched_auc"] = float(roc_auc_score(y[mm], s[mm]))
        out["matched_fp"] = int((pred & (y == 0) & scores["matched"].to_numpy()).sum())
        out["matched_n_clean"] = int(((y == 0) & scores["matched"].to_numpy()).sum())
    return out


# --------------------------------------------------------------------------- #
# v1 baseline (runs the original src/ in its own process)
# --------------------------------------------------------------------------- #
_V1_SCRIPT = r'''
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
V1 = Path(sys.argv[1]); V2SRC = Path(sys.argv[2])
sys.path.insert(0, str(V1 / "src"))
from common import Config, MODELS_DIR
import build_windows as bw, predict
sys.path.append(str(V2SRC))
import importlib.util
spec = importlib.util.spec_from_file_location("labels_v2", V2SRC / "labels.py")
cfg = Config.from_json(MODELS_DIR / "config.json")
art, scorer = predict.load_winner()
out = {"threshold": art["threshold"], "videos": {}}
# labels.py imports v2 common for LABELS_DIR only; parse the CSVs directly here instead
import csv, re
def ts(x):
    p = [float(v) for v in x.strip().split(":")]
    return p[0] * 60 + p[1] if len(p) >= 2 else p[0]
for f in sorted((V1 / "manual_review_for_candidate").glob("*.csv")):
    ir = re.search(r"ir(\d+)", f.name).group(1)
    rows = list(csv.reader(open(f, newline="", encoding="utf-8-sig")))[1:]
    segs = [(ts(r[0]), ts(r[1])) for r in rows if len(r) > 1 and r[0].strip() and r[1].strip()]
    pq = next((V1 / "data" / "features").glob(f"cheating__*ir{ir}*.parquet"), None)
    if pq is None:
        continue
    df = pd.read_parquet(pq)
    c = bw.clean_video(df, cfg, False)
    n, dt = len(c["gaze_x"]), c["dt"]
    wl = int(round(cfg.window_seconds * cfg.target_fps)); st = int(round(cfg.stride_seconds * cfg.target_fps))
    L = int(round(cfg.window_seconds * cfg.seq_resample_fps))
    ws, X = [], []
    for s0 in range(0, n - wl + 1, st):
        ws.append(s0 * dt); X.append(bw._window_sequence(c, s0, s0 + wl, L))
    prob = scorer(None, np.stack(X))
    out["videos"][ir] = dict(name=pq.stem, t0=float(df.t.iloc[0]), n_sec=int(np.ceil(n * dt)),
                             win_start=ws, win_len=cfg.window_seconds, prob=prob.tolist(), segments=segs)
print(json.dumps(out))
'''


def per_second_from_windows(win_start, win_len, prob, n_sec, how="max"):
    s = np.full(n_sec, np.nan)
    for a, p in zip(win_start, prob):
        i0, i1 = max(0, int(np.floor(a))), min(n_sec, int(np.ceil(a + win_len)))
        seg = s[i0:i1]
        s[i0:i1] = np.where(np.isnan(seg), p, np.maximum(seg, p) if how == "max" else seg)
    return s


def v1_baseline():
    py = sys.executable
    res = subprocess.run([py, "-c", _V1_SCRIPT, str(V1_ROOT), str(PROJECT_ROOT / "src")],
                         capture_output=True, text=True, cwd=str(V1_ROOT),
                         env={"KMP_DUPLICATE_LIB_OK": "TRUE", "PATH": "/usr/bin:/bin"})
    if res.returncode != 0:
        print(res.stderr[-3000:])
        return 1
    data = json.loads(res.stdout.strip().splitlines()[-1])
    thr = data["threshold"]
    rows = []
    for ir, v in data["videos"].items():
        score = per_second_from_windows(v["win_start"], v["win_len"], v["prob"], v["n_sec"])
        truth = np.zeros(v["n_sec"], bool)
        for a, b in v["segments"]:
            truth[int(a - v["t0"]):int(b - v["t0"])] = True
        m = per_second_metrics(score, truth, score >= thr, v["segments"], v["t0"])
        rows.append(dict(video=v["name"].split("__")[-1], **m))

    # video level: v1's own out-of-fold scores, restricted to the 100 on-disk recordings
    vs = pd.read_csv(V1_ROOT / "models" / "cv_video_scores.csv", float_precision="round_trip")
    vs = vs[vs.group.str.contains("__interviews")].copy()
    vs["matched"] = vs.group.str.startswith("clean__interviews__")
    vm = video_metrics(vs, thr)

    cols = ["video", "reading_s", "flagged_s", "auc", "precision", "recall", "iou", "spans_caught"]
    summary = (f"Original model (`models/`, 1D-CNN, threshold {thr:.3f}) scored on the three "
               "reviewed videos. **In-sample**: v1 was trained on these very videos, so this is "
               "its best case. Per-second score = max probability of the 20 s windows covering "
               "that second. Video-level row uses v1's own out-of-fold CV scores.")
    vrow = dict(scope="all 100", **{k: vm[k] for k in ("auc", "precision", "recall", "tp", "fp", "fn", "tn")},
                matched_auc=vm["matched_auc"], matched_fp=f"{vm['matched_fp']}/{vm['matched_n_clean']}")
    results_log.append("v1-baseline", summary,
                       {"Per-second vs manual reviews": results_log.table(rows, cols),
                        "Video level (v1 out-of-fold)": results_log.table([vrow])},
                       dict(per_second=rows, video=vm))
    print(pd.DataFrame(rows)[cols].to_string(index=False))
    print(json.dumps(vm, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="v2 evaluation helpers")
    ap.add_argument("--v1-baseline", action="store_true")
    args = ap.parse_args(argv)
    if args.v1_baseline:
        return v1_baseline()
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
