"""
train.py — Stage 3: train Model A (XGBoost) + Model B (GRU/1D-CNN) + Model C
(MiniROCKET), compare with leakage-safe grouped cross-validation, and ship the winner.

What it does
------------
* Loads data/windows/windows.parquet + sequences.npz + models/config.json.
* Backend bake-off: runs the SAME CV over each available feature set
  (mediapipe-only, mediapipe+l2cs) and each model, so nothing is guessed.
* Model C (MiniROCKET) is a pretrained-style fixed convolutional feature
  transform + a linear head — no weights to download, trains in seconds, and is
  a strong baseline on small time-series datasets. Runs on the same raw gaze
  sequences as Model B; select it with --seq-kind minirocket (needs `sktime`).
* CV: StratifiedGroupKFold grouped by video (windows from one video never span
  train/val). Falls back to LeaveOneGroupOut when videos/class is small.
* Aggregates out-of-fold window probabilities to a VIDEO-level verdict
  (mean of the top-k window probabilities), tunes a precision-favoring threshold,
  and reports window- and video-level ROC-AUC / PR-AUC / confusion matrix.
* Picks the winner by video-level ROC-AUC, applying the deployability rule:
  an appearance backend (l2cs) must beat mediapipe-only by >= --deploy-margin,
  else mediapipe-only wins (it runs in-browser). Model B breaks ties (ONNX/raw
  channels -> easiest to deploy).
* Retrains the winner on all data and saves artifacts + threshold + importances
  + full metrics to models/, plus an optional ONNX export.

Usage:
    python src/train.py
    python src/train.py --skip-seq                # Model A (XGBoost) only
    python src/train.py --seq-kind minirocket     # Model C (MiniROCKET) vs XGBoost
    python src/train.py --deploy-margin 0.03
    python src/train.py --export-onnx
"""

from __future__ import annotations

import os

# XGBoost (libgomp) and PyTorch (libomp) each ship an OpenMP runtime; loading both
# in one process can abort or DEADLOCK. Allow coexistence before either is imported.
# If Model B (the GRU) still hangs, also run with OMP_NUM_THREADS=1 (see README).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from common import Config, DEFAULT_CONFIG, WINDOWS_DIR, MODELS_DIR, L2CS_CHANNELS

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Data loading + feature sets
# --------------------------------------------------------------------------- #
def load_data():
    wpath = WINDOWS_DIR / "windows.parquet"
    if not wpath.exists():
        print(f"Missing {wpath}. Run build_windows.py first.")
        sys.exit(1)
    windows = pd.read_parquet(wpath)
    meta = json.loads((WINDOWS_DIR / "_feature_columns.json").read_text())
    with np.load(WINDOWS_DIR / "sequences.npz", allow_pickle=True) as z:
        seq = {k: z[k] for k in z.files}  # materialize once: repeated lazy reads of
    return windows, meta, seq             # the zip (esp. across subprocesses) can CRC-fail


def feature_sets(meta) -> dict:
    """Return {name: [feature columns]} for the bake-off. Includes a gaze+head-only
    variant that drops the ear/blink features — these are confounded by frame rate /
    resolution across recording sources and generalize worse (see README notes)."""
    all_cols = meta["feature_cols"]
    l2cs_cols = [c for c in all_cols if c.startswith("l2cs_")]
    mp_only = [c for c in all_cols if not c.startswith("l2cs_")]
    eye = [c for c in mp_only if c.startswith("ear") or c.startswith("blink")]
    gaze_head = [c for c in mp_only if c not in eye]
    sets = {"mediapipe": mp_only, "mediapipe_gaze_head": gaze_head}
    if l2cs_cols:
        sets["mediapipe+l2cs"] = mp_only + l2cs_cols
        sets["mediapipe+l2cs_gaze_head"] = gaze_head + l2cs_cols
    return sets


# --------------------------------------------------------------------------- #
# CV splitter
# --------------------------------------------------------------------------- #
def make_splitter(y, groups, cfg: Config):
    from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut

    g = pd.DataFrame({"y": y, "g": groups})
    per_class = g.drop_duplicates().groupby("y")["g"].nunique()
    min_videos = int(per_class.min()) if len(per_class) else 0
    n_groups = pd.unique(groups).size

    if min_videos < cfg.loo_threshold_videos:
        return LeaveOneGroupOut(), "LeaveOneGroupOut", n_groups
    n_splits = min(cfg.n_splits, min_videos)
    return (StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                 random_state=cfg.random_state),
            f"StratifiedGroupKFold(n_splits={n_splits})", n_splits)


# --------------------------------------------------------------------------- #
# Model A: XGBoost
# --------------------------------------------------------------------------- #
def xgb_model(y_train, cfg: Config):
    import xgboost as xgb

    pos = max(1, int(np.sum(y_train == 1)))
    neg = max(1, int(np.sum(y_train == 0)))
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
        reg_lambda=1.0, tree_method="hist", eval_metric="logloss",
        scale_pos_weight=neg / pos, random_state=cfg.random_state, n_jobs=0,
    )


def oof_xgb(X, y, groups, splitter, cfg: Config):
    oof = np.full(len(y), np.nan)
    for tr, te in splitter.split(X, y, groups):
        m = xgb_model(y[tr], cfg)
        m.fit(X[tr], y[tr])
        proba = m.predict_proba(X[te])
        if proba.shape[1] > 1:
            oof[te] = proba[:, 1]
        else:  # degenerate single-class train fold
            oof[te] = float(m.classes_[0]) * proba[:, 0]
    return oof


# --------------------------------------------------------------------------- #
# Model B: GRU (fallback 1D-CNN) on raw sequences
# --------------------------------------------------------------------------- #
def _torch_available():
    # check WITHOUT importing torch into this (XGBoost-holding) process
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def _build_seq_net(n_channels, kind="gru"):
    import torch.nn as nn

    class GRUNet(nn.Module):
        def __init__(self, c, hidden=32):
            super().__init__()
            self.gru = nn.GRU(c, hidden, batch_first=True, num_layers=1)
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, 1))

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    class CNNNet(nn.Module):
        def __init__(self, c, hidden=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(c, hidden, 5, padding=2), nn.ReLU(),
                nn.Conv1d(hidden, hidden, 5, padding=2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, 1))

        def forward(self, x):  # x: (B, L, C)
            z = self.net(x.transpose(1, 2)).squeeze(-1)
            return self.head(z).squeeze(-1)

    return GRUNet(n_channels) if kind == "gru" else CNNNet(n_channels)


def pick_seq_device(n_windows: int, override: str = None):
    """Choose the device for the sequence net.

    A GRU runs seq_len steps sequentially; on Apple MPS the per-step kernel-launch
    latency makes it SLOWER than CPU at the dataset sizes seen here (hundreds–low
    thousands of windows), so `auto` stays on CPU. Force MPS with --seq-device mps
    (worth it mainly for the 1D-CNN or very large data); CUDA is used if present.
    """
    import torch
    if override and override != "auto":
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _train_seq_net(Xtr, ytr, cfg: Config, kind="gru", epochs=40, device=None,
                   patience=6):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(cfg.random_state)
    if device is None:
        device = pick_seq_device(len(Xtr))
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    Xn = (Xtr - mean) / std

    net = _build_seq_net(Xtr.shape[-1], kind).to(device)
    pos = max(1, int((ytr == 1).sum())); neg = max(1, int((ytr == 0).sum()))
    pos_weight = torch.tensor([neg / pos], device=device)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    ds = TensorDataset(torch.tensor(Xn, dtype=torch.float32),
                       torch.tensor(ytr, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    net.train()
    best_loss, bad = float("inf"), 0
    for _ in range(epochs):
        epoch_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
        if epoch_loss < best_loss - 1e-4:
            best_loss, bad = epoch_loss, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return net, mean, std, device


def _predict_seq_net(net, mean, std, device, X):
    import torch
    net.eval()
    Xn = (X - mean) / std
    with torch.no_grad():
        logits = net(torch.tensor(Xn, dtype=torch.float32).to(device))
        return torch.sigmoid(logits).cpu().numpy()


def _seq_fold_worker(job):
    """Train + predict ONE CV fold in an isolated (torch-only, no XGBoost) process.
    Runs single-threaded so N folds fan out across N cores without oversubscription
    and without the XGBoost/torch OpenMP conflict."""
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch
    torch.set_num_threads(1)
    Xtr, ytr, Xte, te_idx, cfg, kind = job
    net, mean, std, dev = _train_seq_net(Xtr, ytr, cfg, kind, device=torch.device("cpu"))
    return te_idx, _predict_seq_net(net, mean, std, dev, Xte)


def oof_seq(X, y, groups, splitter, cfg: Config, kind="gru", device=None, workers=1):
    """Out-of-fold predictions for the sequence model, folds run in parallel."""
    folds = list(splitter.split(X, y, groups))
    jobs = [(X[tr], y[tr], X[te], te, cfg, kind) for tr, te in folds]
    oof = np.full(len(y), np.nan)
    if workers and workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for te_idx, probs in ex.map(_seq_fold_worker, jobs):
                oof[te_idx] = probs
    else:
        for job in jobs:
            te_idx, probs = _seq_fold_worker(job)
            oof[te_idx] = probs
    return oof


# --------------------------------------------------------------------------- #
# Model C: MiniROCKET (fixed convolutional transform + linear head) on sequences
#
# MiniROCKET applies ~10k fixed convolutional kernels to the raw gaze sequences
# and feeds their PPV summaries to a logistic head. There are no learned/downloaded
# weights (the kernels are fixed by construction), it trains in seconds, and it is
# a strong baseline on small time-series datasets — a good "pretrained-style"
# contender against the from-scratch GRU when data is scarce. It rides on `sktime`,
# which pulls in numba/OpenMP, so — like the torch models — it runs in an isolated
# subprocess to avoid the XGBoost/OpenMP conflict (see the KMP note at the top).
# --------------------------------------------------------------------------- #
def _minirocket_available():
    import importlib.util
    return importlib.util.find_spec("sktime") is not None


def _minirocket_input(X):
    """(N, seq_len, C) float sequences -> (N, C, seq_len) float32, NaN/inf-safe,
    the panel layout sktime's ROCKET transforms expect. Workers up-cast to float64
    for MultiROCKET (its numba kernel requires float64; MiniROCKET takes float32)."""
    Xr = np.transpose(np.asarray(X, dtype=np.float32), (0, 2, 1))
    return np.ascontiguousarray(np.nan_to_num(Xr))


def _rocket_cast(Xr, kind):
    """MultiROCKET's numba transform requires float64 input; MiniROCKET takes float32."""
    return Xr.astype(np.float64) if kind == "multirocket" else Xr


def _build_rocket_pipe(n_channels, cfg: Config, kind="minirocket"):
    """MiniROCKET (10k kernels) or MultiROCKET (2.5k kernels, multiple pooling ops
    -> richer features; wins on this data at video AUC 0.799 vs 0.779) + a linear head."""
    from sktime.transformations.panel.rocket import (
        MiniRocketMultivariate, MultiRocketMultivariate)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    # n_jobs=-1: parallelize the numba kernel transform across all cores (the
    # heavy step). Keeps the single-process OOF design (numba JIT compiled once)
    # while using every core instead of one.
    if kind == "multirocket":
        tf = MultiRocketMultivariate(num_kernels=2500, random_state=cfg.random_state,
                                     n_jobs=-1)
    else:
        tf = MiniRocketMultivariate(num_kernels=10000, random_state=cfg.random_state,
                                    n_jobs=-1)
    return make_pipeline(
        tf, StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced",
                           random_state=cfg.random_state),
    )


def _minirocket_oof_worker(job):
    """Run the WHOLE OOF loop in one isolated (sktime/numba-only, no XGBoost)
    process: sktime imports + numba JIT compile once, then loop every fold."""
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    Xr, y, folds, cfg, n_channels, kind = job
    Xr = _rocket_cast(Xr, kind)
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2:  # degenerate single-class train fold
            oof[te] = float(y[tr][0])
            continue
        pipe = _build_rocket_pipe(n_channels, cfg, kind)
        pipe.fit(Xr[tr], y[tr])
        proba = pipe.predict_proba(Xr[te])
        oof[te] = proba[:, 1] if proba.shape[1] > 1 else np.full(len(te), float(pipe.classes_[0]))
    return oof


def oof_minirocket(X, y, groups, splitter, cfg: Config, kind="minirocket"):
    """Out-of-fold predictions for a ROCKET model, all folds in one isolated process."""
    from concurrent.futures import ProcessPoolExecutor
    Xr = _minirocket_input(X)
    folds = list(splitter.split(X, y, groups))
    job = (Xr, y, folds, cfg, X.shape[-1], kind)
    with ProcessPoolExecutor(max_workers=1) as ex:
        return ex.submit(_minirocket_oof_worker, job).result()


def _minirocket_multi_worker(job):
    """Fit+predict SEVERAL (train_idx, test_idx) folds in one isolated process, so
    sktime import + numba JIT happen once and are reused across every fold. Returns
    a list of per-fold probability arrays aligned with `folds`. Used for the
    leave-one-source-pair-out sweep, whose test sets overlap (a video's source can
    appear in many pairs), so results can't be merged into a single OOF vector."""
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    Xr, y, folds, cfg, n_channels, kind = job
    Xr = _rocket_cast(Xr, kind)
    out = []
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2:  # degenerate single-class train fold
            out.append(np.full(len(te), float(y[tr][0])))
            continue
        pipe = _build_rocket_pipe(n_channels, cfg, kind)
        pipe.fit(Xr[tr], y[tr])
        proba = pipe.predict_proba(Xr[te])
        out.append(proba[:, 1] if proba.shape[1] > 1 else np.full(len(te), float(pipe.classes_[0])))
    return out


def _minirocket_final_worker(job):
    """Fit the ROCKET model on ALL data and pickle it, in an isolated process."""
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import joblib
    Xr, y, cfg, n_channels, kind = job
    Xr = _rocket_cast(Xr, kind)
    pipe = _build_rocket_pipe(n_channels, cfg, kind)
    pipe.fit(Xr, y)
    joblib.dump(pipe, MODELS_DIR / "model_minirocket.pkl")


# --------------------------------------------------------------------------- #
# Aggregation + metrics
# --------------------------------------------------------------------------- #
def video_scores(oof, groups, labels, cfg: Config) -> pd.DataFrame:
    """Aggregate window OOF probs to one score/video via mean of top-k windows."""
    df = pd.DataFrame({"group": groups, "label": labels, "p": oof}).dropna(subset=["p"])
    rows = []
    for g, sub in df.groupby("group"):
        p = np.sort(sub["p"].to_numpy())[::-1]
        k = max(cfg.topk_min, int(np.ceil(cfg.topk_fraction * len(p))))
        k = min(k, len(p))
        rows.append(dict(group=g, label=int(sub["label"].iloc[0]),
                         score=float(np.mean(p[:k])), n_windows=len(p)))
    return pd.DataFrame(rows)


def tune_threshold(scores: np.ndarray, labels: np.ndarray, beta=0.5):
    """Pick the threshold maximizing F-beta (beta<1 favors precision)."""
    from sklearn.metrics import precision_recall_curve
    if len(np.unique(labels)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(labels, scores)
    prec, rec = prec[:-1], rec[:-1]  # align with thr
    b2 = beta * beta
    denom = (b2 * prec + rec)
    f = np.where(denom > 0, (1 + b2) * prec * rec / denom, 0.0)
    if not len(f):
        return 0.5
    return float(thr[int(np.argmax(f))])


def eval_block(oof, groups, labels, cfg: Config) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

    m = {}
    mask = ~np.isnan(oof)
    yw, pw = labels[mask], oof[mask]
    if len(np.unique(yw)) == 2:
        m["window_roc_auc"] = float(roc_auc_score(yw, pw))
        m["window_pr_auc"] = float(average_precision_score(yw, pw))
    else:
        m["window_roc_auc"] = m["window_pr_auc"] = float("nan")

    vs = video_scores(oof, groups, labels, cfg)
    if len(np.unique(vs["label"])) == 2:
        m["video_roc_auc"] = float(roc_auc_score(vs["label"], vs["score"]))
        m["video_pr_auc"] = float(average_precision_score(vs["label"], vs["score"]))
        thr = tune_threshold(vs["score"].to_numpy(), vs["label"].to_numpy())
        pred = (vs["score"].to_numpy() >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(vs["label"], pred, labels=[0, 1]).ravel()
        m.update(threshold=thr, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                 precision=float(tp / (tp + fp)) if (tp + fp) else 0.0,
                 recall=float(tp / (tp + fn)) if (tp + fn) else 0.0)
    else:
        m["video_roc_auc"] = m["video_pr_auc"] = float("nan")
        m["threshold"] = 0.5
    m["_video_scores"] = vs
    m["_oof"] = oof
    return m


# --------------------------------------------------------------------------- #
# Cross-source generalization: leave-one-source-PAIR-out
# --------------------------------------------------------------------------- #
def source_pair_eval(windows, fcols, cfg: Config, seq=None, model_kind="xgboost",
                     seq_workers=1):
    """Hold out one cheating source × one clean source as test; train on the
    remaining sources. Every source is single-class, so naive leave-one-source-
    out yields single-class folds — pairs keep both classes in test.

    This is the honest generalization metric: the model has never seen EITHER
    test source's recording conditions, so capture artifacts can't carry the
    score. The matched-platform pair (same recording platform for cheating and
    clean) is the headline number.
    """
    from sklearn.metrics import roc_auc_score

    y = windows["label"].to_numpy()
    sources = windows["source"].to_numpy()
    groups = windows["group"].to_numpy()
    cheat_sources = sorted(set(sources[y == 1]))
    clean_sources = sorted(set(sources[y == 0]))

    # Enumerate the valid pairs first (both classes present in train AND test).
    pairs = []  # (cheat_source, clean_source, train_idx, test_idx)
    for cs in cheat_sources:
        for ns in clean_sources:
            test_mask = np.isin(sources, [cs, ns])
            if len(np.unique(y[~test_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
                continue
            pairs.append((cs, ns, np.where(~test_mask)[0], np.where(test_mask)[0]))
    if not pairs:
        return []

    # Compute per-pair test probabilities according to model_kind.
    if model_kind in ("minirocket", "multirocket"):
        # ALL pairs in ONE isolated process: sktime import + numba JIT once, reused.
        from concurrent.futures import ProcessPoolExecutor
        job = (_minirocket_input(seq["X"]), y, [(tr, te) for _, _, tr, te in pairs],
               cfg, seq["X"].shape[-1], model_kind)
        with ProcessPoolExecutor(max_workers=1) as ex:
            probs_per_pair = ex.submit(_minirocket_multi_worker, job).result()
    elif model_kind == "xgboost":
        X = windows[fcols].to_numpy(dtype=float)
        probs_per_pair = []
        for _, _, tr, te in pairs:
            m = xgb_model(y[tr], cfg)
            m.fit(X[tr], y[tr])
            proba = m.predict_proba(X[te])
            probs_per_pair.append(proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(te)))
    else:  # torch sequence model: one isolated process per pair
        from concurrent.futures import ProcessPoolExecutor
        X_seq = seq["X"]
        probs_per_pair = []
        for _, _, tr, te in pairs:
            job = (X_seq[tr], y[tr], X_seq[te], np.arange(len(te)), cfg, model_kind)
            with ProcessPoolExecutor(max_workers=1) as ex:
                _, probs = ex.submit(_seq_fold_worker, job).result()
            probs_per_pair.append(probs)

    results = []
    for (cs, ns, tr, te), probs in zip(pairs, probs_per_pair):
        vs = video_scores(probs, groups[te], y[te], cfg)
        auc = float(roc_auc_score(vs["label"], vs["score"]))
        results.append(dict(
            cheat_source=cs, clean_source=ns, video_auc=round(auc, 4),
            n_test_videos=int(vs.shape[0]),
            n_test_cheat=int((vs["label"] == 1).sum()),
            n_test_clean=int((vs["label"] == 0).sum()),
        ))
    return results


def per_source_scores(oof, windows, cfg: Config):
    """Mean out-of-fold video score per source — the confound tripwire: clean
    sources scoring far apart means capture conditions still leak into scores."""
    vs = video_scores(oof, windows["group"].to_numpy(), windows["label"].to_numpy(), cfg)
    src_of_group = windows.drop_duplicates("group").set_index("group")["source"]
    lab_of_group = windows.drop_duplicates("group").set_index("group")["label_str"]
    vs["source"] = vs["group"].map(src_of_group)
    vs["label_str"] = vs["group"].map(lab_of_group)
    tbl = (vs.groupby(["label_str", "source"])
             .agg(n=("score", "size"), mean_score=("score", "mean"),
                  min_score=("score", "min"), max_score=("score", "max"))
             .reset_index())
    tbl[["mean_score", "min_score", "max_score"]] = tbl[
        ["mean_score", "min_score", "max_score"]].round(3)
    return tbl.to_dict(orient="records")


# --------------------------------------------------------------------------- #
# Winner selection (with deployability rule)
# --------------------------------------------------------------------------- #
def pick_winner(results, deploy_margin: float):
    """results: list of dicts {model, feature_set, metrics}. Returns winner dict.
    Deployability: an l2cs/openface set must beat the best mediapipe-only combo's
    video ROC-AUC by >= deploy_margin, else it is dropped. Model B breaks ties."""
    def auc(r):
        a = r["metrics"]["video_roc_auc"]
        return -1.0 if (a is None or np.isnan(a)) else a

    def is_appearance(fs):  # needs a heavy non-browser gaze model (L2CS/OpenFace)
        return "l2cs" in fs or "openface" in fs

    deployable = [r for r in results if not is_appearance(r["feature_set"])]
    best_mp = max(deployable, key=auc) if deployable else None

    survivors = list(deployable)
    for r in results:
        if not is_appearance(r["feature_set"]):
            continue
        if best_mp is None or auc(r) >= auc(best_mp) + deploy_margin:
            survivors.append(r)
        else:
            r["dropped_for_deployability"] = True

    # sort by AUC desc, then tie-break: prefer deployable (no appearance model), then seq
    def sort_key(r):
        dep_bonus = 0 if is_appearance(r["feature_set"]) else 1
        seq_bonus = 1 if r["model"] in ("gru", "cnn") else 0
        return (round(auc(r), 4), dep_bonus, seq_bonus)

    survivors.sort(key=sort_key, reverse=True)
    return survivors[0], best_mp


# --------------------------------------------------------------------------- #
# Save winner artifacts
# --------------------------------------------------------------------------- #
def save_winner(winner, windows, meta, seq, cfg: Config, export_onnx: bool):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fsets = feature_sets(meta)
    fs_name = winner["feature_set"]
    fcols = fsets[fs_name]
    y = windows["label"].to_numpy()
    groups = windows["group"].to_numpy()
    thr = winner["metrics"]["threshold"]

    artifact = dict(kind=winner["model"], feature_set=fs_name, threshold=thr,
                    metrics={k: v for k, v in winner["metrics"].items() if not k.startswith("_")},
                    window_seconds=cfg.window_seconds, stride_seconds=cfg.stride_seconds,
                    topk_fraction=cfg.topk_fraction, topk_min=cfg.topk_min)

    # Always train an XGBoost-A on the winning feature set for interpretable importances.
    X = windows[fcols].to_numpy(dtype=float)
    xgb_final = xgb_model(y, cfg)
    xgb_final.fit(X, y)
    imp = pd.DataFrame({"feature": fcols, "importance": xgb_final.feature_importances_}) \
        .sort_values("importance", ascending=False)
    imp.to_csv(MODELS_DIR / "feature_importances.csv", index=False)

    if winner["model"] == "xgboost":
        xgb_final.get_booster().save_model(str(MODELS_DIR / "model_xgb.json"))
        artifact["feature_cols"] = fcols
        if export_onnx:
            _export_xgb_onnx(xgb_final, len(fcols))
    elif winner["model"] in ("minirocket", "multirocket"):
        # Fit the ROCKET model on all data in an isolated (sktime-only) process, pickle.
        artifact.update(seq_channels=list(meta["seq_channels"]), seq_len=int(meta["seq_len"]))
        from concurrent.futures import ProcessPoolExecutor
        job = (_minirocket_input(seq["X"]), y, cfg, len(meta["seq_channels"]), winner["model"])
        with ProcessPoolExecutor(max_workers=1) as ex:
            ex.submit(_minirocket_final_worker, job).result()
        if export_onnx:
            print(f"  (ONNX export not supported for {winner['model']}; skipped.)")
    else:
        # Retrain the winning sequence net on all data in an isolated process
        # (torch-only, no XGBoost) to avoid the OpenMP conflict, then save.
        artifact.update(seq_channels=list(meta["seq_channels"]), seq_len=int(meta["seq_len"]))
        from concurrent.futures import ProcessPoolExecutor
        job = (seq["X"], y, cfg, winner["model"], int(meta["seq_len"]),
               len(meta["seq_channels"]), export_onnx)
        with ProcessPoolExecutor(max_workers=1) as ex:
            ex.submit(_seq_final_worker, job).result()

    (MODELS_DIR / "model.json").write_text(json.dumps(artifact, indent=2))
    return imp


def _seq_final_worker(job):
    """Retrain the sequence model on ALL data and persist it, in an isolated
    torch-only process (uses multiple threads for this single big training)."""
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    X_seq, y, cfg, kind, seq_len, n_channels, export_onnx = job
    net, mean, std, device = _train_seq_net(X_seq, y, cfg, kind=kind, device=torch.device("cpu"))
    _save_seq(net, mean, std)
    if export_onnx:
        _export_seq_onnx(net, mean, std, seq_len, n_channels, device)


def _save_seq(net, mean, std):
    import torch
    torch.save({"state_dict": net.state_dict(), "mean": mean, "std": std,
                "kind": net.__class__.__name__}, MODELS_DIR / "model_gru.pt")
    np.savez(MODELS_DIR / "seq_scaler.npz", mean=mean, std=std)


def _export_xgb_onnx(model, n_features):
    from skl2onnx.common.data_types import FloatTensorType
    itypes = [("input", FloatTensorType([None, n_features]))]
    # onnxmltools handles XGBoost tree models most reliably; fall back to skl2onnx.
    try:
        try:
            from onnxmltools import convert_xgboost
            onx = convert_xgboost(model, initial_types=itypes)
        except Exception:
            from skl2onnx import convert_sklearn
            onx = convert_sklearn(model, initial_types=itypes, target_opset=12)
        (MODELS_DIR / "model.onnx").write_bytes(onx.SerializeToString())
        print("  ONNX export -> models/model.onnx")
    except Exception as e:
        print(f"  (ONNX export skipped: {e}. `pip install onnxmltools onnx` to enable.)")


def _export_seq_onnx(net, mean, std, seq_len, n_channels, device):
    try:
        import torch
        net_cpu = net.to("cpu").eval()
        dummy = torch.zeros(1, seq_len, n_channels)
        torch.onnx.export(net_cpu, dummy, str(MODELS_DIR / "model.onnx"),
                          input_names=["input"], output_names=["logit"],
                          dynamic_axes={"input": {0: "batch"}}, opset_version=17)
        print("  ONNX export -> models/model.onnx  (apply (x-mean)/std before feeding)")
    except Exception as e:
        print(f"  (ONNX export skipped: {e})")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 3: train + compare + ship winner.")
    ap.add_argument("--skip-seq", action="store_true", help="skip Model B (sequence net)")
    ap.add_argument("--skip-xgb", action="store_true",
                    help="skip Model A (XGBoost); e.g. train only the ROCKET seq model")
    ap.add_argument("--seq-kind", default="gru",
                    choices=["gru", "cnn", "minirocket", "multirocket"],
                    help="Model B net (gru/cnn) or Model C ROCKET (minirocket/multirocket; "
                         "needs sktime). multirocket is the stronger ROCKET variant here.")
    ap.add_argument("--seq-device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                    help="device for Model B (auto: CPU for small data, MPS when large)")
    ap.add_argument("--seq-workers", type=int, default=None,
                    help="parallel CV folds for Model B (default: ~all cores)")
    ap.add_argument("--deploy-margin", type=float, default=0.02,
                    help="min video ROC-AUC gain for an appearance backend to win")
    ap.add_argument("--export-onnx", action="store_true")
    ap.add_argument("--skip-source-eval", action="store_true",
                    help="skip the slow leave-one-source-pair-out cross-source report "
                         "(ships the winner faster; grouped-CV metrics are unaffected)")
    args = ap.parse_args(argv)

    cfg = DEFAULT_CONFIG
    cfg_path = MODELS_DIR / "config.json"
    if cfg_path.exists():
        cfg = Config.from_json(cfg_path)

    windows, meta, seq = load_data()
    y = windows["label"].to_numpy()
    groups = windows["group"].to_numpy()
    fsets = feature_sets(meta)

    n_pos_vids = windows[windows.label == 1]["group"].nunique()
    n_neg_vids = windows[windows.label == 0]["group"].nunique()
    print(f"windows={len(windows)}  videos: cheating={n_pos_vids} clean={n_neg_vids}")
    if n_pos_vids < 2 or n_neg_vids < 2:
        print("⚠️  Need >= 2 videos per class for grouped CV. Add more data.")
        return 1

    splitter, split_name, nsp = make_splitter(y, groups, cfg)
    print(f"CV: {split_name}   feature sets: {list(fsets)}   "
          f"seq: {'off' if args.skip_seq else args.seq_kind}\n")

    results = []
    # Model A over each feature set
    if args.skip_xgb:
        print("[A/xgboost] skipped (--skip-xgb)")
    else:
        for fs_name, fcols in fsets.items():
            X = windows[fcols].to_numpy(dtype=float)
            oof = oof_xgb(X, y, groups, splitter, cfg)
            m = eval_block(oof, groups, y, cfg)
            results.append(dict(model="xgboost", feature_set=fs_name, metrics=m))
            print(f"[A/xgboost  {fs_name:16s}] video AUC={m['video_roc_auc']:.3f} "
                  f"PR={m['video_pr_auc']:.3f} | window AUC={m['window_roc_auc']:.3f} "
                  f"| P={m.get('precision',0):.2f} R={m.get('recall',0):.2f}")

    # Model C (MiniROCKET) or Model B (GRU/CNN) over the raw *sequence* channels
    # (seq channels are backend-agnostic, so a single "mediapipe" combo covers it).
    if not args.skip_seq and args.seq_kind in ("minirocket", "multirocket"):
        if not _minirocket_available():
            print(f"[C] sktime not available — skipping {args.seq_kind} "
                  "(pip install sktime numba to enable).")
        else:
            print(f"[C] {args.seq_kind} CV: {nsp} folds (isolated sktime process)")
            oof = oof_minirocket(seq["X"], y, groups, splitter, cfg, kind=args.seq_kind)
            m = eval_block(oof, groups, y, cfg)
            results.append(dict(model=args.seq_kind, feature_set="mediapipe", metrics=m))
            print(f"[C/{args.seq_kind:11s} {'mediapipe':16s}] video AUC={m['video_roc_auc']:.3f} "
                  f"PR={m['video_pr_auc']:.3f} | window AUC={m['window_roc_auc']:.3f} "
                  f"| P={m.get('precision',0):.2f} R={m.get('recall',0):.2f}")
    elif not args.skip_seq:
        if not _torch_available():
            print("[B] torch not available — skipping Model B (install torch to enable).")
        else:
            import os
            n_cpu = os.cpu_count() or 4
            workers = args.seq_workers or max(1, min(nsp, n_cpu - 1))
            print(f"[B] {args.seq_kind} CV: {nsp} folds across {workers} parallel CPU workers")
            oof = oof_seq(seq["X"], y, groups, splitter, cfg, kind=args.seq_kind, workers=workers)
            m = eval_block(oof, groups, y, cfg)
            results.append(dict(model=args.seq_kind, feature_set="mediapipe", metrics=m))
            print(f"[B/{args.seq_kind:9s} {'mediapipe':16s}] video AUC={m['video_roc_auc']:.3f} "
                  f"PR={m['video_pr_auc']:.3f} | window AUC={m['window_roc_auc']:.3f} "
                  f"| P={m.get('precision',0):.2f} R={m.get('recall',0):.2f}")

    if not results:
        print("⚠️  No models were trained (check --skip-xgb / --skip-seq / sktime install).")
        return 1
    winner, best_mp = pick_winner(results, args.deploy_margin)
    print(f"\n🏆 winner: {winner['model']} on {winner['feature_set']} "
          f"(video ROC-AUC={winner['metrics']['video_roc_auc']:.3f}, "
          f"threshold={winner['metrics']['threshold']:.3f})")

    # Cross-source generalization for the winner: leave-one-source-pair-out.
    # This is the headline honesty metric — grouped CV above is the optimistic one.
    pair_results, per_source = [], []
    if args.skip_source_eval:
        print("\n(cross-source eval skipped: --skip-source-eval)")
    elif "source" in windows.columns and windows["source"].nunique() >= 2:
        fs_cols = fsets.get(winner["feature_set"], fsets[next(iter(fsets))])
        pair_results = source_pair_eval(windows, fs_cols, cfg, seq=seq,
                                        model_kind=winner["model"])
        if pair_results:
            print("\ncross-source (leave-one-source-pair-out, video AUC):")
            for p in pair_results:
                print(f"  {p['cheat_source']:14s} vs {p['clean_source']:10s} "
                      f"AUC={p['video_auc']:.3f}  ({p['n_test_cheat']}+{p['n_test_clean']} videos)")
            mean_auc = float(np.mean([p["video_auc"] for p in pair_results]))
            print(f"  mean pair AUC = {mean_auc:.3f}")
        per_source = per_source_scores(
            np.asarray(winner["metrics"]["_oof"]), windows, cfg) \
            if "_oof" in winner["metrics"] else []

    imp = save_winner(winner, windows, meta, seq, cfg, args.export_onnx)

    # persist all metrics + per-video OOF scores for the evaluation report
    metrics_out = []
    for r in results:
        mm = {k: v for k, v in r["metrics"].items() if not k.startswith("_")}
        mm.update(model=r["model"], feature_set=r["feature_set"],
                  dropped_for_deployability=r.get("dropped_for_deployability", False),
                  is_winner=(r is winner))
        metrics_out.append(mm)
    (MODELS_DIR / "metrics.json").write_text(json.dumps(dict(
        combos=metrics_out, source_pairs=pair_results, per_source=per_source,
    ), indent=2))
    winner["metrics"]["_video_scores"].to_csv(MODELS_DIR / "cv_video_scores.csv", index=False)

    print("\ntop features:")
    for _, row in imp.head(12).iterrows():
        print(f"  {row['importance']:.3f}  {row['feature']}")
    print(f"\nartifacts -> {MODELS_DIR}")
    print("next: python src/report.py --eval   (and per-video: python src/predict.py <video>)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
