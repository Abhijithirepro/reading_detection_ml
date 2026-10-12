"""
train.py — Stage 3: train the 1D-CNN on raw gaze sequences, evaluate it with
leakage-safe grouped cross-validation, and ship it.

What it does
------------
* Loads data/windows/windows.parquet + sequences.npz + models/config.json.
* The CNN reads each window's raw sequence channels (gaze_x, gaze_y, yaw, pitch,
  ear, mouth) — not the engineered window features, which only feed the
  human-readable reason tags in predict.py.
* CV: StratifiedGroupKFold grouped by video (windows from one video never span
  train/val). Falls back to LeaveOneGroupOut when videos/class is small.
* Aggregates out-of-fold window probabilities to a VIDEO-level verdict
  (mean of the top-k window probabilities), tunes a precision-favoring threshold,
  and reports window- and video-level ROC-AUC / PR-AUC / confusion matrix.
* Cross-source check: leave-one-source-pair-out video AUC.
* Retrains the CNN on all data and saves weights + threshold + full metrics to
  models/, plus an optional ONNX export for the browser frontend.

Usage:
    python src/train.py
    python src/train.py --export-onnx          # also write models/model.onnx
    python src/train.py --skip-source-eval
"""

from __future__ import annotations

import os

# PyTorch ships its own OpenMP runtime; tolerate a second copy (e.g. scikit-learn's)
# in the same process instead of aborting. Set before torch is ever imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

from common import Config, DEFAULT_CONFIG, WINDOWS_DIR, MODELS_DIR

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Data loading
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
# 1D-CNN on raw sequences
# --------------------------------------------------------------------------- #
def _torch_available():
    # check WITHOUT importing torch into the parent process (folds run in subprocesses)
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def _build_seq_net(n_channels):
    import torch.nn as nn

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

    return CNNNet(n_channels)


def pick_seq_device(n_windows: int, override: str = None):
    """Choose the device for the CNN: an explicit override, else CUDA if present,
    else CPU. Apple MPS is opt-in — at these dataset sizes (hundreds–low thousands
    of windows) its kernel-launch overhead outweighs the speedup."""
    import torch
    if override and override != "auto":
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _train_seq_net(Xtr, ytr, cfg: Config, epochs=40, device=None, patience=6):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(cfg.random_state)
    if device is None:
        device = pick_seq_device(len(Xtr))
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    Xn = (Xtr - mean) / std

    net = _build_seq_net(Xtr.shape[-1]).to(device)
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
    """Train + predict ONE CV fold in an isolated torch process. Runs
    single-threaded so N folds fan out across N cores without oversubscription."""
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch
    torch.set_num_threads(1)
    Xtr, ytr, Xte, te_idx, cfg = job
    net, mean, std, dev = _train_seq_net(Xtr, ytr, cfg, device=torch.device("cpu"))
    return te_idx, _predict_seq_net(net, mean, std, dev, Xte)


def oof_seq(X, y, groups, splitter, cfg: Config, workers=1):
    """Out-of-fold predictions for the CNN, folds run in parallel."""
    folds = list(splitter.split(X, y, groups))
    jobs = [(X[tr], y[tr], X[te], te, cfg) for tr, te in folds]
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
def source_pair_eval(windows, cfg: Config, seq):
    """Hold out one cheating source × one clean source as test; train on the
    remaining sources. Every source is single-class, so naive leave-one-source-
    out yields single-class folds — pairs keep both classes in test.

    This is the honest generalization metric: the model has never seen EITHER
    test source's recording conditions, so capture artifacts can't carry the
    score. The matched-platform pair (same recording platform for cheating and
    clean) is the headline number.
    """
    from concurrent.futures import ProcessPoolExecutor
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

    # One isolated torch process per pair.
    X_seq = seq["X"]
    probs_per_pair = []
    for _, _, tr, te in pairs:
        job = (X_seq[tr], y[tr], X_seq[te], np.arange(len(te)), cfg)
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
# Save artifacts
# --------------------------------------------------------------------------- #
def save_model(metrics, meta, seq, y, cfg: Config, export_onnx: bool):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = dict(kind="cnn", feature_set="mediapipe", threshold=metrics["threshold"],
                    metrics={k: v for k, v in metrics.items() if not k.startswith("_")},
                    window_seconds=cfg.window_seconds, stride_seconds=cfg.stride_seconds,
                    topk_fraction=cfg.topk_fraction, topk_min=cfg.topk_min,
                    seq_channels=list(meta["seq_channels"]), seq_len=int(meta["seq_len"]))

    # Retrain the CNN on all data in an isolated torch process, then save.
    from concurrent.futures import ProcessPoolExecutor
    job = (seq["X"], y, cfg, int(meta["seq_len"]), len(meta["seq_channels"]), export_onnx)
    with ProcessPoolExecutor(max_workers=1) as ex:
        ex.submit(_seq_final_worker, job).result()

    (MODELS_DIR / "model.json").write_text(json.dumps(artifact, indent=2))


def _seq_final_worker(job):
    """Retrain the CNN on ALL data and persist it, in an isolated torch process
    (uses multiple threads for this single big training)."""
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    X_seq, y, cfg, seq_len, n_channels, export_onnx = job
    net, mean, std, device = _train_seq_net(X_seq, y, cfg, device=torch.device("cpu"))
    _save_seq(net, mean, std)
    if export_onnx:
        _export_seq_onnx(net, mean, std, seq_len, n_channels, device)


def _save_seq(net, mean, std):
    import torch
    torch.save({"state_dict": net.state_dict(), "mean": mean, "std": std,
                "kind": net.__class__.__name__}, MODELS_DIR / "model_cnn.pt")
    np.savez(MODELS_DIR / "seq_scaler.npz", mean=mean, std=std)


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
    ap = argparse.ArgumentParser(description="Stage 3: train + evaluate + ship the CNN.")
    ap.add_argument("--seq-device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
                    help="device for the CNN (auto: CUDA if present, else CPU)")
    ap.add_argument("--seq-workers", type=int, default=None,
                    help="parallel CV folds (default: ~all cores)")
    ap.add_argument("--export-onnx", action="store_true")
    ap.add_argument("--skip-source-eval", action="store_true",
                    help="skip the slow leave-one-source-pair-out cross-source report "
                         "(ships the model faster; grouped-CV metrics are unaffected)")
    args = ap.parse_args(argv)

    cfg = DEFAULT_CONFIG
    cfg_path = MODELS_DIR / "config.json"
    if cfg_path.exists():
        cfg = Config.from_json(cfg_path)

    windows, meta, seq = load_data()
    y = windows["label"].to_numpy()
    groups = windows["group"].to_numpy()

    n_pos_vids = windows[windows.label == 1]["group"].nunique()
    n_neg_vids = windows[windows.label == 0]["group"].nunique()
    print(f"windows={len(windows)}  videos: cheating={n_pos_vids} clean={n_neg_vids}")
    if n_pos_vids < 2 or n_neg_vids < 2:
        print("⚠️  Need >= 2 videos per class for grouped CV. Add more data.")
        return 1
    if not _torch_available():
        print("⚠️  torch is not installed — it is required to train the CNN.")
        return 1

    splitter, split_name, nsp = make_splitter(y, groups, cfg)
    print(f"CV: {split_name}   model: cnn\n")

    n_cpu = os.cpu_count() or 4
    workers = args.seq_workers or max(1, min(nsp, n_cpu - 1))
    print(f"[cnn] CV: {nsp} folds across {workers} parallel CPU workers")
    oof = oof_seq(seq["X"], y, groups, splitter, cfg, workers=workers)
    m = eval_block(oof, groups, y, cfg)
    print(f"[cnn] video AUC={m['video_roc_auc']:.3f} "
          f"PR={m['video_pr_auc']:.3f} | window AUC={m['window_roc_auc']:.3f} "
          f"| P={m.get('precision',0):.2f} R={m.get('recall',0):.2f} "
          f"| threshold={m['threshold']:.3f}")

    # Cross-source generalization: leave-one-source-pair-out.
    # This is the headline honesty metric — grouped CV above is the optimistic one.
    pair_results, per_source = [], []
    if args.skip_source_eval:
        print("\n(cross-source eval skipped: --skip-source-eval)")
    elif "source" in windows.columns and windows["source"].nunique() >= 2:
        pair_results = source_pair_eval(windows, cfg, seq)
        if pair_results:
            print("\ncross-source (leave-one-source-pair-out, video AUC):")
            for p in pair_results:
                print(f"  {p['cheat_source']:14s} vs {p['clean_source']:10s} "
                      f"AUC={p['video_auc']:.3f}  ({p['n_test_cheat']}+{p['n_test_clean']} videos)")
            mean_auc = float(np.mean([p["video_auc"] for p in pair_results]))
            print(f"  mean pair AUC = {mean_auc:.3f}")
        per_source = per_source_scores(np.asarray(m["_oof"]), windows, cfg)

    save_model(m, meta, seq, y, cfg, args.export_onnx)

    # persist metrics + per-video OOF scores for the evaluation report
    mm = {k: v for k, v in m.items() if not k.startswith("_")}
    mm.update(model="cnn", feature_set="mediapipe", is_winner=True)
    (MODELS_DIR / "metrics.json").write_text(json.dumps(dict(
        combos=[mm], source_pairs=pair_results, per_source=per_source,
    ), indent=2))
    m["_video_scores"].to_csv(MODELS_DIR / "cv_video_scores.csv", index=False)

    print(f"\nartifacts -> {MODELS_DIR}")
    print("next: python src/report.py --eval   (and per-video: python src/predict.py <video>)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
