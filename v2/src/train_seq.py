"""
train_seq.py (v2, experiment) — a 1D-CNN on the raw per-frame channels, trained
on the same strong labels and scored with the same folds / metrics as train.py,
so RESULTS_LOG.md compares the two model families like for like.

Input per window: 8 s of 10 channels at 12.5 Hz (100 steps) — eye_h, eye_v,
gaze_x, gaze_y, jaw, mouth, blink, yaw, pitch, speaking — each robust-normalised
per video (median / MAD), so a person's baseline is removed before the network
sees it. Windows are sliced on the fly from per-video arrays (no huge tensor).

This is an experiment runner: it logs results but does not ship a model (the web
app serves whatever v2/models/model.json describes). --compare-xgb re-runs the
XGBoost recipe on the identical windows and adds an XGB+CNN average, so the log
gets a true head-to-head.

Usage:
    python src/train_seq.py
    python src/train_seq.py --epochs 12 --exp cnn-strong-labels
    python src/train_seq.py --pseudo-frac 0.15 --clean-speech-weight 1 \
        --video-stat speech_prob_mean --compare-xgb --exp cnn-vs-xgb
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from common import Config, DEFAULT_CONFIG, FEATURES_DIR, MODELS_DIR, RESULTS_DIR
import build_windows as bw
import labels as lab
import results_log
import train as tr

CHANNELS = ["eye_h", "eye_v", "gaze_x", "gaze_y", "jaw", "mouth", "blink", "yaw", "pitch", "speaking"]
DECIM = 2            # 25 Hz -> 12.5 Hz


def _video_arrays(job):
    pq, cfg_dict = job
    cfg = Config(**cfg_dict)
    df = pd.read_parquet(pq)
    c = bw.clean_video(df, cfg)
    arr = []
    for ch in CHANNELS:
        x = bw._fill(c[ch]).astype(np.float32)
        if ch not in ("blink", "speaking"):
            med = np.median(x)
            mad = 1.4826 * np.median(np.abs(x - med)) + 1e-4
            x = np.clip((x - med) / mad, -8, 8)
        arr.append(x[::DECIM])
    return Path(pq).stem, np.stack(arr, 1), c["t0"], c["dt"] * DECIM


def load_arrays(windows, cfg, workers):
    from dataclasses import asdict
    vids = windows["video"].unique()
    jobs = [(str(FEATURES_DIR / f"{v}.parquet"), asdict(cfg)) for v in vids]
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for stem, arr, t0, dt in ex.map(_video_arrays, jobs):
            out[stem] = (arr, t0, dt)
    return out


def window_index(windows, arrays, cfg):
    """(video, start sample) for every window row, on the decimated grid."""
    starts = np.zeros(len(windows), dtype=np.int64)
    for v, idx in windows.groupby("video").groups.items():
        _, t0, dt = arrays[v]
        starts[idx] = np.round((windows.loc[idx, "win_start_s"].to_numpy() - t0) / dt).astype(int)
    return starts


def build_net(c, hidden=48):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv1d(c, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Conv1d(hidden, hidden, 5, padding=8, dilation=4), nn.BatchNorm1d(hidden), nn.ReLU(),
            )
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * hidden, 1))

        def forward(self, x):                 # x: (B, L, C)
            z = self.f(x.transpose(1, 2))
            pooled = torch.cat([z.mean(-1), z.amax(-1)], 1)   # average + max over time
            return self.head(pooled).squeeze(-1)

    return Net()


def batches(arrays, vids, starts, L, idx):
    X = np.stack([arrays[vids[i]][0][starts[i]:starts[i] + L] for i in idx])
    return X


def fit_predict(arrays, vids, starts, tr_idx, y_tr, w_tr, te_sets, L, epochs, seed, device):
    """Train on rows tr_idx (labels y_tr, weights w_tr), return probabilities per te set."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = build_net(len(CHANNELS)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * 200)
    p = w_tr / w_tr.sum()
    for _ in range(epochs):
        net.train()
        for _ in range(200):                      # weighted sampling = class/video balance
            j = rng.choice(len(tr_idx), size=256, p=p)
            xb = torch.tensor(batches(arrays, vids, starts, L, tr_idx[j]), device=device)
            yb = torch.tensor(y_tr[j], dtype=torch.float32, device=device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    outs = []
    with torch.no_grad():
        for te in te_sets:
            pr = np.zeros(len(te))
            for k in range(0, len(te), 2048):
                xb = torch.tensor(batches(arrays, vids, starts, L, te[k:k + 2048]), device=device)
                pr[k:k + 2048] = torch.sigmoid(net(xb)).cpu().numpy()
            outs.append(pr)
    return outs


def cv_cnn(windows, videos, arrays, starts, L, args, device, fold_unrev=None):
    """Same protocol as train.cross_validate, with the CNN as the model:
    plain run (fold_unrev=None) or self-training with fold-specific pseudo-labels
    and A/B cross-scoring of the unreviewed cheating videos."""
    vids = windows["video"].to_numpy()
    folds = tr.assign_folds(videos, args.folds, args.seed)
    fold = windows["video"].map(folds).to_numpy()
    trainable = tr.trainable_mask(windows)
    test_only = fold == -1
    oof = np.full(len(windows), np.nan)
    acc = np.zeros(len(windows))
    fold_probs = {}
    use_pseudo = args.pseudo_frac > 0 and fold_unrev is not None
    if use_pseudo:
        unrev = sorted(windows.loc[test_only, "video"].unique())
        rng = np.random.default_rng(args.seed)
        rng.shuffle(unrev)
        halves = [set(unrev[0::2]), set(unrev[1::2])]
    for k in range(args.folds):
        base = trainable & (fold != k)
        te_idx = np.where(fold == k)[0]
        if use_pseudo:
            pseudo = tr.pseudo_labels(windows, fold_unrev[k], args.pseudo_frac)
            plans = []
            for own, other in ((halves[0], halves[1]), (halves[1], halves[0])):
                rows = np.concatenate([pseudo[v] for v in own if v in pseudo] or [np.array([], int)])
                plans.append((rows, np.where(test_only & windows["video"].isin(other).to_numpy())[0]))
        else:
            plans = [(np.array([], int), np.where(test_only)[0])]
        fold_probs[k] = np.full(len(windows), np.nan)
        preds = []
        for rows, score_idx in plans:
            idx, sub = tr.training_frame(windows, base, rows)
            w_tr = tr.sample_weights(sub, args.reviewed_neg_weight, args.pseudo_weight,
                                     args.clean_speech_weight)
            p_te, p_sc = fit_predict(arrays, vids, starts, idx, sub["wlabel"].to_numpy().astype(float),
                                     w_tr, [te_idx, score_idx], L, args.epochs, args.seed, device)
            preds.append(p_te)
            acc[score_idx] += p_sc
            fold_probs[k][score_idx] = p_sc
        oof[te_idx] = np.mean(preds, axis=0)
        print(f"  fold {k}: {'self-training' if use_pseudo else 'plain'}, scored {len(te_idx)}", flush=True)
    oof[test_only] = acc[test_only] / args.folds
    return oof, fold_probs


def _xgb_oof(job):
    """XGBoost recipe on the same window rows, in a SPAWNED process: PyTorch and
    XGBoost each bring an OpenMP runtime, and both in one process crashes
    (OMP pthread_mutex_init). The child never imports torch."""
    keep, feats_drop, a = job
    windows, videos, all_feats = tr.load_data()
    windows = windows[keep].reset_index(drop=True)
    cfg = Config.from_json(MODELS_DIR / "config.json") if (MODELS_DIR / "config.json").exists() else DEFAULT_CONFIG
    feats = tr.feature_list(all_feats, feats_drop)
    scored, fold_unrev = tr.cross_validate(windows, videos, feats, cfg, a["folds"],
                                           a["reviewed_neg_weight"], a["seed"],
                                           clean_speech_weight=a["clean_speech_weight"])
    if a["pseudo_frac"] > 0:
        scored, _ = tr.cross_validate(windows, videos, feats, cfg, a["folds"], a["reviewed_neg_weight"],
                                      a["seed"], a["pseudo_frac"], a["pseudo_weight"], fold_unrev,
                                      a["clean_speech_weight"])
    return scored["prob"].to_numpy()


def _report_rows(name, ev):
    ps = pd.DataFrame(ev["per_second"])
    vm = ev["video"]
    spans = (sum(int(s.split("/")[0]) for s in ps.spans_caught), sum(int(s.split("/")[1]) for s in ps.spans_caught))
    recall = float((ps.recall * ps.reading_s).sum() / max(ps.reading_s.sum(), 1))
    precision = float((ps.precision * ps.flagged_s).sum() / max(ps.flagged_s.sum(), 1))
    cf = ev["clean"]
    row = dict(model=name, spans_caught=f"{spans[0]}/{spans[1]}", sec_auc_mean=float(ps.auc.mean()),
               sec_precision=precision, sec_recall=recall, iou_mean=float(ps.iou.mean()),
               clean_flagged=f"{cf.flagged_s.sum() / max(cf.seconds.sum(), 1):.2%}",
               video_auc=vm["auc"], caught=f"{vm['tp']}/{vm['tp'] + vm['fn']}", false_alarms=vm["fp"],
               matched_auc=vm["matched_auc"])
    for _, r in ps.iterrows():
        short = r["video"].replace("combined_", "").split("_")[1]
        row[f"{short} auc/rec/spans"] = f"{r['auc']:.2f} / {r['recall']:.2f} / {r['spans_caught']}"
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description="v2 experiment: 1D-CNN reading detector.")
    ap.add_argument("--exp", default="cnn-strong-labels")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reviewed-neg-weight", type=float, default=0.5)
    ap.add_argument("--pseudo-frac", type=float, default=0.0)
    ap.add_argument("--pseudo-weight", type=float, default=0.5)
    ap.add_argument("--clean-speech-weight", type=float, default=0.0)
    ap.add_argument("--clean-budget", type=float, default=0.02)
    ap.add_argument("--video-stat", default="reading_seconds")
    ap.add_argument("--compare-xgb", action="store_true",
                    help="also run the XGBoost recipe on the same windows + an XGB/CNN average")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = Config.from_json(MODELS_DIR / "config.json") if (MODELS_DIR / "config.json").exists() else DEFAULT_CONFIG
    windows, videos, all_feats = tr.load_data()
    reviews = lab.load_reviews()
    print(f"loading per-video arrays ({windows.video.nunique()} videos)...", flush=True)
    arrays = load_arrays(windows, cfg, args.workers)
    L = int(round(cfg.read_window_seconds * cfg.target_fps / DECIM))
    starts = window_index(windows, arrays, cfg)
    ok = np.array([starts[i] + L <= len(arrays[v][0]) for i, v in enumerate(windows.video)])
    windows = windows[ok].reset_index(drop=True)
    starts = starts[ok]

    oof, fold_unrev = cv_cnn(windows, videos, arrays, starts, L, args, device)
    if args.pseudo_frac > 0:
        print("CNN self-training pass...", flush=True)
        oof, _ = cv_cnn(windows, videos, arrays, starts, L, args, device, fold_unrev)
    scored = windows.assign(prob=oof)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)       # save before anything else can fail
    scored[["video", "win_start_s", "win_end_s", "prob"]].to_parquet(
        RESULTS_DIR / f"oof_windows__{args.exp}.parquet", index=False)
    ev = tr.evaluate_cv(scored, videos, cfg, reviews, args.clean_budget, args.video_stat)

    contenders = [("CNN", ev)]
    if args.compare_xgb:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        print("XGBoost on the same windows (separate process)...", flush=True)
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as ex:
            x_prob = ex.submit(_xgb_oof, (ok, [], vars(args))).result()
        ev_x = tr.evaluate_cv(windows.assign(prob=x_prob), videos, cfg, reviews,
                              args.clean_budget, args.video_stat)
        ens = windows.assign(prob=(x_prob + oof) / 2)
        ev_e = tr.evaluate_cv(ens, videos, cfg, reviews, args.clean_budget, args.video_stat)
        contenders = [("XGBoost", ev_x), ("CNN", ev), ("XGB+CNN average", ev_e)]

    rows = [_report_rows(n, e) for n, e in contenders]
    table = pd.DataFrame(rows)
    print("\n" + table.T.to_string(header=False))
    recipe = (f"self-training {args.pseudo_frac:.0%} (weight {args.pseudo_weight}), "
              f"clean-speech weight {args.clean_speech_weight}" if args.pseudo_frac > 0
              else "no self-training")
    results_log.append(
        args.exp,
        f"1D-CNN on raw per-frame channels ({', '.join(CHANNELS)}), 8 s @ 12.5 Hz, {args.epochs} epochs; "
        f"{recipe}; verdict stat {args.video_stat}. Same folds, leak-free pseudo-labels, thresholds "
        f"and metrics as the XGBoost runs"
        + (" — XGBoost re-run on the identical window set for a head-to-head." if args.compare_xgb else ".")
        + f" {args.note}",
        {"Head-to-head (out-of-fold)": results_log.table(rows),
         "CNN per-second vs manual reviews": results_log.table(
             ev["per_second"], ["video", "reading_s", "flagged_s", "auc", "precision", "recall", "iou",
                                "spans_caught"])},
        dict(args=vars(args), contenders={n: dict(per_second=e["per_second"], video=e["video"],
                                                  stat_auc=e["stat_auc"]) for n, e in contenders}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
