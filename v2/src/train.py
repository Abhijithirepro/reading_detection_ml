"""
train.py (v2) — Stage 3: train the per-second READING detector on strong labels,
evaluate it honestly against the manual reviews, and ship it.

Model: gradient-boosted trees (XGBoost) on the v2 window features. Trees are
interpretable (feature importances become the reason tags), robust with only a
few reviewed candidates, and fast on the server.

Cross-validation (5 folds, every video in exactly one role):
  * reviewed cheating videos -> one per fold (folds 0..R-1). Their per-second
    scores come from a model that never saw them.
  * clean videos -> spread over the 5 folds; scored out-of-fold.
  * UNREVIEWED cheating videos -> never trained on (no window labels). Each is
    scored by every fold model (mean), so they are a clean video-level test set.

Operating points, both picked on out-of-fold scores:
  * second_threshold: per-second reading probability -> "reading" (max F1 on the
    reviewed videos, with at most `--clean-budget` of clean seconds flagged)
  * video rule: total reading seconds >= video_threshold -> CHEATING (F0.5, favours
    precision like v1)

Every run is appended to RESULTS_LOG.md (+ results/*.json).

Usage:
    python src/train.py                         # CV + log + ship to v2/models/
    python src/train.py --exp eyes-only --no-ship --drop-group speech
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

from common import Config, DEFAULT_CONFIG, WINDOWS_DIR, MODELS_DIR, RESULTS_DIR
import labels as lab
import reading
import results_log
from evaluate import per_second_metrics, video_metrics

warnings.filterwarnings("ignore")

META_COLS = ["video", "label_str", "label", "source", "reviewed", "win_start_s",
             "win_end_s", "label_frac", "wlabel"]

# Feature groups, for ablations (--drop-group).
GROUPS = {
    "speech": ["speak_frac", "mouth_act", "speak_act", "jaw_act", "jaw_mean", "smile_mean",
               "speak_ctx30", "speak_prev10"],
    "sweeps": ["h_sweeps_dom", "h_sweeps_max", "gx_sweeps_dom", "h_skew_dom", "h_skew_abs",
               "h_prog_frac", "h_ac_peak", "ctx_sweeps_speak"],
    "ctx": ["ctx_*"],          # answer-context features (prefix match)
}

VIDEO_STATS = ["reading_seconds", "reading_share", "reading_share_of_speech", "top10_mean",
               "p95", "speech_prob_mean", "speech_prob_p90"]

XGB_PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.7, min_child_weight=5, reg_lambda=1.0,
                  tree_method="hist", eval_metric="logloss")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data():
    wpath = WINDOWS_DIR / "windows.parquet"
    if not wpath.exists():
        print(f"Missing {wpath}. Run build_windows.py first.")
        sys.exit(1)
    windows = pd.read_parquet(wpath)
    videos = pd.read_parquet(WINDOWS_DIR / "videos.parquet")
    feats = json.loads((WINDOWS_DIR / "_feature_columns.json").read_text())["feature_cols"]
    return windows, videos, feats


def feature_list(all_feats, drop_groups=()):
    drop = set()
    for g in drop_groups:
        for f in GROUPS.get(g, []):
            if f.endswith("*"):
                drop |= {a for a in all_feats if a.startswith(f[:-1])}
            else:
                drop |= {f, f + "_vrel"}
    return [f for f in all_feats if f not in drop and f != "valid_ratio"]


def assign_folds(videos: pd.DataFrame, n_folds: int, seed: int) -> dict:
    """{video: fold}; unreviewed cheating videos get -1 (test-only)."""
    rng = np.random.default_rng(seed)
    folds = {}
    rev = sorted(videos.loc[videos.reviewed, "video"])
    for i, v in enumerate(rev):
        folds[v] = i % n_folds
    clean = videos.loc[videos.label_str == "clean", "video"].to_numpy().copy()
    rng.shuffle(clean)
    for i, v in enumerate(clean):
        folds[v] = i % n_folds
    for v in videos.loc[(videos.label_str == "cheating") & ~videos.reviewed, "video"]:
        folds[v] = -1
    return folds


def sample_weights(df: pd.DataFrame, reviewed_neg_weight: float,
                   pseudo_weight: float = 0.5, clean_speech_weight: float = 0.0) -> np.ndarray:
    """Reviewed reading, pseudo-labelled reading, reviewed not-reading and clean
    windows each get a fixed share of the total weight; inside a share every VIDEO
    counts equally (long recordings don't dominate)."""
    w = np.zeros(len(df))
    y = df["wlabel"].to_numpy()
    pseudo = df["pseudo"].to_numpy(bool) if "pseudo" in df else np.zeros(len(df), bool)
    groups = {
        "pos": ((y == 1) & ~pseudo, 1.0),
        "pseudo_pos": ((y == 1) & pseudo, pseudo_weight),
        "rev_neg": ((y == 0) & df["reviewed"].to_numpy(), reviewed_neg_weight),
        "clean_neg": ((y == 0) & (df["label_str"] == "clean").to_numpy(), 1.0),
    }
    if clean_speech_weight > 0:
        # clean candidates ANSWERING are the hardest negatives: give them their own share
        clean = (y == 0) & (df["label_str"] == "clean").to_numpy()
        talking = (df["speak_frac"] >= 0.5).to_numpy()
        groups["clean_neg"] = (clean & ~talking, 1.0)
        groups["clean_speech_neg"] = (clean & talking, clean_speech_weight)
    vids = df["video"].to_numpy()
    for mask, share in groups.values():
        if not mask.any():
            continue
        uv, counts = np.unique(vids[mask], return_counts=True)
        per_video = share / len(uv)
        cnt = dict(zip(uv, counts))
        w[mask] = [per_video / cnt[v] for v in vids[mask]]
    return w * len(df) / max(w.sum(), 1e-9)


def fit(X, y, w, seed):
    from xgboost import XGBClassifier
    m = XGBClassifier(**XGB_PARAMS, random_state=seed, n_jobs=max(1, (os.cpu_count() or 4) - 1))
    m.fit(X, y, sample_weight=w)
    return m


def trainable_mask(windows: pd.DataFrame) -> np.ndarray:
    """Rows with a usable label: reviewed windows (0/1) + every clean window.
    Unreviewed cheating videos are never trained on."""
    unreviewed_cheat = (windows.label_str == "cheating") & ~windows.reviewed
    return (windows["wlabel"].isin([0, 1]) & ~unreviewed_cheat).to_numpy()


def training_frame(windows: pd.DataFrame, base_mask: np.ndarray, pseudo_rows=()):
    """Rows to train on: `base_mask` rows with their labels, plus pseudo-labelled
    reading windows (row positions) from unreviewed cheating videos."""
    idx = np.union1d(np.where(base_mask)[0], np.asarray(pseudo_rows, dtype=int))
    sub = windows.iloc[idx].copy()
    is_pseudo = np.isin(idx, np.asarray(pseudo_rows, dtype=int))
    sub["pseudo"] = is_pseudo
    sub.loc[sub.index[is_pseudo], "wlabel"] = 1
    return idx, sub


# --------------------------------------------------------------------------- #
# Pseudo-labels (self-training on the unreviewed cheating videos)
# --------------------------------------------------------------------------- #
def pseudo_labels(windows: pd.DataFrame, prob: np.ndarray, frac: float) -> dict:
    """{video: row positions} — for every unreviewed cheating video, the top `frac`
    of its SPEAKING windows by `prob` (from a model that never trained on it).
    In the reviewed videos 30-75% of speaking time is reading, so the top slice of
    a cheater's speech is mostly reading; it adds readers the reviews don't cover."""
    pos = np.arange(len(windows))
    unrev = (windows.label_str == "cheating").to_numpy() & ~windows.reviewed.to_numpy()
    speaking = (windows["speak_frac"] >= 0.5).to_numpy()
    vid = windows["video"].to_numpy()
    out = {}
    for v in np.unique(vid[unrev]):
        m = unrev & (vid == v) & speaking & np.isfinite(prob)
        if m.sum() < 5:
            continue
        k = max(1, int(round(frac * m.sum())))
        out[v] = pos[m][np.argsort(-prob[m])[:k]]
    return out


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def cross_validate(windows, videos, feats, cfg, n_folds, reviewed_neg_weight, seed,
                   pseudo_frac: float = 0.0, pseudo_weight: float = 0.5, fold_unrev=None,
                   clean_speech_weight: float = 0.0):
    """Out-of-fold window probabilities -> (scored windows, per-fold unreviewed probs).

    Self-training (pseudo_frac > 0, needs `fold_unrev` from a plain run): fold k
    picks its pseudo-labels with fold k's OWN plain model, which never saw fold k's
    held-out reviewed video, so that video's labels can't leak in through the
    choice of pseudo-labels. The unreviewed cheating videos are also split into
    halves A/B: one model trains on A's pseudo-labels and scores B, the other the
    reverse; reviewed and clean videos get the mean of the two. No video is ever
    scored by a model that saw its own pseudo-labels."""
    folds = assign_folds(videos, n_folds, seed)
    windows = windows.assign(fold=windows["video"].map(folds))
    X = windows[feats].to_numpy(dtype=np.float32)
    oof = np.full(len(windows), np.nan)
    test_only = (windows["fold"] == -1).to_numpy()
    test_acc = np.zeros(len(windows))
    trainable = trainable_mask(windows)

    use_pseudo = pseudo_frac > 0 and fold_unrev is not None
    if use_pseudo:
        unrev = sorted(windows.loc[test_only, "video"].unique())
        rng = np.random.default_rng(seed)
        rng.shuffle(unrev)
        halves = [set(unrev[0::2]), set(unrev[1::2])]
    fold_probs = {}
    n_pseudo = 0

    for k in range(n_folds):
        base = trainable & (windows["fold"] != k).to_numpy()
        te = (windows["fold"] == k).to_numpy()
        if use_pseudo:
            pseudo = pseudo_labels(windows, fold_unrev[k], pseudo_frac)
            plans = []                   # (pseudo rows to train on, rows this model scores)
            for own, other in ((halves[0], halves[1]), (halves[1], halves[0])):
                rows = np.concatenate([pseudo[v] for v in own if v in pseudo] or [np.array([], int)])
                plans.append((rows, test_only & windows["video"].isin(other).to_numpy()))
            n_pseudo = sum(len(r) for r in pseudo.values())
        else:
            plans = [(np.array([], int), test_only)]
        fold_probs[k] = np.full(len(windows), np.nan)
        preds = []
        for rows, scores_mask in plans:
            idx, sub = training_frame(windows, base, rows)
            model = fit(X[idx], sub["wlabel"].to_numpy(),
                        sample_weights(sub, reviewed_neg_weight, pseudo_weight, clean_speech_weight), seed)
            preds.append(model.predict_proba(X[te])[:, 1])
            p_un = model.predict_proba(X[scores_mask])[:, 1]
            test_acc[scores_mask] += p_un
            fold_probs[k][scores_mask] = p_un
        oof[te] = np.mean(preds, axis=0)
        print(f"  fold {k}: train {base.sum()} windows"
              + (f" (+{n_pseudo} pseudo-reading)" if use_pseudo else "") + f", test {te.sum()}", flush=True)
    oof[test_only] = test_acc[test_only] / n_folds
    return windows.assign(prob=oof), fold_probs


def per_second_tables(scored: pd.DataFrame, videos: pd.DataFrame) -> dict:
    """{video: dict(score=per-second array, speaking_s=.., ...)}"""
    out = {}
    vinfo = videos.set_index("video")
    for v, sub in scored.groupby("video", sort=False):
        n_sec = int(np.ceil(sub["win_end_s"].max()))
        sc = reading.per_second(sub["win_start_s"], sub["win_end_s"], sub["prob"], n_sec)
        spk = reading.per_second(sub["win_start_s"], sub["win_end_s"], sub["speak_frac"], n_sec,
                                 smooth_s=1, center_radius_s=0.5)
        out[v] = dict(score=sc, speaking=spk, speaking_s=float(vinfo.loc[v, "speak_share"] * vinfo.loc[v, "duration"]),
                      label_str=vinfo.loc[v, "label_str"], reviewed=bool(vinfo.loc[v, "reviewed"]),
                      source=vinfo.loc[v, "source"])
    return out


def choose_second_threshold(tables, reviews, cfg, clean_budget):
    """Max pooled F1 on reviewed videos s.t. <= clean_budget of clean seconds flagged."""
    # log-spaced at the low end: a reader the model never saw often scores well
    # below 0.1 while clean speech sits far lower still
    grid = np.unique(np.round(np.concatenate([np.geomspace(0.005, 0.1, 16),
                                              np.arange(0.1, 0.96, 0.025)]), 4))
    rows = []
    for thr in grid:
        tp = fp = fn = 0
        clean_flag = clean_total = 0
        for v, d in tables.items():
            segs = reading.segments(d["score"], thr, cfg.min_segment_seconds, cfg.merge_gap_seconds)
            flag = reading.flagged_mask(segs, len(d["score"]))
            if d["reviewed"]:
                rv = lab.review_for(v, reviews)
                truth = lab.truth_mask(rv.segments, len(d["score"]), 1.0)
                m = np.isfinite(d["score"])
                tp += int((flag & truth & m).sum()); fp += int((flag & ~truth & m).sum())
                fn += int((~flag & truth & m).sum())
            elif d["label_str"] == "clean":
                clean_flag += int(flag.sum()); clean_total += int(np.isfinite(d["score"]).sum())
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        rows.append(dict(threshold=thr, precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9),
                         clean_flag_rate=clean_flag / max(clean_total, 1)))
    tbl = pd.DataFrame(rows)
    ok = tbl[tbl.clean_flag_rate <= clean_budget]
    best = (ok if len(ok) else tbl).sort_values("f1", ascending=False).iloc[0]
    return float(best.threshold), tbl


def tune_video_threshold(scores, labels, beta=0.5):
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(labels, scores)
    prec, rec = prec[:-1], rec[:-1]
    b2 = beta * beta
    f = np.where((b2 * prec + rec) > 0, (1 + b2) * prec * rec / (b2 * prec + rec), 0)
    return float(thr[int(np.argmax(f))])


def evaluate_cv(scored, videos, cfg, reviews, clean_budget, video_stat):
    tables = per_second_tables(scored, videos)
    sec_thr, sweep = choose_second_threshold(tables, reviews, cfg, clean_budget)

    rows = []
    for v, d in tables.items():
        if not d["reviewed"]:
            continue
        rv = lab.review_for(v, reviews)
        n = len(d["score"])
        truth = lab.truth_mask(rv.segments, n, 1.0)
        segs = reading.segments(d["score"], sec_thr, cfg.min_segment_seconds, cfg.merge_gap_seconds)
        flag = reading.flagged_mask(segs, n)
        rows.append(dict(video=v.split("__")[-1], **per_second_metrics(d["score"], truth, flag, rv.segments)))

    clean_rows = []
    for v, d in tables.items():
        if d["label_str"] != "clean":
            continue
        segs = reading.segments(d["score"], sec_thr, cfg.min_segment_seconds, cfg.merge_gap_seconds)
        clean_rows.append(dict(video=v, flagged_s=sum(e - s for s, e, *_ in segs),
                               seconds=int(np.isfinite(d["score"]).sum())))
    cf = pd.DataFrame(clean_rows)

    vrows = []
    for v, d in tables.items():
        st = reading.video_stats(d["score"], d["speaking_s"], sec_thr,
                                 cfg.min_segment_seconds, cfg.merge_gap_seconds, d["speaking"])
        vrows.append(dict(video=v, label=int(d["label_str"] == "cheating"), reviewed=d["reviewed"],
                          matched=(d["label_str"] == "clean") and d["source"] == "interviews", **st))
    vs = pd.DataFrame(vrows)
    from sklearn.metrics import roc_auc_score
    stat_auc = {s: float(roc_auc_score(vs.label, vs[s])) for s in VIDEO_STATS}
    vs["score"] = vs[video_stat]
    vthr = tune_video_threshold(vs["score"].to_numpy(), vs["label"].to_numpy())
    vm = video_metrics(vs, vthr)
    unrev = vs[(vs.label == 1) & ~vs.reviewed]
    vm["unreviewed_caught"] = f"{int((unrev.score >= vthr).sum())}/{len(unrev)}"
    return dict(second_threshold=sec_thr, sweep=sweep, per_second=rows, clean=cf,
                video_scores=vs, video=vm, stat_auc=stat_auc, video_threshold=vthr, tables=tables)


# --------------------------------------------------------------------------- #
# Ship
# --------------------------------------------------------------------------- #
# Interpretable features for segment reasons: (feature, label if higher, label if lower)
REASON_FEATURES = [
    ("blink_rate", "more blinking", "reduced blinking"),
    ("brow_down_mean", "brow lowered (concentrating)", "brow raised"),
    ("face_scale_rel", "leaning toward the screen", "leaning back"),
    ("v_mean_listen", "eyes raised vs. when listening", "eyes lowered vs. when listening"),
    ("parked_frac", "gaze held on one screen region", "gaze wandering"),
    ("aversion_frac", "frequent glances away", "no thinking glances away"),
    ("h_sweeps_dom", "frequent side-to-side eye moves", "fewer side-to-side glances"),
    ("h_prog_frac", "line-by-line eye drift", "little eye drift"),
    ("head_speed", "head moving", "head still"),
    ("squint_mean", "squinting at text", "eyes wide"),
    ("h_std", "horizontal eye scanning", "eyes steady"),
    ("big_jump_rate", "frequent big eye jumps", "few big eye jumps"),
]


def reason_reference(windows: pd.DataFrame, min_effect: float = 0.25) -> list:
    """How reading windows differ from clean candidates' SPEECH windows, per
    interpretable feature. predict.py tags a segment with a reason when its mean
    lies beyond the clean-speech quartile in the reading direction."""
    pos = windows[windows.wlabel == 1]
    ref = windows[(windows.label_str == "clean") & (windows.speak_frac >= 0.5)]
    out = []
    for feat, hi_label, lo_label in REASON_FEATURES:
        if feat not in windows or pos.empty or ref.empty:
            continue
        p, c = pos[feat].dropna(), ref[feat].dropna()
        q25, q50, q75 = np.percentile(c, [25, 50, 75])
        scale = max(q75 - q25, 1e-6)
        diff = float(np.median(p) - q50)
        if abs(diff) / scale < min_effect:
            continue
        d = 1 if diff > 0 else -1
        out.append(dict(feature=feat, direction=d, label=hi_label if d > 0 else lo_label,
                        clean_median=float(q50), scale=float(scale),
                        ref=float(q75 if d > 0 else q25), effect=round(abs(diff) / scale, 3)))
    return sorted(out, key=lambda r: -r["effect"])


def ship(model, windows, feats, cfg, ev, args, importances):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODELS_DIR / "reading_xgb.json"))
    art = dict(
        kind="reading_xgb", version=2, features=feats,
        read_window_seconds=cfg.read_window_seconds, read_stride_seconds=cfg.read_stride_seconds,
        second_threshold=ev["second_threshold"], smooth_s=3,
        min_segment_seconds=cfg.min_segment_seconds, merge_gap_seconds=cfg.merge_gap_seconds,
        video_stat=args.video_stat, video_threshold=ev["video_threshold"],
        threshold=ev["video_threshold"], experiment=args.exp,
        cv=dict(per_second=ev["per_second"], video=ev["video"], stat_auc=ev["stat_auc"]),
        trained_on=dict(reviewed=sorted(windows.loc[windows.reviewed, "video"].unique().tolist()),
                        n_clean=int(windows.loc[windows.label_str == "clean", "video"].nunique())),
        top_features=importances[:15],
        reason_reference=reason_reference(windows),
    )
    (MODELS_DIR / "model.json").write_text(json.dumps(art, indent=2, default=float))
    print(f"shipped -> {MODELS_DIR / 'reading_xgb.json'} + model.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 3 (v2): reading detector.")
    ap.add_argument("--exp", default="xgb-strong-labels", help="name for the results log")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reviewed-neg-weight", type=float, default=0.5)
    ap.add_argument("--clean-budget", type=float, default=0.02,
                    help="max share of clean-video seconds the second threshold may flag")
    ap.add_argument("--video-stat", default="reading_seconds", choices=VIDEO_STATS)
    ap.add_argument("--drop-group", action="append", default=[], choices=list(GROUPS))
    ap.add_argument("--pseudo-frac", type=float, default=0.0,
                    help="self-training: top fraction of each unreviewed cheating video's "
                         "speaking windows used as extra reading labels (0 = off)")
    ap.add_argument("--pseudo-weight", type=float, default=0.5,
                    help="weight share of the pseudo-labelled reading windows")
    ap.add_argument("--clean-speech-weight", type=float, default=0.0,
                    help="extra weight share for clean candidates' speaking windows (0 = off)")
    ap.add_argument("--no-ship", action="store_true")
    ap.add_argument("--no-log", action="store_true", help="smoke tests: don't append to RESULTS_LOG.md")
    ap.add_argument("--note", default="", help="free text added to the log entry")
    args = ap.parse_args(argv)

    cfg = DEFAULT_CONFIG
    if (MODELS_DIR / "config.json").exists():
        cfg = Config.from_json(MODELS_DIR / "config.json")
    windows, videos, all_feats = load_data()
    feats = feature_list(all_feats, args.drop_group)
    reviews = lab.load_reviews()
    n_rev = int(videos.reviewed.sum())
    print(f"windows={len(windows)} videos={len(videos)} reviewed={n_rev} features={len(feats)}")
    if n_rev == 0:
        print("No reviewed videos found — add manual reviews first.")
        return 1

    scored, fold_unrev = cross_validate(windows, videos, feats, cfg, args.folds,
                                        args.reviewed_neg_weight, args.seed,
                                        clean_speech_weight=args.clean_speech_weight)
    pseudo = {}
    if args.pseudo_frac > 0:
        # shipped model: pseudo-labels picked by the mean of the plain fold models
        pseudo = pseudo_labels(windows, scored["prob"].to_numpy(), args.pseudo_frac)
        print(f"self-training: ~{sum(len(r) for r in pseudo.values())} pseudo-reading windows "
              f"from {len(pseudo)} unreviewed cheating videos; re-running CV", flush=True)
        scored, _ = cross_validate(windows, videos, feats, cfg, args.folds, args.reviewed_neg_weight,
                                   args.seed, args.pseudo_frac, args.pseudo_weight, fold_unrev,
                                   args.clean_speech_weight)
    ev = evaluate_cv(scored, videos, cfg, reviews, args.clean_budget, args.video_stat)

    # final model on every trainable row (+ all pseudo-labels): importances + shipped model
    all_pseudo = np.concatenate(list(pseudo.values())) if pseudo else np.array([], int)
    idx, sub = training_frame(windows, trainable_mask(windows), all_pseudo)
    full = fit(windows[feats].to_numpy(dtype=np.float32)[idx], sub["wlabel"].to_numpy(),
               sample_weights(sub, args.reviewed_neg_weight, args.pseudo_weight,
                              args.clean_speech_weight), args.seed)
    imp = sorted(zip(feats, full.feature_importances_.astype(float)), key=lambda x: -x[1])

    # ---- console ----
    cols = ["video", "reading_s", "flagged_s", "auc", "precision", "recall", "iou", "spans_caught"]
    ps = pd.DataFrame(ev["per_second"])
    print(f"\nsecond threshold = {ev['second_threshold']}")
    print(ps[cols].to_string(index=False))
    cf = ev["clean"]
    print(f"clean videos: {cf.flagged_s.sum():.0f}s flagged of {cf.seconds.sum()}s "
          f"({cf.flagged_s.sum() / max(cf.seconds.sum(), 1):.2%}); "
          f"videos with any flag: {(cf.flagged_s > 0).sum()}/{len(cf)}")
    print("video-stat AUCs:", {k: round(v, 3) for k, v in ev["stat_auc"].items()})
    print("video:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ev["video"].items()})
    print("top features:", [(f, round(i, 3)) for f, i in imp[:12]])

    # ---- results log ----
    vm = ev["video"]
    vrow = dict(stat=args.video_stat, auc=vm["auc"], precision=vm["precision"], recall=vm["recall"],
                tp=vm["tp"], fp=vm["fp"], fn=vm["fn"], tn=vm["tn"],
                matched_auc=vm["matched_auc"], matched_fp=f"{vm['matched_fp']}/{vm['matched_n_clean']}",
                unreviewed_caught=vm["unreviewed_caught"], threshold=vm["threshold"])
    pooled = dict(video="ALL reviewed (pooled)",
                  reading_s=int(ps.reading_s.sum()), flagged_s=int(ps.flagged_s.sum()),
                  auc=float(ps.auc.mean()),
                  precision=float((ps.precision * ps.flagged_s).sum() / max(ps.flagged_s.sum(), 1)),
                  recall=float((ps.recall * ps.reading_s).sum() / max(ps.reading_s.sum(), 1)),
                  iou=float(ps.iou.mean()),
                  spans_caught=f"{sum(int(s.split('/')[0]) for s in ps.spans_caught)}/"
                               f"{sum(int(s.split('/')[1]) for s in ps.spans_caught)}")
    summary = (f"XGBoost reading detector, {len(feats)} features, {args.folds}-fold CV "
               f"(reviewed videos held out one per fold; unreviewed cheating never trained on). "
               f"Second threshold {ev['second_threshold']} (max F1, clean budget {args.clean_budget:.0%}); "
               f"video rule: {args.video_stat} >= {ev['video_threshold']:.2f}. "
               f"Dropped groups: {args.drop_group or 'none'}. "
               + (f"Self-training: top {args.pseudo_frac:.0%} of each unreviewed cheating video's "
                  f"speaking windows as pseudo-reading (weight {args.pseudo_weight}), A/B cross-scored. "
                  if args.pseudo_frac > 0 else "")
               + (f"Clean speaking windows weighted as their own share ({args.clean_speech_weight}). "
                  if args.clean_speech_weight > 0 else "") + args.note)
    clean_line = (f"Clean videos: {cf.flagged_s.sum():.0f} s flagged of {cf.seconds.sum()} s "
                  f"({cf.flagged_s.sum() / max(cf.seconds.sum(), 1):.2%}); "
                  f"{(cf.flagged_s > 0).sum()}/{len(cf)} clean videos have any flagged span.")
    if not args.no_log:
        results_log.append(
            args.exp, summary,
            {"Per-second vs manual reviews (out-of-fold; pooled AUC = mean over videos)":
                 results_log.table(ev["per_second"] + [pooled], cols) + "\n" + clean_line,
             "Video level (out-of-fold)": results_log.table([vrow]),
             "Video-statistic AUCs": results_log.table([ev["stat_auc"]]),
             "Top features": ", ".join(f"{f} ({i:.3f})" for f, i in imp[:12])},
            dict(args=vars(args), per_second=ev["per_second"], video=vm, stat_auc=ev["stat_auc"],
                 second_threshold=ev["second_threshold"], sweep=ev["sweep"].to_dict(orient="records"),
                 importances=imp[:40]))

    if not args.no_log:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(RESULTS_DIR / f"oof_seconds__{args.exp}.npz",
                            **{v.split("__")[-1]: d["score"] for v, d in ev["tables"].items()})
        ev["video_scores"].to_csv(RESULTS_DIR / f"oof_video_scores__{args.exp}.csv", index=False)

    if not args.no_ship:
        ship(full, windows, feats, cfg, ev, args, imp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
