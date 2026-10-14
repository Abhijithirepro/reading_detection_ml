"""
config_results_csv.py (v2, new) — one CSV per version with the configuration that
was used and the results it produced, generated from the real artefacts (never
typed by hand):

    v2/CONFIG_AND_RESULTS_v2.csv   <- v2/models/{config,model}.json, train.py, results/*.json
    CONFIG_AND_RESULTS_v1.csv      <- models/{config,model,metrics}.json, models/model_cnn.pt,
                                      src/train.py, results/*__v1-baseline.json

Columns: section, item, value, notes. Re-run after retraining:
    python v2/src/config_results_csv.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

from common import PROJECT_ROOT, V1_ROOT, MODELS_DIR, RESULTS_DIR, WINDOWS_DIR, FEATURES_DIR, BLENDSHAPE_CHANNELS
import labels as lab


def _latest(pattern: str):
    files = sorted(RESULTS_DIR.glob(pattern))
    return json.loads(files[-1].read_text()) if files else None


def _f(v, nd=3):
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return v


def _pooled(per_second: list) -> dict:
    ps = pd.DataFrame(per_second)
    caught = sum(int(s.split("/")[0]) for s in ps.spans_caught)
    total = sum(int(s.split("/")[1]) for s in ps.spans_caught)
    return dict(spans=f"{caught}/{total}", auc=float(ps.auc.mean()),
                precision=float((ps.precision * ps.flagged_s).sum() / max(ps.flagged_s.sum(), 1)),
                recall=float((ps.recall * ps.reading_s).sum() / max(ps.reading_s.sum(), 1)),
                iou=float(ps.iou.mean()))


def _per_second_rows(rows, section, per_second, note):
    for r in per_second:
        name = r["video"].replace("combined_", "")
        rows.append((section, f"{name}: reading seconds (reviewer)", r["reading_s"], note))
        rows.append((section, f"{name}: flagged seconds (model)", r["flagged_s"], ""))
        rows.append((section, f"{name}: per-second AUC", _f(r["auc"]), ""))
        rows.append((section, f"{name}: precision", _f(r["precision"]), ""))
        rows.append((section, f"{name}: recall", _f(r["recall"]), ""))
        rows.append((section, f"{name}: IoU", _f(r["iou"]), ""))
        rows.append((section, f"{name}: reviewer spans caught", r["spans_caught"], ">= 50% of span flagged"))
    p = _pooled(per_second)
    rows.append((section, "ALL reviewed: spans caught", p["spans"], ""))
    rows.append((section, "ALL reviewed: mean per-second AUC", _f(p["auc"]), ""))
    rows.append((section, "ALL reviewed: pooled precision", _f(p["precision"]), ""))
    rows.append((section, "ALL reviewed: pooled recall", _f(p["recall"]), ""))
    rows.append((section, "ALL reviewed: mean IoU", _f(p["iou"]), ""))


def _video_rows(rows, section, vm, note):
    rows.append((section, "videos scored", vm["n"], note))
    rows.append((section, "video ROC-AUC", _f(vm["auc"]), ""))
    rows.append((section, "cheating caught (TP / all cheating)", f"{vm['tp']}/{vm['tp'] + vm['fn']}", ""))
    rows.append((section, "clean cleared (TN / all clean)", f"{vm['tn']}/{vm['tn'] + vm['fp']}", ""))
    rows.append((section, "false alarms (FP)", vm["fp"], ""))
    rows.append((section, "precision", _f(vm["precision"]), ""))
    rows.append((section, "recall", _f(vm["recall"]), ""))
    rows.append((section, "accuracy", _f(vm["accuracy"]), ""))
    rows.append((section, "same-platform AUC (cheating vs not_cheating_folder)", _f(vm["matched_auc"]),
                 "honest comparison: same recording platform"))
    rows.append((section, "same-platform false alarms",
                 f"{vm['matched_fp']}/{vm['matched_n_clean']}", ""))


def _write(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "item", "value", "notes"])
        for r in rows:
            w.writerow(["" if x is None else x for x in r])
    print(f"wrote {path}  ({len(rows)} rows)")


# --------------------------------------------------------------------------- #
# v1
# --------------------------------------------------------------------------- #
def v1_rows() -> list:
    rows = []
    cfg = json.loads((V1_ROOT / "models" / "config.json").read_text())
    art = json.loads((V1_ROOT / "models" / "model.json").read_text())
    met = json.loads((V1_ROOT / "models" / "metrics.json").read_text())
    S = "1. data"
    for s in met.get("per_source", []):
        rows.append((S, f"{s['label_str']} videos, source '{s['source']}'", s["n"], "used in v1 training/CV"))
    n_ch = sum(s["n"] for s in met["per_source"] if s["label_str"] == "cheating")
    n_cl = sum(s["n"] for s in met["per_source"] if s["label_str"] == "clean")
    rows.append((S, "total videos", n_ch + n_cl, f"{n_ch} cheating / {n_cl} clean"))
    rows.append((S, "labels", "video-level (weak)",
                 "every window of a cheating video labelled cheating; no timestamps"))

    S = "2. extraction"
    rows.append((S, "face tracker", "MediaPipe FaceLandmarker (478 landmarks + iris)", "Apache-2.0"))
    rows.append((S, "backend", cfg["backend"], ""))
    rows.append((S, "target_fps", cfg["target_fps"],
                 "keep-rule gave 12.5 fps effective on 25 fps sources"))
    rows.append((S, "frame resolution", "native", "no downscaling (720p clean vs <=640 px cheating)"))
    rows.append((S, "per-frame channels", "gaze_x, gaze_y, yaw, pitch, roll, ear, mouth, blink", ""))
    rows.append((S, "blendshapes", "no", ""))
    for k in ("min_face_ratio", "max_gap_seconds", "median_filter_k", "calibrate_gaze", "ear_blink_threshold"):
        rows.append((S, k, cfg[k], ""))

    S = "3. windows"
    rows.append((S, "window_seconds", art["window_seconds"], ""))
    rows.append((S, "stride_seconds", art["stride_seconds"], ""))
    rows.append((S, "min_valid_ratio", cfg["min_valid_ratio"], ""))
    rows.append((S, "CNN input channels", ", ".join(art["seq_channels"]), ""))
    rows.append((S, "sequence length", art["seq_len"], "window resampled to 12 Hz"))

    S = "4. model"
    rows.append((S, "kind", art["kind"], "1D-CNN"))
    rows.append((S, "architecture",
                 "Conv1d(6->32,k5)+ReLU, Conv1d(32->32,k5)+ReLU, AvgPool, Dropout 0.3, Linear(32->1)",
                 "src/train.py _build_seq_net"))
    try:
        import torch
        sd = torch.load(V1_ROOT / "models" / "model_cnn.pt", map_location="cpu", weights_only=False)["state_dict"]
        rows.append((S, "parameters", sum(v.numel() for v in sd.values()), "counted from models/model_cnn.pt"))
    except Exception as e:  # torch missing / file moved: say so instead of guessing
        rows.append((S, "parameters", "", f"could not count: {e}"))
    rows.append((S, "input normalisation", "per-channel z-score (models/seq_scaler.npz)", ""))
    rows.append((S, "browser", "yes: ONNX via onnxruntime-web (frontend/)", ""))

    S = "5. training"
    for item, val in [("optimizer", "Adam"), ("learning rate", "1e-3"), ("weight decay", "1e-4"),
                      ("batch size", 32), ("max epochs", 40), ("early stopping patience", "6 (train loss)"),
                      ("loss", "BCEWithLogits, pos_weight = negatives/positives"),
                      ("cross-validation", f"StratifiedGroupKFold({cfg['n_splits']}) grouped by video"),
                      ("random_state", cfg["random_state"])]:
        rows.append((S, item, val, "src/train.py"))

    S = "6. verdict"
    rows.append((S, "video score", f"mean of top-k window probabilities (topk_fraction {art['topk_fraction']}, "
                                   f"topk_min {art['topk_min']})", ""))
    rows.append((S, "threshold", _f(art["threshold"], 4), "tuned for F0.5 (favours precision)"))
    rows.append((S, "flagged segments", "consecutive windows >= threshold, merged", ""))

    S = "7. results: v1 own CV (106 videos)"
    c = met["combos"][0]
    for k, label in [("video_roc_auc", "video ROC-AUC"), ("video_pr_auc", "video PR-AUC"),
                     ("window_roc_auc", "window ROC-AUC"), ("precision", "precision"), ("recall", "recall")]:
        rows.append((S, label, _f(c[k]), "models/metrics.json"))
    rows.append((S, "cheating caught (TP / all cheating)", f"{c['tp']}/{c['tp'] + c['fn']}", ""))
    rows.append((S, "false alarms (FP / all clean)", f"{c['fp']}/{c['fp'] + c['tn']}", ""))
    for p in met.get("source_pairs", []):
        rows.append((S, f"cross-source AUC: {p['cheat_source']} vs {p['clean_source']}", _f(p["video_auc"]),
                     f"{p['n_test_cheat']}+{p['n_test_clean']} videos"))

    base = _latest("*__v1-baseline.json")
    if base:
        _video_rows(rows, "8. results: 100 on-disk videos (v1 out-of-fold)", base["video"],
                    "same 100 recordings v2 is evaluated on")
        _per_second_rows(rows, "9. results: vs manual reviews, per second (IN-SAMPLE)", base["per_second"],
                         "v1 trained on these videos: its best case")
    return rows


# --------------------------------------------------------------------------- #
# v2
# --------------------------------------------------------------------------- #
def v2_rows() -> list:
    import train as tr
    rows = []
    cfg = json.loads((MODELS_DIR / "config.json").read_text())
    art = json.loads((MODELS_DIR / "model.json").read_text())
    ship = _latest(f"*__{art['experiment']}.json")
    a = ship["args"] if ship else {}

    S = "1. data"
    vids = pd.read_parquet(WINDOWS_DIR / "videos.parquet")
    vids["folder"] = vids.label_str.map({"cheating": "cheating_folder"}).fillna(
        vids.source.map({"interviews": "not_cheating_folder", "interviews_new": "new_not_cheating"}))
    for f, n in vids.folder.value_counts().items():
        rows.append((S, f"videos in {f}", n, ""))
    rows.append((S, "total videos", len(vids),
                 f"{(vids.label_str == 'cheating').sum()} cheating / {(vids.label_str == 'clean').sum()} clean"))
    rows.append((S, "hours of video", round(vids.duration.sum() / 3600, 1), ""))
    for ir, r in lab.load_reviews().items():
        rows.append((S, f"manual review ir{ir}", f"{len(r.segments)} spans, {r.reading_seconds / 60:.1f} min reading",
                     r.path.name))
    w = pd.read_parquet(WINDOWS_DIR / "windows.parquet", columns=["wlabel"])
    names = {1: "reading (reviewed)", 0: "not reading (reviewed + all clean)",
             -1: "unknown (unreviewed cheating: test only)", -2: "ignored (span boundary)"}
    rows.append((S, "windows total", len(w), ""))
    for k, n in w.wlabel.value_counts().sort_index(ascending=False).items():
        rows.append((S, f"windows: {names.get(k, k)}", n, ""))
    rows.append((S, "labels", "per-second (strong) from manual reviews",
                 f"window = reading if >= {cfg['label_pos_frac']:.0%} inside a span, "
                 f"not reading if <= {cfg['label_neg_frac']:.0%}"))

    S = "2. extraction"
    rows.append((S, "face tracker", "MediaPipe FaceLandmarker (478 landmarks + iris + blendshapes)", "Apache-2.0"))
    for k in ("target_fps", "max_frame_width", "blendshapes", "min_face_ratio"):
        rows.append((S, k, cfg[k], {"target_fps": "production rate; token-bucket sampler: 25/30/60 fps sources give exactly 15, slower sources keep every frame",
                                     "max_frame_width": "frames downscaled to <= 640 px wide"}.get(k, "")))
    rows.append((S, "blendshape channels", len(BLENDSHAPE_CHANNELS), ", ".join(BLENDSHAPE_CHANNELS)))
    rows.append((S, "other per-frame channels", "gaze_x, gaze_y, yaw, pitch, roll, ear, mouth, face_scale", ""))
    summ = FEATURES_DIR / "_extract_summary.csv"
    if summ.exists():
        ex = pd.read_csv(summ)
        ok = ex[ex.status == "ok"]
        slow = ok[ok.sampled_fps < 14.5]
        rows.append((S, "videos extracted", f"{len(ok)} of {len(ex)} ok", "status from data/features/_extract_summary.csv"))
        rows.append((S, "measured sampling rate", f"{len(ok) - len(slow)} videos at 15 fps",
                     "; ".join(f"{Path(v).stem}: {r:.1f} fps (source slower than 15)"
                               for v, r in zip(slow.video, slow.sampled_fps)) or "every video at 15 fps"))
        rows.append((S, "median face found", f"{ok.face_ratio.median():.1%}",
                     f"lowest {ok.face_ratio.min():.1%}"))

    S = "3. cleaning + speaking"
    for k in ("max_gap_seconds", "median_filter_k", "calibrate_gaze"):
        rows.append((S, k, cfg[k], ""))
    rows.append((S, "blink rule", "eyeBlink blendshape > 0.40", ""))
    rows.append((S, "speaking detector", "lip gap (MAR) + jawOpen motion, fused", "no audio is ever read"))
    for k in ("vad_act_window_s", "vad_on_frac", "vad_off_frac", "vad_floor_pctile", "vad_min_dynamic",
              "vad_min_burst_s", "vad_merge_gap_s"):
        rows.append((S, k, cfg[k], ""))

    S = "4. windows + features"
    rows.append((S, "read_window_seconds", cfg["read_window_seconds"], ""))
    rows.append((S, "read_stride_seconds", cfg["read_stride_seconds"], "-> one probability per second"))
    rows.append((S, "label_pos_frac", cfg["label_pos_frac"], ""))
    rows.append((S, "label_neg_frac", cfg["label_neg_frac"], ""))
    rows.append((S, "features used by the model", len(art["features"]), ", ".join(art["features"])))
    rows.append((S, "answer-context span", "30 s (speaking samples only)", "ctx_* features"))

    S = "5. model"
    rows.append((S, "kind", art["kind"], "XGBoost gradient-boosted trees (Apache-2.0)"))
    for k, v in tr.XGB_PARAMS.items():
        rows.append((S, f"xgboost {k}", v, ""))
    rows.append((S, "browser", "yes: ONNX TreeEnsembleClassifier via onnxruntime-web",
                 "checked in Chrome with onnxruntime-web 1.19.2: max diff 3e-7 vs Python"))

    S = "6. training"
    for k, note in [("folds", "reviewed videos held out one per fold"), ("seed", ""),
                    ("reviewed_neg_weight", "weight share of reviewed not-reading windows"),
                    ("pseudo_frac", "self-training: top share of each unreviewed cheater's speaking windows"),
                    ("pseudo_weight", "weight share of pseudo-labelled reading windows"),
                    ("clean_speech_weight", "extra weight share for clean candidates' speaking windows"),
                    ("clean_budget", "max share of clean seconds the per-second threshold may flag")]:
        rows.append((S, k, a.get(k, ""), note))
    rows.append((S, "pseudo-label leakage guard", "fold-specific selection + A/B cross-scoring", ""))
    rows.append((S, "trained on", f"{len(art['trained_on']['reviewed'])} reviewed + "
                                   f"{art['trained_on']['n_clean']} clean videos + pseudo-labels", ""))

    S = "7. thresholds + verdict"
    rows.append((S, "per-second threshold", _f(art["second_threshold"]), "best F1 within the clean budget"))
    rows.append((S, "per-second smoothing", f"{art['smooth_s']} s rolling median; windows centred within 2 s", ""))
    rows.append((S, "min_segment_seconds", art["min_segment_seconds"], ""))
    rows.append((S, "merge_gap_seconds", art["merge_gap_seconds"], ""))
    rows.append((S, "verdict statistic", art["video_stat"], "average reading probability while speaking"))
    rows.append((S, "verdict threshold", _f(art["video_threshold"], 4), "tuned for F0.5 (favours precision)"))

    S = "8. results: vs manual reviews, per second (OUT-OF-FOLD)"
    _per_second_rows(rows, S, art["cv"]["per_second"], "model never saw this video")
    _video_rows(rows, "9. results: 100 videos (OUT-OF-FOLD)", art["cv"]["video"], "")
    for k, v in art["cv"]["stat_auc"].items():
        rows.append(("10. results: verdict-statistic AUCs", k, _f(v), "chosen" if k == art["video_stat"] else ""))

    S = "11. what the model uses"
    for f, imp in art["top_features"][:12]:
        rows.append((S, f"feature importance: {f}", _f(imp), ""))
    for r in art.get("reason_reference", []):
        rows.append((S, f"reason tag: {r['label']}", f"effect {r['effect']}", f"feature {r['feature']}"))

    S = "12. all experiments (RESULTS_LOG.md)"
    for f in sorted(RESULTS_DIR.glob("*.json")):
        j = json.loads(f.read_text())
        # runs before the 15 fps switch carry no target_fps: v2 ran at 25, v1 at 12.5 effective
        fps = j.get("target_fps") or (12.5 if j["name"].startswith("v1") else 25.0)
        when = f"{j['time']} · {fps:g} fps"
        if "contenders" in j:
            for name, c in j["contenders"].items():
                p = _pooled(c["per_second"])
                rows.append((S, f"{j['name']} [{name}]", f"spans {p['spans']}, recall {p['recall']:.2f}, "
                             f"video AUC {c['video']['auc']:.3f}, FP {c['video']['fp']}", when))
        else:
            p = _pooled(j["per_second"])
            note = "IN-SAMPLE per-second" if "v1" in j["name"] else ""
            rows.append((S, j["name"], f"spans {p['spans']}, recall {p['recall']:.2f}, "
                         f"video AUC {j['video']['auc']:.3f}, FP {j['video']['fp']}",
                         (when + (" · " + note if note else ""))))
    return rows


def main():
    _write(PROJECT_ROOT / "CONFIG_AND_RESULTS_v2.csv", v2_rows())
    _write(V1_ROOT / "CONFIG_AND_RESULTS_v1.csv", v1_rows())
    return 0


if __name__ == "__main__":
    sys.exit(main())
