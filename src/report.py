"""
report.py — Stage 4 reporting: per-video HTML review + overall evaluation report.

Per-video (needs a prediction from predict.py; auto-runs it if missing):
    gaze_x / gaze_y traces with suspicious segments shaded, head-pose strip,
    blink/EAR markers, a window-probability timeline, a verdict banner, and the
    flagged segments.

Evaluation report (from train.py artifacts):
    CNN cross-validation metrics, cross-source generalization, and the per-video
    out-of-fold score table (predicted vs actual).

Usage:
    python src/report.py path/to/video.mp4      # per-video report
    python src/report.py --eval                 # overall evaluation report
    python src/report.py --all                  # eval + every video with a prediction
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import Config, DEFAULT_CONFIG, MODELS_DIR, REPORTS_DIR

PLOT_BG = "#0f1419"
FG = "#e6e6e6"
ACCENT = "#4fc3f7"
FLAG = "#ff5252"


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _style(ax):
    ax.set_facecolor(PLOT_BG)
    for s in ax.spines.values():
        s.set_color("#33404d")
    ax.tick_params(colors=FG, labelsize=8)
    ax.yaxis.label.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.title.set_color(FG)


# --------------------------------------------------------------------------- #
# Per-video report
# --------------------------------------------------------------------------- #
def build_video_report(video: Path):
    stem = video.stem
    tl_path = REPORTS_DIR / f"{stem}__timeline.npz"
    pred_path = REPORTS_DIR / f"{stem}__prediction.json"

    if not tl_path.exists() or not pred_path.exists():
        print(f"No prediction for {stem}; running predict.py first...")
        from predict import main as predict_main
        rc = predict_main([str(video)])
        if rc != 0:
            return rc

    tl = np.load(tl_path, allow_pickle=True)
    result = json.loads(pred_path.read_text())
    dt = float(tl["dt"][0])
    t = np.arange(len(tl["gaze_x"])) * dt
    segments = result["flagged_segments"]

    # ---- figure: 4 stacked panels sharing the time axis ----
    fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True,
                             facecolor=PLOT_BG, gridspec_kw={"height_ratios": [2, 2, 1.5, 1.5]})

    def shade(ax):
        for s in segments:
            ax.axvspan(s["start_s"], s["end_s"], color=FLAG, alpha=0.16, lw=0)

    # v2: candidate-speaking segments (green strip along the top of panel 1)
    speaking = tl["speaking"] if "speaking" in tl.files else None

    ax = axes[0]; _style(ax); shade(ax)
    ax.plot(t, tl["gaze_x"], color=ACCENT, lw=0.8)
    if speaking is not None and np.any(speaking > 0.5):
        sp = speaking > 0.5
        i = 0
        n_sp = len(sp)
        while i < n_sp:
            if sp[i]:
                j = i
                while j < n_sp and sp[j]:
                    j += 1
                ax.axvspan(t[i], t[min(j, n_sp - 1)], ymin=0.92, ymax=1.0,
                           color="#4caf50", alpha=0.55, lw=0)
                i = j
            else:
                i += 1
    ax.set_ylabel("gaze_x\n(L↔R)"); ax.set_ylim(-1.5, 1.5)
    ax.set_title("Gaze trace — red = flagged, green strip = candidate speaking",
                 loc="left", fontsize=10)

    ax = axes[1]; _style(ax); shade(ax)
    ax.plot(t, tl["gaze_y"], color="#81c784", lw=0.8)
    ax.set_ylabel("gaze_y\n(up↔dn)"); ax.set_ylim(-1.5, 1.5)

    ax = axes[2]; _style(ax); shade(ax)
    ax.plot(t, tl["yaw"], color="#ffb74d", lw=0.8, label="yaw")
    ax.plot(t, tl["pitch"], color="#ba68c8", lw=0.8, label="pitch")
    ax.set_ylabel("head°"); ax.legend(loc="upper right", fontsize=7, facecolor=PLOT_BG,
                                       labelcolor=FG, framealpha=0.3)

    ax = axes[3]; _style(ax); shade(ax)
    win_mid = (tl["win_start_s"] + tl["win_end_s"]) / 2
    ax.plot(win_mid, tl["prob"], color=FLAG, lw=1.2, marker="o", ms=2)
    ax.axhline(result["threshold"], color=FG, ls="--", lw=0.8, alpha=0.6)
    ax.set_ylabel("P(cheat)\nper window"); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("time (s)")
    # blink markers along the bottom
    blink_t = t[tl["blink"] > 0.5]
    if len(blink_t):
        ax.scatter(blink_t, np.full(len(blink_t), -0.0) + 0.02, s=2, color="#90a4ae",
                   marker="|", alpha=0.5)

    img = _fig_to_b64(fig)

    seg_rows = "".join(
        f"<tr><td>{s['range']}</td><td>{s['reason']}</td><td>{s['peak_prob']:.2f}</td></tr>"
        for s in segments) or "<tr><td colspan=3>None above flag threshold</td></tr>"

    verdict = result["verdict"]
    is_cheat = verdict == "CHEATING"
    banner_cls = "cheat" if is_cheat else "clean"
    banner_txt = "🚩 CHEATING DETECTED" if is_cheat else "✅ CLEAN"

    html = f"""{_HEAD}
<body>
<div class="wrap">
  <div class="banner {banner_cls}">
    <div class="verdict">{banner_txt}</div>
    <div class="prob">probability {result['probability']:.3f} &nbsp;·&nbsp; threshold {result['threshold']:.3f}</div>
  </div>
  <div class="meta">
    <span><b>video</b> {Path(result['video']).name}</span>
    <span><b>model</b> {result['model']['kind']} · {result['model']['feature_set']}</span>
    <span><b>windows</b> {result['n_windows']} ({result['n_flagged_windows']} flagged)</span>
  </div>
  <img class="plot" src="data:image/png;base64,{img}"/>
  <div class="card">
    <h3>Flagged segments</h3>
    <table><tr><th>range</th><th>reason</th><th>peak p</th></tr>{seg_rows}</table>
  </div>
  <p class="foot">Weak-label caveat: window probabilities reflect learned reading/overlay
  gaze signatures; review flagged segments against the raw video before acting.</p>
</div>
</body></html>"""

    out = REPORTS_DIR / f"{stem}__report.html"
    out.write_text(html)
    print(f"report -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# Evaluation report
# --------------------------------------------------------------------------- #
def _load_metrics():
    """metrics.json is a dict {combos, source_pairs, per_source} in v2 (was a
    bare list in v1) — normalize to the dict form."""
    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        return None
    raw = json.loads(metrics_path.read_text())
    if isinstance(raw, list):
        return dict(combos=raw, source_pairs=[], per_source=[])
    return raw


def build_eval_report():
    blob = _load_metrics()
    if blob is None:
        print("No metrics.json — run train.py first.")
        return 1
    metrics = blob["combos"]

    def cell(v, fmt="{:.3f}"):
        try:
            return fmt.format(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else "—"
        except Exception:
            return "—"

    def _auc_key(x):
        a = x.get("video_roc_auc")
        return a if (a is not None and a == a) else -1.0  # a==a is False for NaN

    rows = ""
    for m in sorted(metrics, key=lambda x: -_auc_key(x)):
        cls = "winner" if m.get("is_winner") else ("dropped" if m.get("dropped_for_deployability") else "")
        tag = " 🏆" if m.get("is_winner") else (" (dropped: deployability)" if m.get("dropped_for_deployability") else "")
        rows += (f"<tr class='{cls}'><td>{m['model']} · {m['feature_set']}{tag}</td>"
                 f"<td>{cell(m.get('video_roc_auc'))}</td><td>{cell(m.get('video_pr_auc'))}</td>"
                 f"<td>{cell(m.get('window_roc_auc'))}</td><td>{cell(m.get('window_pr_auc'))}</td>"
                 f"<td>{cell(m.get('precision'),'{:.2f}')}</td><td>{cell(m.get('recall'),'{:.2f}')}</td>"
                 f"<td>{cell(m.get('threshold'),'{:.2f}')}</td></tr>")

    # per-video OOF table
    vs_rows = ""
    vs_path = MODELS_DIR / "cv_video_scores.csv"
    winner = next((m for m in metrics if m.get("is_winner")), None)
    thr = winner["threshold"] if winner else 0.5
    if vs_path.exists():
        vs = pd.read_csv(vs_path).sort_values("score", ascending=False)
        for _, r in vs.iterrows():
            pred = int(r["score"] >= thr)
            actual = int(r["label"])
            ok = "✓" if pred == actual else "✗"
            rcls = "" if pred == actual else "err"
            vs_rows += (f"<tr class='{rcls}'><td>{r['group']}</td>"
                        f"<td>{'cheating' if actual else 'clean'}</td>"
                        f"<td>{r['score']:.3f}</td>"
                        f"<td>{'cheating' if pred else 'clean'} {ok}</td>"
                        f"<td>{int(r['n_windows'])}</td></tr>")

    # v2: cross-source generalization tables
    pair_rows = ""
    for p in blob.get("source_pairs", []):
        a = p.get("video_auc")
        cls = "winner" if (a is not None and a >= 0.75) else ("err" if (a is not None and a < 0.6) else "")
        pair_rows += (f"<tr class='{cls}'><td>{p['cheat_source']}</td><td>{p['clean_source']}</td>"
                      f"<td>{cell(a)}</td><td>{p['n_test_cheat']}+{p['n_test_clean']}</td></tr>")
    pair_card = ""
    if pair_rows:
        aucs = [p["video_auc"] for p in blob["source_pairs"] if p.get("video_auc") is not None]
        mean_auc = sum(aucs) / len(aucs) if aucs else float("nan")
        pair_card = f"""<div class="card"><h3>Cross-source generalization
      (leave-one-source-pair-out — the honest metric)</h3>
      <p style="opacity:.7;font-size:12px;margin:0 0 10px">The model never saw either
      held-out source's recording setup. Mean pair AUC = <b>{mean_auc:.3f}</b>.</p>
      <table><tr><th>cheating source (held out)</th><th>clean source (held out)</th>
      <th>video AUC</th><th>test videos</th></tr>{pair_rows}</table></div>"""

    src_rows = ""
    for s in blob.get("per_source", []):
        src_rows += (f"<tr><td>{s['label_str']}</td><td>{s['source']}</td><td>{s['n']}</td>"
                     f"<td>{cell(s.get('mean_score'))}</td>"
                     f"<td>{cell(s.get('min_score'))}–{cell(s.get('max_score'))}</td></tr>")
    src_card = ""
    if src_rows:
        src_card = f"""<div class="card"><h3>Per-source mean OOF score (confound tripwire)</h3>
      <p style="opacity:.7;font-size:12px;margin:0 0 10px">Clean sources scoring far apart
      means capture conditions still leak into scores.</p>
      <table><tr><th>label</th><th>source</th><th>videos</th><th>mean score</th><th>range</th></tr>{src_rows}</table></div>"""

    wtxt = f"{winner['model']} · {winner['feature_set']}" if winner else "—"
    html = f"""{_HEAD}
<body>
<div class="wrap">
  <div class="banner clean"><div class="verdict">Evaluation report</div>
    <div class="prob">winner: {wtxt} &nbsp;·&nbsp; leakage-safe grouped CV + cross-source pairs</div></div>
  {pair_card}
  <div class="card"><h3>Cross-validation (grouped by video — optimistic)</h3>
    <table><tr><th>model · features</th><th>video ROC</th><th>video PR</th>
    <th>win ROC</th><th>win PR</th><th>prec</th><th>rec</th><th>thr</th></tr>{rows}</table>
  </div>
  {src_card}
  <div class="card"><h3>Per-video out-of-fold scores</h3>
    <table><tr><th>video</th><th>actual</th><th>OOF score</th><th>predicted</th><th>windows</th></tr>{vs_rows}</table>
  </div>
</div>
</body></html>"""
    out = REPORTS_DIR / "evaluation_report.html"
    out.write_text(html)
    print(f"eval report -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# Overall verdict report — cheating YES/NO for every labeled video
# --------------------------------------------------------------------------- #
def build_verdicts_report():
    """One consolidated CHEATING yes/no table for all videos, using the honest
    out-of-fold scores (each video judged by a model that never trained on it)."""
    vs_path = MODELS_DIR / "cv_video_scores.csv"
    model_path = MODELS_DIR / "model.json"
    if not vs_path.exists() or not model_path.exists():
        print("Missing OOF scores / model — run train.py first.")
        return 1
    art = json.loads(model_path.read_text())
    thr = art["threshold"]
    vs = pd.read_csv(vs_path).sort_values("score", ascending=False)

    rows, n_correct = "", 0
    tp = fp = fn = tn = 0
    console = []
    for _, r in vs.iterrows():
        name = str(r["group"]).split("__", 1)[-1]
        actual_cheat = int(r["label"]) == 1
        verdict_cheat = r["score"] >= thr
        correct = verdict_cheat == actual_cheat
        n_correct += correct
        tp += verdict_cheat and actual_cheat
        fp += verdict_cheat and not actual_cheat
        fn += (not verdict_cheat) and actual_cheat
        tn += (not verdict_cheat) and not actual_cheat
        verdict = "CHEATING" if verdict_cheat else "CLEAN"
        vcls = "cheat" if verdict_cheat else "clean"
        mark = "✓" if correct else "✗ WRONG"
        rcls = "" if correct else "err"
        rows += (f"<tr class='{rcls}'><td>{name[:48]}</td>"
                 f"<td>{'cheating' if actual_cheat else 'clean'}</td>"
                 f"<td>{r['score']:.3f}</td>"
                 f"<td><span class='pill {vcls}'>{verdict}</span></td>"
                 f"<td>{mark}</td></tr>")
        console.append(f"  {verdict:9s} p={r['score']:.3f}  actual={'cheating' if actual_cheat else 'clean':8s} {mark:8s} {name[:46]}")

    n = len(vs)
    acc = n_correct / n if n else 0
    auc = art["metrics"].get("video_roc_auc", float("nan"))

    # console output
    print("\n" + "=" * 70)
    print("  OVERALL CHEATING VERDICTS  (out-of-fold; threshold = %.3f)" % thr)
    print("=" * 70)
    for line in console:
        print(line)
    print("-" * 70)
    print(f"  accuracy {n_correct}/{n} = {acc:.0%}   |   caught {tp}/{tp+fn} cheating, "
          f"{fp} false alarms   |   video ROC-AUC {auc:.3f}")
    print("=" * 70)

    if auc < 0.65:
        warn = ("<div class='warnbox'>⚠️ <b>Reliability warning.</b> Video ROC-AUC is "
                f"{auc:.2f} (0.50 = random). These verdicts are <b>near chance</b> — treat "
                "as a diagnostic, not a detector.</div>")
    else:
        warn = ("<div class='warnbox'>⚠️ <b>Caveat.</b> Video ROC-AUC is "
                f"{auc:.2f} out-of-fold, precision-tuned (few/no false alarms). But the "
                "cheating and clean sets come from <b>different recording sources</b>, so "
                "part of this separation may reflect capture conditions rather than behavior. "
                "Validate on clean videos recorded in the SAME setup before production use.</div>")

    html = f"""{_HEAD}
<body>
<div class="wrap">
  <div class="banner clean"><div class="verdict">Cheating verdicts — all videos</div>
    <div class="prob">out-of-fold · threshold {thr:.3f} · accuracy {acc:.0%} · caught {tp}/{tp+fn} · {fp} false alarms</div></div>
  {warn}
  <div class="card">
    <table><tr><th>video</th><th>actual label</th><th>score</th><th>verdict</th><th>correct?</th></tr>{rows}</table>
  </div>
  <p class="foot">Score = mean of the top-k most-suspicious 20 s windows. Verdict = CHEATING when score ≥ threshold.</p>
</div>
</body></html>"""
    out = REPORTS_DIR / "cheating_verdicts.html"
    out.write_text(html)
    print(f"\nverdict report -> {out}")
    return 0


_HEAD = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gaze cheat-detection report</title>
<style>
 body{margin:0;background:#0b0e12;color:#e6e6e6;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
 .wrap{max-width:1000px;margin:0 auto;padding:24px}
 .banner{border-radius:12px;padding:20px 24px;margin-bottom:16px}
 .banner.cheat{background:linear-gradient(135deg,#4a1414,#2a0d0d);border:1px solid #ff5252}
 .banner.clean{background:linear-gradient(135deg,#12332a,#0d211b);border:1px solid #4caf50}
 .verdict{font-size:24px;font-weight:700}
 .prob{opacity:.8;margin-top:4px}
 .meta{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:16px;opacity:.85;font-size:13px}
 .meta b{color:#4fc3f7}
 img.plot{width:100%;border-radius:10px;border:1px solid #1e2833;background:#0f1419;margin-bottom:16px}
 .cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
 @media(max-width:720px){.cols{grid-template-columns:1fr}}
 .card{background:#12171d;border:1px solid #1e2833;border-radius:10px;padding:16px;margin-bottom:16px;overflow-x:auto}
 h3{margin:0 0 12px;font-size:14px;color:#4fc3f7}
 table{width:100%;border-collapse:collapse;font-size:13px}
 th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #1e2833}
 th{color:#8aa;font-weight:600}
 tr.winner{background:rgba(79,195,247,.12)}
 tr.dropped{opacity:.5}
 .pill{padding:2px 10px;border-radius:20px;font-weight:700;font-size:12px}
 .pill.cheat{background:#4a1414;color:#ff8a80;border:1px solid #ff5252}
 .pill.clean{background:#12332a;color:#a5d6a7;border:1px solid #4caf50}
 .warnbox{background:#3a2a0d;border:1px solid #ffb74d;border-radius:10px;padding:14px 16px;margin-bottom:16px;font-size:13px}
 tr.err{background:rgba(255,82,82,.12)}
 .bar{height:10px;background:#4fc3f7;border-radius:3px;min-width:2px}
 .foot{opacity:.55;font-size:12px;margin-top:8px}
</style></head>"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 4: HTML reports.")
    ap.add_argument("video", nargs="?", help="video to build a per-video report for")
    ap.add_argument("--eval", action="store_true", help="build the overall evaluation report")
    ap.add_argument("--verdicts", action="store_true",
                    help="overall CHEATING yes/no table for every labeled video")
    ap.add_argument("--all", action="store_true", help="eval report + every video with a prediction")
    args = ap.parse_args(argv)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.all:
        build_eval_report()
        build_verdicts_report()
        for pj in sorted(REPORTS_DIR.glob("*__prediction.json")):
            stem = pj.name[:-len("__prediction.json")]
            build_video_report(Path(stem))
        return 0
    if args.verdicts:
        return build_verdicts_report()
    if args.eval:
        return build_eval_report()
    if args.video:
        return build_video_report(Path(args.video))
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
