"""
report.py (v2) — per-video HTML review + overall evaluation report.

Per-video (needs a prediction from predict.py; auto-runs it if missing):
    horizontal / vertical eye traces, mouth activity with the speaking lane, and
    the per-second reading probability — detected reading segments shaded red and,
    when a manual review exists for the video's IR number, the reviewer's spans as
    an amber band — plus the segment table and the agreement numbers.

Evaluation report: the latest shipped model's out-of-fold results (per reviewed
video and video level) and the per-video table.

Usage:
    python src/report.py path/to/video.mp4      # per-video report
    python src/report.py --eval                 # overall evaluation report
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import MODELS_DIR, REPORTS_DIR, RESULTS_DIR
import labels as lab

PLOT_BG = "#0f1419"
FG = "#e6e6e6"
ACCENT = "#4fc3f7"
FLAG = "#ff5252"
REVIEW = "#f0a726"
SPEAK = "#4caf50"


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
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


def _spans(mask, t):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((t[i], t[min(j, n - 1)]))
            i = j
        else:
            i += 1
    return out


# --------------------------------------------------------------------------- #
# Per-video report
# --------------------------------------------------------------------------- #
def _plot_rows(tl, result, review, t_lo, t_hi):
    dt, t0 = float(tl["dt"][0]), float(tl["t0"][0])
    t = t0 + np.arange(len(tl["eye_h"])) * dt
    m = (t >= t_lo) & (t < t_hi)
    sec = tl["second_prob"]
    ts = np.arange(len(sec)) + 0.5
    ms = (ts >= t_lo) & (ts < t_hi)
    segs = result["flagged_segments"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 7.2), sharex=True, facecolor=PLOT_BG,
                             gridspec_kw={"height_ratios": [2, 1.4, 1.4, 1.6]})

    def shade(ax, band=False):
        for s in segs:
            if s["end_s"] > t_lo and s["start_s"] < t_hi:
                ax.axvspan(s["start_s"], s["end_s"], color=FLAG, alpha=0.16, lw=0)
        if review is not None:
            for a, b, *_ in review.segments:
                if b > t_lo and a < t_hi:
                    ax.axvspan(a, b, ymin=0.93, ymax=1.0, color=REVIEW, alpha=0.85, lw=0)

    ax = axes[0]; _style(ax); shade(ax)
    ax.plot(t[m], tl["eye_h"][m], color=ACCENT, lw=0.6)
    ax.set_ylabel("eyes L↔R")
    ax.set_title("red = detected reading" + (" · amber band = reviewer's reading spans" if review else ""),
                 loc="left", fontsize=9)
    ax = axes[1]; _style(ax); shade(ax)
    ax.plot(t[m], tl["eye_v"][m], color="#81c784", lw=0.6)
    ax.set_ylabel("eyes ↑↓")
    ax = axes[2]; _style(ax); shade(ax)
    ax.plot(t[m], tl["mouth"][m], color="#ce93d8", lw=0.5)
    sp = tl["speaking"][m] > 0.5
    for a, b in _spans(sp, t[m]):
        ax.axvspan(a, b, ymin=0, ymax=0.08, color=SPEAK, alpha=0.8, lw=0)
    ax.set_ylabel("mouth\n(green=speaking)")
    ax = axes[3]; _style(ax); shade(ax)
    ax.plot(ts[ms], sec[ms], color=FLAG, lw=1.0)
    ax.axhline(result["second_threshold"], color=FG, ls="--", lw=0.8, alpha=0.6)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("P(reading)\nper second")
    ax.set_xlabel("video time (s)")
    ax.set_xlim(t_lo, t_hi)
    return _fig_to_b64(fig)


def build_video_report(video: Path, review=None):
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
    if review is None:
        review = lab.review_for(Path(result["video"]).name)

    dur = len(tl["second_prob"])
    chunk = 300                                   # 5-minute rows keep the traces readable
    imgs = [_plot_rows(tl, result, review, a, min(dur, a + chunk)) for a in range(0, dur, chunk)]

    esc = html.escape
    seg_rows = "".join(
        f"<tr><td>{esc(s['range'])}</td><td>{s['duration_s']:.0f}s</td><td>{esc(s['reason'])}</td>"
        f"<td>{s['peak_prob']:.2f}</td></tr>" for s in result["flagged_segments"]
    ) or "<tr><td colspan=4>No reading segments above the threshold</td></tr>"

    review_card = ""
    if review is not None:
        from evaluate import per_second_metrics
        sec = tl["second_prob"]
        truth = lab.truth_mask(review.segments, len(sec), 1.0)
        flag = np.zeros(len(sec), bool)
        for s in result["flagged_segments"]:
            flag[int(s["start_s"]):int(s["end_s"])] = True
        mt = per_second_metrics(sec, truth, flag, review.segments)
        span_rows = ""
        for a, b, tag, reason in review.segments:
            cov = flag[int(a):int(b)].mean() if min(b, len(flag)) > a else 0.0
            span_rows += (f"<tr class='{'' if cov >= 0.5 else 'err'}'><td>{lab.mmss(a)}–{lab.mmss(b)}</td>"
                          f"<td>{cov:.0%}</td><td>{esc(reason[:110])}</td></tr>")
        review_card = f"""<div class="card"><h3>Agreement with the manual review ({esc(review.path.name)})</h3>
        <p class="kv"><span>per-second AUC <b>{mt['auc']:.3f}</b></span>
        <span>precision <b>{mt['precision']:.2f}</b></span><span>recall <b>{mt['recall']:.2f}</b></span>
        <span>IoU <b>{mt['iou']:.2f}</b></span><span>reviewer spans caught <b>{mt['spans_caught']}</b></span></p>
        <table><tr><th>reviewer span</th><th>covered by detection</th><th>reviewer note</th></tr>{span_rows}</table></div>"""

    is_cheat = result["verdict"] == "CHEATING"
    banner_cls = "cheat" if is_cheat else "clean"
    banner_txt = "🚩 READING DETECTED" if is_cheat else "✅ CLEAN"
    rule = (f"{result.get('verdict_rule', '')} (this video: {result.get('video_stat_value', 0):.3f})"
            .replace(">=", "≥"))
    plots = "".join(f'<img class="plot" src="data:image/png;base64,{im}"/>' for im in imgs)
    page = f"""{_HEAD}
<body>
<div class="wrap">
  <div class="banner {banner_cls}">
    <div class="verdict">{banner_txt}</div>
    <div class="prob">reading {lab.mmss(result['reading_seconds'])} in {result['reading_segments']} segment(s)
      &nbsp;·&nbsp; {result['reading_share_of_speech']:.0%} of speaking time &nbsp;·&nbsp; {rule}</div>
  </div>
  <div class="meta">
    <span><b>video</b> {esc(Path(result['video']).name)}</span>
    <span><b>model</b> v2 reading detector ({esc(str(result['model'].get('experiment')))})</span>
    <span><b>per-second threshold</b> {result['second_threshold']:.2f}</span>
  </div>
  {review_card}
  <div class="card"><h3>Reading segments</h3>
    <table><tr><th>range</th><th>length</th><th>signals</th><th>peak p</th></tr>{seg_rows}</table></div>
  {plots}
  <p class="foot">A reading segment is evidence to review, never a conclusion: open the ranges in the
  recording before acting. Signals compare the segment with how clean candidates look while answering.</p>
</div>
</body></html>"""
    out = REPORTS_DIR / f"{stem}__report.html"
    out.write_text(page)
    print(f"report -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# Evaluation report
# --------------------------------------------------------------------------- #
def build_eval_report():
    mpath = MODELS_DIR / "model.json"
    if not mpath.exists():
        print("No model.json — run train.py first.")
        return 1
    art = json.loads(mpath.read_text())
    exp = art.get("experiment", "")
    cv = art.get("cv", {})
    ps_rows = "".join(
        f"<tr><td>{html.escape(r['video'])}</td><td>{r['auc']:.3f}</td><td>{r['precision']:.2f}</td>"
        f"<td>{r['recall']:.2f}</td><td>{r['iou']:.2f}</td><td>{r.get('spans_caught', '')}</td></tr>"
        for r in cv.get("per_second", []))
    v = cv.get("video", {})
    vs_path = RESULTS_DIR / f"oof_video_scores__{exp}.csv"
    vrows = ""
    if vs_path.exists():
        vs = pd.read_csv(vs_path, float_precision="round_trip").sort_values("score", ascending=False)
        thr = art["video_threshold"]
        for _, r in vs.iterrows():
            pred = r["score"] >= thr
            ok = pred == bool(r["label"])
            vrows += (f"<tr class='{'' if ok else 'err'}'><td>{html.escape(r['video'].split('__')[-1])}</td>"
                      f"<td>{'cheating' if r['label'] else 'clean'}{' (reviewed)' if r['reviewed'] else ''}</td>"
                      f"<td>{lab.mmss(r['reading_seconds'])}</td><td>{'CHEATING' if pred else 'clean'}"
                      f" {'✓' if ok else '✗'}</td></tr>")
    page = f"""{_HEAD}
<body><div class="wrap">
  <div class="banner clean"><div class="verdict">v2 evaluation — {html.escape(exp)}</div>
    <div class="prob">out-of-fold · reviewed videos held out one per fold · unreviewed cheating never trained on</div></div>
  <div class="card"><h3>Per-second agreement with the manual reviews</h3>
    <table><tr><th>video</th><th>AUC</th><th>precision</th><th>recall</th><th>IoU</th><th>spans caught</th></tr>{ps_rows}</table></div>
  <div class="card"><h3>Video level</h3><p class="kv">
    <span>AUC <b>{v.get('auc', float('nan')):.3f}</b></span><span>precision <b>{v.get('precision', 0):.2f}</b></span>
    <span>recall <b>{v.get('recall', 0):.2f}</b></span><span>false alarms <b>{v.get('fp', '?')}</b></span>
    <span>same-platform AUC <b>{v.get('matched_auc', float('nan')):.3f}</b></span></p></div>
  <div class="card"><h3>Per-video out-of-fold verdicts</h3>
    <table><tr><th>video</th><th>actual</th><th>reading time</th><th>verdict</th></tr>{vrows}</table></div>
</div></body></html>"""
    out = REPORTS_DIR / "evaluation_report.html"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"eval report -> {out}")
    return 0


_HEAD = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reading detection report</title>
<style>
 body{margin:0;background:#0b0e12;color:#e6e6e6;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
 .wrap{max-width:1100px;margin:0 auto;padding:24px}
 .banner{border-radius:12px;padding:20px 24px;margin-bottom:16px}
 .banner.cheat{background:linear-gradient(135deg,#4a1414,#2a0d0d);border:1px solid #ff5252}
 .banner.clean{background:linear-gradient(135deg,#12332a,#0d211b);border:1px solid #4caf50}
 .verdict{font-size:24px;font-weight:700}
 .prob{opacity:.85;margin-top:4px}
 .meta{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:16px;opacity:.85;font-size:13px}
 .meta b{color:#4fc3f7}
 .kv{display:flex;gap:18px;flex-wrap:wrap;margin:0 0 12px}
 .kv b{color:#f0a726}
 img.plot{width:100%;border-radius:10px;border:1px solid #1e2833;background:#0f1419;margin-bottom:12px}
 .card{background:#12171d;border:1px solid #1e2833;border-radius:10px;padding:16px;margin-bottom:16px;overflow-x:auto}
 h3{margin:0 0 12px;font-size:14px;color:#4fc3f7}
 table{width:100%;border-collapse:collapse;font-size:13px}
 th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #1e2833}
 th{color:#8aa;font-weight:600}
 tr.err{background:rgba(255,82,82,.12)}
 .foot{opacity:.55;font-size:12px;margin-top:8px}
</style></head>"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 4 (v2): HTML reports.")
    ap.add_argument("video", nargs="?", help="video to build a per-video report for")
    ap.add_argument("--eval", action="store_true", help="build the overall evaluation report")
    args = ap.parse_args(argv)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.eval:
        return build_eval_report()
    if args.video:
        return build_video_report(Path(args.video))
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
