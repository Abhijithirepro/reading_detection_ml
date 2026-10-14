"""
cheating_cases_xlsx.py (v2) — detailed workbooks for Google Sheets / Excel, one for
the cheating candidates and one for the not-cheating (clean) candidates.

  First tab "How to read this"
                           a plain-language guide for someone new to the project: what
                           is measured, how a verdict is reached, every tab, every column
                           (with an example value from this workbook), the 7 behaviour
                           measurements and their normal values, colours, glossary.
  Tab 1 "Cheating cases" / "Not cheating cases"
                           one row per candidate: verdict, score, reading time, every
                           time reading was caught / flagged, main signs, review match,
                           v1 and the archived 25 fps model side by side.
  Tab 2 "Reading timeline" / "Flagged moments"
                           every moment the model flagged as reading, one row each: when,
                           how long, strongest second, confidence, what the eyes /
                           blinks / head did (measured, next to normal speaking), and
                           the reviewer's verdict where there is a review. Reviewer
                           spans the model missed are listed too; a candidate with no
                           moment gets one row with its strongest (unflagged) moment.
  One tab per IR id        that candidate's summary, the same moments in full, the
                           reviewer's spans, the 5 strongest moments, minute by minute,
                           second by second.

For a cheating candidate the flagged moments are reading the model caught; for a
clean candidate they are false alarms (or reading nobody knew about) — the clean
workbook shows where and why the model was suspicious.

Everything comes from the web-app library (webapp/library/, built by
webapp/library.py), so the numbers are the same OUT-OF-FOLD scores as
results/ALL_VIDEOS_RESULTS.*: no video is scored by a model that trained on it. The
eye / blink / head measurements come from data/windows/windows.parquet, the same
8 s windows the model scored; "normal" is the median of clean candidates while they
speak (model.json reason_reference). The v1 and 25 fps columns come from
results/ALL_VIDEOS_RESULTS.csv.

Writes results/CHEATING_CASES_REPORT.xlsx and results/NOT_CHEATING_CASES_REPORT.xlsx
(Google Sheets: File → Import → Upload keeps every tab). --v2-only leaves out the v1 and
25 fps comparison and writes *_v2_only.xlsx instead.

Usage (from the project root, after webapp/library.py):
    venv/bin/python v2/src/cheating_cases_xlsx.py                 # both workbooks
    venv/bin/python v2/src/cheating_cases_xlsx.py --only clean    # one of them
    venv/bin/python v2/src/cheating_cases_xlsx.py --only clean --v2-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from copy import copy

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.hyperlink import Hyperlink

from common import PROJECT_ROOT, MODELS_DIR, RESULTS_DIR, WINDOWS_DIR
from predict import segment_reasons

LIB_DIR = PROJECT_ROOT / "webapp" / "library"
GUIDE_TAB = "How to read this"
BORDERLINE = 0.05            # within 5 % of the verdict threshold (as all_videos_results.py)
CONFIDENCE = ((0.6, "High"), (0.35, "Medium"), (0.0, "Low"))   # on a moment's average probability
TOP_K, TOP_GAP_S, TOP_HALF_S = 5, 30, 3                         # strongest moments: count, spacing, ± width

# wording per workbook; `right` / `wrong` label a verdict that matches / contradicts the folder
MODES = {
    "cheating": dict(
        out="CHEATING_CASES_REPORT.xlsx", summary="Cheating cases", timeline="Reading timeline",
        who="Every candidate in cheating_folder/ (known cheaters)", truth="cheating",
        right="CAUGHT", wrong="MISSED", verb="caught", moments="reading moments",
        moment_title="When reading was caught",
        moment_meaning="A stretch of seconds the model flagged as reading."),
    "clean": dict(
        out="NOT_CHEATING_CASES_REPORT.xlsx", summary="Not cheating cases", timeline="Flagged moments",
        who="Every candidate in not_cheating_folder/ and new_not_cheating/ (known clean)", truth="clean",
        right="CLEARED", wrong="FLAGGED", verb="flagged", moments="flagged moments",
        moment_title="When the model flagged reading",
        moment_meaning="A stretch of seconds the model flagged as reading. These candidates are known to be "
                       "clean, so each one is a false alarm (unless the candidate really read briefly)."),
}

# eye / blink / head measurements shown per moment:
# (window feature, column header, number format, sentence used when it is one of the moment's signs)
MEASURES = [
    ("blink_rate", "Blinks per min", "0",
     lambda v, n: f"blinking {v:.0f}/min (normal ~{n:.0f})"),
    ("head_speed", "Head movement (°/s)", "0.0",
     lambda v, n: f"head almost still ({v:.1f}°/s, normal ~{n:.1f})"),
    ("h_prog_frac", "Eyes drifting along a line", "0%",
     lambda v, n: f"eyes drifting along a line {v:.0%} of the time (normal ~{n:.0%})"),
    ("big_jump_rate", "Big eye jumps per min", "0",
     lambda v, n: f"few big eye jumps ({v:.0f}/min, normal ~{n:.0f})"),
    ("h_std", "Side-to-side eye spread", "0.000",
     lambda v, n: f"eyes steady (side-to-side spread {v:.3f}, normal ~{n:.3f})"),
    ("v_mean_listen", "Eye height vs listening (+ = higher)", "+0.000;-0.000;0.000",
     lambda v, n: f"eyes held higher than while listening to the question ({v:+.3f})"),
    ("brow_down_mean", "Brow lowered (0–1)", "0.00",
     lambda v, n: f"brow lowered ({v:.2f}, normal ~{n:.2f})"),
]
FEATS = [f for f, *_ in MEASURES]
COMPARE_ONLY = {"v1 score", "v1 result", "v2 @ 25 fps score", "v2 @ 25 fps result"}
REVIEW_ONLY = {"Manually reviewed", "Reviewer spans caught", "Reviewer reading seconds flagged (recall)",
               "Flagged seconds the reviewer marked (precision)", "Reviewer check", "Reviewer note"}

HEAD_FILL, HEAD_FONT = PatternFill("solid", fgColor="1F3864"), Font(bold=True, color="FFFFFF")
GOOD_FILL, GOOD_FONT = PatternFill("solid", fgColor="C6EFCE"), Font(bold=True, color="006100")
BAD_FILL, BAD_FONT = PatternFill("solid", fgColor="FFC7CE"), Font(bold=True, color="9C0006")
WARN_FILL = PatternFill("solid", fgColor="FFEB9C")
READ_FILL = PatternFill("solid", fgColor="FCE4D6")      # seconds inside a flagged moment
CONF_FILL = {"High": PatternFill("solid", fgColor="F4B183"), "Medium": PatternFill("solid", fgColor="F8CBAD"),
             "Low": PatternFill("solid", fgColor="FFF2CC")}
REF_FONT = Font(italic=True, color="595959")
LINK_FONT = Font(color="0563C1", underline="single")
TITLE_FONT, SECTION_FONT = Font(bold=True, size=14), Font(bold=True, size=12, color="1F3864")
TOP = Alignment(vertical="top", wrap_text=True)


def mmss(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def span(s: float, e: float) -> str:
    return f"{mmss(s)}–{mmss(e)}"


def num(v):
    return None if v is None or not np.isfinite(v) else float(v)


def confidence(avg: float) -> str:
    return next(lab for lo, lab in CONFIDENCE if avg >= lo)


def sheet_name(ir: str, name: str, used: set) -> str:
    """'ir8266049 Sai Koti Reddy Sai' — Excel/Sheets allow 31 chars and no []:*?/\\."""
    base = re.sub(r"[\[\]:*?/\\']", "", f"ir{ir} {name}")
    if len(base) > 31:
        base = base[:31].rsplit(" ", 1)[0]
    out, k = base, 2
    while out in used:
        out = f"{base[:27]} ({k})"
        k += 1
    used.add(out)
    return out


def link(cell, sheet: str, text: str) -> None:
    cell.value = text
    cell.hyperlink = Hyperlink(ref=cell.coordinate, location=f"'{sheet}'!A1", display=text)
    cell.font = LINK_FONT


def result_cell(cell, flagged: bool | None, M: dict) -> None:
    """CAUGHT / MISSED for a cheating candidate, FLAGGED / CLEARED for a clean one;
    green when the verdict matches the folder, red when it does not."""
    if flagged is None:
        return
    right = flagged == (M["truth"] == "cheating")
    cell.value = M["right"] if right else M["wrong"]
    cell.fill, cell.font = (GOOD_FILL, GOOD_FONT) if right else (BAD_FILL, BAD_FONT)


def header(ws, row: int, names: list[str]) -> None:
    for j, h in enumerate(names, 1):
        c = ws.cell(row, j, h)
        c.fill, c.font = HEAD_FILL, HEAD_FONT
        c.alignment = Alignment(wrap_text=True, vertical="center")


def widths(ws, cols: list[float]) -> None:
    for j, w in enumerate(cols, 1):
        ws.column_dimensions[get_column_letter(j)].width = w


def put(ws, row: int, values: dict, cols: list[tuple]) -> None:
    """Write `values` (keyed by column header) into `row` using `cols` = [(header, width, fmt)]."""
    for j, (h, _, f) in enumerate(cols, 1):
        v = values.get(h)
        c = ws.cell(row, j, num(v) if isinstance(v, (float, np.floating)) else v)
        c.alignment = TOP
        if f:
            c.number_format = f


def col(cols: list[tuple], name: str) -> int:
    return [h for h, _, _ in cols].index(name) + 1


def drop(cols: list[tuple], names: set) -> list[tuple]:
    return [c for c in cols if c[0] not in names]


# --------------------------------------------------------------------------- data

def load(truth: str, v2_only: bool = False):
    index = json.loads((LIB_DIR / "index.json").read_text())
    art = json.loads((MODELS_DIR / "model.json").read_text())
    if index["experiment"] != art["experiment"]:
        sys.exit(f"library is from {index['experiment']} but the shipped model is {art['experiment']} "
                 f"— run webapp/library.py --force first")
    allv = pd.read_csv(RESULTS_DIR / "ALL_VIDEOS_RESULTS.csv", float_precision="round_trip")
    allv.index = allv["name"].str.extract(r"ir(\d+)")[0]
    refs = {r["feature"]: r for r in art.get("reason_reference", [])}
    keep = ["video", "win_start_s", "win_end_s"] + sorted(set(FEATS) | set(refs))
    win = pd.read_parquet(WINDOWS_DIR / "windows.parquet", columns=keep)
    win = win[win["video"].str.startswith(f"{truth}__")]
    win_ir = win["video"].str.extract(r"_ir(\d+)_")[0]
    cases = [case(c, allv.loc[c["ir"]] if c["ir"] in allv.index and not v2_only else None,
                  win[win_ir == c["ir"]], art)
             for c in index["candidates"] if c["actual"] == truth]
    cases.sort(key=lambda x: -x["score"])
    return index, art, cases


def measure(win: pd.DataFrame, a: float, b: float) -> dict:
    """Mean of each measurement over the windows centred in [a, b) — the windows
    segment_reasons() looks at."""
    mid = (win["win_start_s"] + win["win_end_s"]) / 2
    sub = win[(mid >= a) & (mid < b)]
    return {f: float(sub[f].mean()) if len(sub) and f in sub else np.nan for f in FEATS}


def describe(speaking: float, reasons: list[str], m: dict, art: dict) -> str:
    """Plain sentence: what the candidate was doing during a moment, with the numbers."""
    by_label = {r["label"]: r for r in art.get("reason_reference", [])}
    text = {f: fn for f, _, _, fn in MEASURES}
    parts = []
    for lab in reasons:
        r = by_label.get(lab)
        if r and r["feature"] in text and np.isfinite(m.get(r["feature"], np.nan)):
            parts.append(text[r["feature"]](m[r["feature"]], r["clean_median"]))
        else:
            parts.append(lab)
    lead = f"Answering ({speaking:.0%} of the time speaking)" if speaking >= 0.5 else \
        f"Mostly silent ({speaking:.0%} speaking)"
    return lead + (": " + "; ".join(parts) if parts else
                   ": no single sign stands out, flagged on the combined eye/head/mouth pattern")


def moment(d: dict, a: int, b: int, reasons: list[str] | None = None) -> dict:
    """Everything the report shows for seconds [a, b) of one video."""
    p = d["prob"][a:b]
    ok = np.isfinite(p)
    peak_i = a + int(np.nanargmax(np.where(ok, p, -1))) if ok.any() else a
    avg = float(p[ok].mean()) if ok.any() else np.nan
    speaking = float(d["speak"][a:b].mean()) if b > a else 0.0
    if reasons is None:
        reasons = segment_reasons(d["win"], a, b, d["art"])
    m = measure(d["win"], a, b)
    return dict(a=a, b=b, avg=avg, peak=float(d["prob"][peak_i]) if ok.any() else np.nan, peak_at=peak_i,
                speaking=speaking, reasons=reasons, m=m, conf=confidence(avg) if np.isfinite(avg) else "",
                kind="reading while answering" if speaking >= 0.5 else "reading on screen",
                saw=describe(speaking, reasons, m, d["art"]))


def case(meta: dict, other, win: pd.DataFrame, art: dict) -> dict:
    """One candidate: the library payload plus everything derived from it."""
    p = json.loads((LIB_DIR / meta["id"] / "payload.json").read_text())
    r, review, stats = p["result"], p.get("review"), p["stats"]
    prob = np.array([np.nan if x is None else x for x in r["seconds"]], float)
    d = dict(meta=meta, result=r, review=review, stats=stats, prob=prob, win=win, art=art,
             speak=np.array(r["speaking"], float), score=r["video_stat_value"], thr=r["video_threshold"],
             flagged=r["verdict"] == "CHEATING", v1_score=None, v1_flagged=None, f25_score=None,
             f25_flagged=None)
    n = len(prob)

    # flagged moments (the model's segments), numbered in time order
    seg_of = np.zeros(n, int)                       # moment number covering each second (0 = none)
    d["moments"] = []
    for k, s in enumerate(r["flagged_segments"], 1):
        a, b = int(s["start_s"]), int(s["end_s"])
        seg_of[a:b] = k
        mo = moment(d, a, b, s["reasons"])
        mo.update(k=k, avg=s["mean_prob"], peak=s["peak_prob"], conf=confidence(s["mean_prob"]))
        d["moments"].append(mo)
    d["seg_of"] = seg_of

    # reviewer spans (reviewed candidates only) and how each moment relates to them
    marked = np.zeros(n, bool)
    d["spans"] = []
    for s in (review or {}).get("spans", []):
        a, b = int(s["start_s"]), int(s["end_s"])
        marked[a:b] = True
        mo = moment(d, a, min(b, n))
        mo.update(review=s, overlaps=[m["k"] for m in d["moments"] if m["a"] < b and m["b"] > a])
        d["spans"].append(mo)
    d["marked"] = marked
    for mo in d["moments"]:
        hit = [s for s in d["spans"] if s["a"] < mo["b"] and s["b"] > mo["a"]]
        if not review:
            mo.update(check="not reviewed", note="")
        elif hit:
            mo.update(check="Confirmed — reviewer marked " + ", ".join(span(s["a"], s["b"]) for s in hit),
                      note=" | ".join(s["review"]["reason"] for s in hit))
        else:
            mo.update(check="Extra — not in the review", note="")

    # the strongest moments, flagged or not (where to look first)
    pp = np.where(np.isfinite(prob), prob, -1)
    picks = []
    for i in np.argsort(-pp, kind="stable"):
        if pp[i] < 0 or len(picks) == TOP_K:
            break
        if all(abs(int(i) - j) >= TOP_GAP_S for j in picks):
            picks.append(int(i))
    d["top"] = []
    for i in picks:
        mo = moment(d, max(0, i - TOP_HALF_S), min(n, i + TOP_HALF_S + 1))
        mo["flag"] = f"Yes — inside moment #{seg_of[i]}" if seg_of[i] else "No — below the reading threshold"
        d["top"].append(mo)

    signs = Counter()
    for mo in d["moments"]:
        for t in mo["reasons"]:
            signs[t] += mo["b"] - mo["a"]
    d["signs"] = ", ".join(t for t, _ in signs.most_common(3))
    d["longest"] = max(d["moments"], key=lambda m: m["b"] - m["a"]) if d["moments"] else None
    d["strongest"] = d["top"][0] if d["top"] else None
    if other is not None:
        d.update(v1_score=float(other["v1_score"]), v1_flagged=other["v1_verdict"] == "CHEATING",
                 f25_score=float(other["v2_25fps_score"]), f25_flagged=other["v2_25fps_verdict"] == "CHEATING")
    d["notes"] = notes(d)
    return d


def notes(d: dict) -> str:
    cheat = d["meta"]["actual"] == "cheating"
    word = lambda f: ("CAUGHT" if f else "MISSED") if cheat else ("FLAGGED" if f else "CLEARED")
    out = []
    if abs(d["score"] - d["thr"]) <= BORDERLINE * d["thr"]:
        out.append("borderline: score within 5% of the threshold")
    if d["v1_flagged"] is not None:
        v2, v1 = d["flagged"], d["v1_flagged"]
        if cheat:
            out += (["caught by v2 only (v1 missed)"] if v2 and not v1 else
                    ["caught by v1 only (v2 missed)"] if v1 and not v2 else
                    ["missed by both v2 and v1"] if not v2 else [])
        else:
            out += (["wrongly flagged by v2 only (v1 cleared)"] if v2 and not v1 else
                    ["wrongly flagged by v1 only (v2 cleared)"] if v1 and not v2 else
                    ["wrongly flagged by both v2 and v1"] if v2 else [])
    elif cheat and not d["flagged"]:
        out.append("missed: v2 says NOT CHEATING")
    elif not cheat and d["flagged"]:
        out.append("false alarm: v2 says CHEATING but the candidate is clean")
    if d["f25_flagged"] is not None and d["f25_flagged"] != d["flagged"]:
        out.append(f"verdict changed with 15 fps (25 fps model: {word(d['f25_flagged'])})")
    if not d["flagged"] and d["moments"]:
        out.append(f"{len(d['moments'])} short moment(s) flagged, too little for a CHEATING verdict")
    if not d["moments"]:
        out.append("no moment flagged as reading")
    if d["stats"]["sampled_fps"] < 14.5:
        out.append(f"recorded at {d['stats']['sampled_fps']:.1f} fps (slower than 15, every frame used)")
    if d["stats"]["face_ratio"] < 0.9:
        out.append(f"face found in only {d['stats']['face_ratio']:.0%} of frames")
    if d["review"]:
        out.append("manually reviewed: reviewer timestamps compared in the tab")
    return "; ".join(out)


def measure_cols(width: float = 11) -> list[tuple]:
    return [(h, width, f) for _, h, f, _ in MEASURES]


def measure_vals(m: dict) -> dict:
    return {h: m.get(f) for f, h, _, _ in MEASURES}


def normal_vals(art: dict) -> dict:
    refs = {r["feature"]: r["clean_median"] for r in art.get("reason_reference", [])}
    return {h: refs.get(f) for f, h, _, _ in MEASURES}


NORMAL_TEXT = "Normal = median of clean candidates while they speak →"


# --------------------------------------------------------------------------- tab 1

SUMMARY_COLS = [
    # (header, width, number format); "Moments at" is renamed per workbook
    ("#", 4, None), ("IR ID", 12, None), ("Candidate name", 24, None), ("Detail tab", 27, None),
    ("Folder", 17, None), ("Manually reviewed", 10, None), ("v2 verdict", 13, None), ("v2 result", 10, None),
    ("v2 score", 8, "0.000"),
    ("Threshold", 9, "0.000"), ("Score − threshold", 10, "+0.000;-0.000;0.000"),
    ("Video length", 9, None), ("Speaking time", 9, None), ("Reading detected", 9, None),
    ("Reading (seconds)", 9, "0"), ("Reading time ÷ speaking time", 10, "0%"),
    ("Moments", 9, "0"), ("Moments at", 60, None),
    ("First moment at", 9, None), ("Longest moment", 17, None), ("Strongest moment", 20, None),
    ("High-confidence moments", 11, "0"), ("Main reading signs", 40, None),
    ("Reviewer spans caught", 10, None), ("Reviewer reading seconds flagged (recall)", 11, "0%"),
    ("Flagged seconds the reviewer marked (precision)", 11, "0%"),
    ("v1 score", 8, "0.000"), ("v1 result", 10, None),
    ("v2 @ 25 fps score", 9, "0.000"), ("v2 @ 25 fps result", 10, None),
    ("Processed fps", 8, "0.0"), ("Face found", 8, "0%"), ("Frames processed", 10, "#,##0"),
    ("Video file", 44, None), ("Notes", 60, None),
]


def write_summary(ws, index: dict, cases: list[dict], tabs: list[str], M: dict, has_review: bool) -> None:
    cols = drop(SUMMARY_COLS, (set() if has_review else REVIEW_ONLY) | (COMPARE_ONLY if M["v2_only"] else set()))
    names = {"Moments": f"{M['moments'].capitalize()}",
             "Moments at": f"Reading {M['verb']} at (every moment)"}
    header(ws, 1, [names.get(h, h) for h, _, _ in cols])
    widths(ws, [w for _, w, _ in cols])
    ws.row_dimensions[1].height = 62
    for i, (d, tab) in enumerate(zip(cases, tabs), 1):
        m, r, rv = d["meta"], d["result"], d["review"]
        L, S = d["longest"], d["strongest"]
        met = (rv or {}).get("metrics", {})
        put(ws, i + 1, {
            "#": i, "IR ID": f"ir{m['ir']}", "Candidate name": m["name"], "Folder": m["folder"],
            "Manually reviewed": "Yes" if rv else "No", "v2 score": d["score"], "Threshold": d["thr"],
            "v2 verdict": "CHEATING" if d["flagged"] else "NOT CHEATING",
            "Score − threshold": d["score"] - d["thr"], "Video length": mmss(d["stats"]["video_seconds"]),
            "Speaking time": mmss(r["speaking_seconds"]), "Reading detected": mmss(r["reading_seconds"]),
            "Reading (seconds)": r["reading_seconds"], "Reading time ÷ speaking time": r["reading_share_of_speech"],
            "Moments": len(d["moments"]),
            "Moments at": ", ".join(span(mo["a"], mo["b"]) for mo in d["moments"]) or "—",
            "First moment at": mmss(d["moments"][0]["a"]) if d["moments"] else "—",
            "Longest moment": f"{span(L['a'], L['b'])} ({L['b'] - L['a']} s)" if L else "—",
            "Strongest moment": f"{mmss(S['peak_at'])} (prob {S['peak']:.2f})" if S else "—",
            "High-confidence moments": sum(mo["conf"] == "High" for mo in d["moments"]),
            "Main reading signs": d["signs"] or "—",
            "Reviewer spans caught": met.get("spans_caught", "").replace("/", " / ") or None,
            "Reviewer reading seconds flagged (recall)": met.get("recall"),
            "Flagged seconds the reviewer marked (precision)": met.get("precision"),
            "v1 score": d["v1_score"], "v2 @ 25 fps score": d["f25_score"],
            "Processed fps": d["stats"]["sampled_fps"], "Face found": d["stats"]["face_ratio"],
            "Frames processed": d["stats"]["n_frames"], "Video file": m["file"], "Notes": d["notes"],
        }, cols)
        link(ws.cell(i + 1, col(cols, "Detail tab")), tab, tab)
        result_cell(ws.cell(i + 1, col(cols, "v2 result")), d["flagged"], M)
        if not M["v2_only"]:
            result_cell(ws.cell(i + 1, col(cols, "v1 result")), d["v1_flagged"], M)
            result_cell(ws.cell(i + 1, col(cols, "v2 @ 25 fps result")), d["f25_flagged"], M)
        if "borderline" in d["notes"]:
            ws.cell(i + 1, col(cols, "v2 score")).fill = WARN_FILL
    last = len(cases) + 1
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{last}"
    ws.freeze_panes = "D2"

    # totals under the table (a blank row keeps them out of the filter range)
    n = len(cases)
    right = lambda f: f == (M["truth"] == "cheating")
    with_moments = sum(bool(d["moments"]) for d in cases)
    lines = [
        (f"{M['right'].capitalize()} by v2 (15 fps, shipped)", f"{sum(right(d['flagged']) for d in cases)} / {n}"),
        *([] if M["v2_only"] else [
            (f"{M['right'].capitalize()} by v1", f"{sum(right(bool(d['v1_flagged'])) for d in cases)} / {n}"),
            (f"{M['right'].capitalize()} by v2 @ 25 fps (previous)",
             f"{sum(right(bool(d['f25_flagged'])) for d in cases)} / {n}")]),
        (M["moments"].capitalize(), f"{sum(len(d['moments']) for d in cases)} in {with_moments} of {n} "
                                    f"candidates — each one is listed in the '{M['timeline']}' tab"),
        ("Verdict rule", cases[0]["result"]["verdict_rule"].replace(">=", "≥")
         + f" (exact threshold {cases[0]['thr']:.4f})"),
        ("Scoring", f"out-of-fold — each video scored by a model that never trained on it "
                    f"({index['experiment']})"),
    ]
    if M["truth"] == "cheating":
        fp = [c for c in index["candidates"] if c["actual"] == "clean" and c["verdict"] == "CHEATING"]
        n_clean = sum(c["actual"] == "clean" for c in index["candidates"])
        lines.append(("Clean videos wrongly flagged (not listed here)",
                      f"{len(fp)} of {n_clean}" + (": " + ", ".join(
                          f"{c['name']} (ir{c['ir']}, score {c['score']:.3f})" for c in fp) if fp else "")))
    else:
        ch = [c for c in index["candidates"] if c["actual"] == "cheating"]
        lines.append(("Cheating candidates (separate report)",
                      f"{sum(c['verdict'] == 'CHEATING' for c in ch)} of {len(ch)} caught — "
                      f"see {MODES['cheating']['out']}"))
    for k, (a, b) in enumerate(lines, last + 2):
        ws.cell(k, 2, a).font = Font(bold=True)
        ws.cell(k, 4, b)


# --------------------------------------------------------------------------- tab 2

TIMELINE_COLS = [
    ("IR ID", 11, None), ("Candidate", 22, None), ("Video result", 10, None), ("Moment #", 8, None),
    ("Time range", 12, None), ("Start", 7, None), ("End", 7, None), ("Duration (s)", 8, "0"),
    ("Strongest at", 9, None), ("Confidence", 10, None), ("Avg reading prob", 9, "0.000"),
    ("Peak prob", 8, "0.000"), ("Type", 19, None), ("Speaking", 9, "0%"),
    ("What the model saw", 70, None), *measure_cols(10),
    ("Reviewer check", 28, None), ("Reviewer note", 50, None), ("Detail tab", 27, None),
]


def write_timeline(ws, cases: list[dict], tabs: list[str], art: dict, M: dict, has_review: bool) -> None:
    cols = TIMELINE_COLS if has_review else drop(TIMELINE_COLS, REVIEW_ONLY)
    header(ws, 1, [h for h, _, _ in cols])
    widths(ws, [w for _, w, _ in cols])
    ws.row_dimensions[1].height = 48
    put(ws, 2, {"What the model saw": NORMAL_TEXT, **normal_vals(art)}, cols)
    for j in range(1, len(cols) + 1):
        ws.cell(2, j).font = REF_FONT
    x = 3
    for d, tab in zip(cases, tabs):
        m = d["meta"]
        base = {"IR ID": f"ir{m['ir']}", "Candidate": m["name"]}
        # flagged moments plus the reviewer spans the model missed, in time order
        rows = [("moment", mo) for mo in d["moments"]] + \
               [("missed", s) for s in d["spans"] if not s["review"]["caught"]]
        rows.sort(key=lambda t: t[1]["a"])
        if not rows and d["strongest"]:
            rows = [("none", d["strongest"])]
        for kind, mo in rows:
            v = dict(base, **{
                "Time range": span(mo["a"], mo["b"]), "Start": mmss(mo["a"]), "End": mmss(mo["b"]),
                "Duration (s)": mo["b"] - mo["a"], "Strongest at": mmss(mo["peak_at"]),
                "Avg reading prob": mo["avg"], "Peak prob": mo["peak"], "Speaking": mo["speaking"],
                "What the model saw": mo["saw"], **measure_vals(mo["m"])})
            if kind == "moment":
                v.update({"Moment #": f"{mo['k']} of {len(d['moments'])}", "Confidence": mo["conf"],
                          "Type": mo["kind"], "Reviewer check": mo["check"], "Reviewer note": mo["note"]})
            elif kind == "missed":
                v.update({"Moment #": "—", "Type": "not flagged",
                          "Reviewer check": f"MISSED — reviewer marked it, model flagged "
                                            f"{mo['review']['coverage']:.0%} of it",
                          "Reviewer note": mo["review"]["reason"]})
            else:
                v.update({"Moment #": "—", "Type": "nothing flagged",
                          "What the model saw": "Strongest moment in the video (below the reading "
                                                "threshold): " + mo["saw"],
                          "Reviewer check": "not reviewed" if not d["review"] else ""})
            put(ws, x, v, cols)
            result_cell(ws.cell(x, col(cols, "Video result")), d["flagged"], M)
            link(ws.cell(x, col(cols, "Detail tab")), tab, tab)
            c = ws.cell(x, col(cols, "Confidence"))
            if c.value in CONF_FILL:
                c.fill = CONF_FILL[c.value]
            if has_review:
                chk = ws.cell(x, col(cols, "Reviewer check"))
                if str(chk.value).startswith("Confirmed"):
                    chk.fill = GOOD_FILL
                elif str(chk.value).startswith("Extra"):
                    chk.fill = WARN_FILL
                elif str(chk.value).startswith("MISSED"):
                    chk.fill, chk.font = BAD_FILL, BAD_FONT
            x += 1
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{x - 1}"
    ws.freeze_panes = "F3"


# --------------------------------------------------------------------------- one tab per id

def kv(ws, row: int, label: str, value, fmt: str | None = None) -> int:
    ws.cell(row, 1, label).font = Font(bold=True)
    c = ws.cell(row, 2, value)
    c.alignment = Alignment(horizontal="left")
    if fmt:
        c.number_format = fmt
    return row + 1


def section(ws, row: int, title: str, sub: str = "") -> int:
    ws.cell(row, 1, title).font = SECTION_FONT
    if sub:
        ws.cell(row + 1, 1, sub).font = REF_FONT
        return row + 2
    return row + 1


# shared column layout of the moment tables in a candidate tab (A..T); "Check" is
# renamed per table (reviewer check / caught? / flagged as reading?)
MOMENT_COLS = [
    ("Moment", 30, None), ("Start", 8, None), ("End", 8, None), ("Duration (s)", 9, "0"),
    ("Strongest at", 10, None), ("Confidence", 11, None), ("Avg reading prob", 9, "0.000"),
    ("Peak prob", 9, "0.000"), ("Type", 22, None), ("Speaking", 9, "0%"),
    ("What the model saw", 62, None), *measure_cols(11), ("Check", 28, None),
    ("Reviewer note", 55, None),
]


def moment_row(mo: dict) -> dict:
    return {"Start": mmss(mo["a"]), "End": mmss(mo["b"]), "Duration (s)": mo["b"] - mo["a"],
            "Strongest at": mmss(mo["peak_at"]), "Confidence": mo["conf"], "Avg reading prob": mo["avg"],
            "Peak prob": mo["peak"], "Type": mo["kind"], "Speaking": mo["speaking"],
            "What the model saw": mo["saw"], **measure_vals(mo["m"])}


def moment_table(ws, x: int, rows: list[dict], art: dict, rename: dict, without: set = frozenset()) -> int:
    """Header (renamed per table), the 'normal' reference row, then `rows`."""
    cols = drop(MOMENT_COLS, set(without))
    header(ws, x, [rename.get(h, h) for h, _, _ in cols])
    ws.row_dimensions[x].height = 48
    put(ws, x + 1, {"What the model saw": NORMAL_TEXT, **normal_vals(art)}, cols)
    for j in range(1, len(cols) + 1):
        ws.cell(x + 1, j).font = REF_FONT
    for i, v in enumerate(rows, x + 2):
        put(ws, i, v, cols)
        c = ws.cell(i, col(cols, "Confidence"))
        if c.value in CONF_FILL:
            c.fill = CONF_FILL[c.value]
        if "Check" in rename:
            chk = ws.cell(i, col(cols, "Check"))
            if str(chk.value).startswith(("Confirmed", "Yes")):
                chk.fill = GOOD_FILL
            elif str(chk.value).startswith("Extra"):
                chk.fill = WARN_FILL
            elif str(chk.value).startswith(("MISSED", "No")):
                chk.fill, chk.font = BAD_FILL, BAD_FONT
    return x + 2 + len(rows)


def write_candidate(ws, d: dict, art: dict, M: dict) -> None:
    m, r, rv, st = d["meta"], d["result"], d["review"], d["stats"]
    met = (rv or {}).get("metrics", {})
    widths(ws, [w for _, w, _ in MOMENT_COLS])
    ws.cell(1, 1, f"ir{m['ir']} — {m['name']}").font = TITLE_FONT
    link(ws.cell(2, 1), M["summary"], f"← back to {M['summary']}")
    link(ws.cell(2, 4), M["timeline"], f"→ all candidates' {M['timeline'].lower()}")

    x = section(ws, 4, "Summary")
    x = kv(ws, x, "IR ID", f"ir{m['ir']}")
    x = kv(ws, x, "Candidate", m["name"])
    x = kv(ws, x, "Video file", m["file"])
    x = kv(ws, x, "Folder (ground truth)", f"{m['folder']} ({M['truth']})")
    x = kv(ws, x, "Manually reviewed", f"Yes — {rv['source']}" if rv else "No")
    ws.cell(x, 1, "v2 result").font = Font(bold=True)
    result_cell(ws.cell(x, 2), d["flagged"], M)
    x += 1
    x = kv(ws, x, "v2 score (avg reading probability while speaking)", d["score"], "0.000")
    x = kv(ws, x, "Threshold", d["thr"], "0.000")
    x = kv(ws, x, "Score − threshold", d["score"] - d["thr"], "+0.000;-0.000;0.000")
    x = kv(ws, x, "Verdict rule", r["verdict_rule"].replace(">=", "≥"))
    x = kv(ws, x, "Video length", mmss(st["video_seconds"]))
    x = kv(ws, x, "Speaking time", mmss(r["speaking_seconds"]))
    x = kv(ws, x, "Reading detected", f"{mmss(r['reading_seconds'])} ({r['reading_seconds']:.0f} s)")
    x = kv(ws, x, "Reading time ÷ speaking time", r["reading_share_of_speech"], "0%")
    conf = Counter(mo["conf"] for mo in d["moments"])
    x = kv(ws, x, M["moments"].capitalize(), f"{len(d['moments'])} (high {conf['High']}, medium "
                                             f"{conf['Medium']}, low {conf['Low']} confidence)"
                                             if d["moments"] else "0")
    x = kv(ws, x, "First moment at", mmss(d["moments"][0]["a"]) if d["moments"] else "—")
    L, S = d["longest"], d["strongest"]
    x = kv(ws, x, "Longest moment", f"{span(L['a'], L['b'])} ({L['b'] - L['a']} s)" if L else "—")
    x = kv(ws, x, "Strongest second", f"{mmss(S['peak_at'])} (reading probability {S['peak']:.2f})"
                                      if S else "—")
    x = kv(ws, x, "Main reading signs", d["signs"] or "—")
    x = kv(ws, x, "Frames processed", st["n_frames"], "#,##0")
    x = kv(ws, x, "Processed fps", st["sampled_fps"], "0.0")
    x = kv(ws, x, "Face found", st["face_ratio"], "0.0%")
    if d["v1_score"] is not None:
        x = kv(ws, x, "v1 score", d["v1_score"], "0.000")
        ws.cell(x, 1, "v1 result").font = Font(bold=True)
        result_cell(ws.cell(x, 2), d["v1_flagged"], M)
        x += 1
        x = kv(ws, x, "v2 @ 25 fps score (previous)", d["f25_score"], "0.000")
        ws.cell(x, 1, "v2 @ 25 fps result").font = Font(bold=True)
        result_cell(ws.cell(x, 2), d["f25_flagged"], M)
        x += 1
    if rv:
        x = kv(ws, x, "Reviewer reading time", f"{mmss(met['reading_s'])} ({met['reading_s']} s)")
        x = kv(ws, x, "Reviewer spans caught", met["spans_caught"].replace("/", " / "))
        x = kv(ws, x, "Reviewer reading seconds flagged (recall)", met["recall"], "0%")
        x = kv(ws, x, "Flagged seconds the reviewer marked (precision)", met["precision"], "0%")
        x = kv(ws, x, "Overlap (IoU)", met["iou"], "0.00")
        x = kv(ws, x, "Per-second AUC vs review", met["auc"], "0.000")
    x = kv(ws, x, "Notes", d["notes"] or "—")
    x = kv(ws, x, "Scoring", f"out-of-fold ({art['experiment']}): this video was not in the training data "
                             f"of the model that scored it")

    # every flagged moment
    x = section(ws, x + 1, f"{M['moment_title']} ({len(d['moments'])} moments)",
                f"A second is flagged when its reading probability ≥ {art['second_threshold']}; gaps under "
                f"{art['merge_gap_seconds']:g} s are merged and moments under {art['min_segment_seconds']:g} s "
                f"dropped. Times are on the video clock (m:ss). Measurements are averages over the moment.")
    rows = [{**moment_row(mo), "Moment": f"#{mo['k']}  {span(mo['a'], mo['b'])}",
             "Check": mo["check"], "Reviewer note": mo["note"]} for mo in d["moments"]]
    x = moment_table(ws, x, rows, art, {"Check": "Reviewer check"} if rv else {},
                     set() if rv else {"Check", "Reviewer note"})
    if not d["moments"]:
        ws.cell(x, 1, "No moment flagged — see the strongest moments below.").font = Font(italic=True)
        x += 1

    # manual review vs model
    if rv:
        x = section(ws, x + 1, f"Manual review vs model ({met['spans_caught'].replace('/', ' / ')} caught)",
                    "Each span the reviewer marked as reading. Caught = at least half of its seconds flagged; "
                    "probabilities and measurements are over the reviewer's span.")
        rows = []
        for s in d["spans"]:
            rows.append({**moment_row(s), "Moment": span(s["a"], s["b"]),
                         "Confidence": f"{s['review']['coverage']:.0%} flagged",
                         "Type": ("model moment " + ", ".join(f"#{k}" for k in s["overlaps"]))
                         if s["overlaps"] else "no model moment",
                         "Check": "Yes — caught" if s["review"]["caught"] else "No — missed",
                         "Reviewer note": s["review"]["reason"]})
        x = moment_table(ws, x, rows, art, {"Moment": "Reviewer span", "Confidence": "Share flagged",
                                            "Type": "Model moment(s)", "Check": "Caught?"})

    # strongest moments, flagged or not
    x = section(ws, x + 1, f"Strongest {len(d['top'])} moments",
                f"The {TOP_K} highest-probability seconds at least {TOP_GAP_S} s apart, with ±{TOP_HALF_S} s "
                f"around each — flagged or not. This is where the model was most suspicious.")
    rows = [{**moment_row(mo), "Moment": f"top {k}  {span(mo['a'], mo['b'])}", "Check": mo["flag"]}
            for k, mo in enumerate(d["top"], 1)]
    x = moment_table(ws, x, rows, art, {"Check": "Flagged as reading?"}, {"Reviewer note"})

    # minute by minute
    n = len(d["prob"])
    x = section(ws, x + 1, "Minute by minute")
    header(ws, x, ["Minute", "Speaking (s)", "Reading flagged (s)", "Avg reading prob", "Peak prob",
                   "Moments"] + (["Reviewer reading (s)"] if rv else []))
    for k, a in enumerate(range(0, n, 60), x + 1):
        b = min(a + 60, n)
        p = d["prob"][a:b]
        ok = np.isfinite(p)
        ks = sorted(set(d["seg_of"][a:b]) - {0})
        vals = [span(a, b), round(float(d["speak"][a:b].sum()), 1), int((d["seg_of"][a:b] > 0).sum()),
                round(float(p[ok].mean()), 3) if ok.any() else None,
                round(float(p[ok].max()), 3) if ok.any() else None, ", ".join(f"#{i}" for i in ks)]
        if rv:
            vals.append(int(d["marked"][a:b].sum()))
        for j, v in enumerate(vals, 1):
            ws.cell(k, j, v)
        ws.cell(k, 2).number_format = "0.0"
        ws.cell(k, 4).number_format = ws.cell(k, 5).number_format = "0.000"
        if vals[2]:
            ws.cell(k, 3).fill = READ_FILL
    x += (n + 59) // 60 + 1

    # second by second
    x = section(ws, x + 1, "Second by second",
                "Shaded rows fall inside a flagged moment. Speaking = share of the second the mouth was "
                "moving as in speech (0–1).")
    header(ws, x, ["Time", "Reading prob", "Moment", "Speaking"] + (["Reviewer marked reading"] if rv else []))
    for i in range(n):
        k = x + 1 + i
        vals = [mmss(i), round(float(d["prob"][i]), 3) if np.isfinite(d["prob"][i]) else None,
                f"#{d['seg_of'][i]}" if d["seg_of"][i] else "", round(float(d["speak"][i]), 2)]
        if rv:
            vals.append("Yes" if d["marked"][i] else "")
        for j, v in enumerate(vals, 1):
            c = ws.cell(k, j, v)
            if d["seg_of"][i]:
                c.fill = READ_FILL
        ws.cell(k, 2).number_format = "0.000"


# --------------------------------------------------------------------------- guide tab

GUIDE_W = [30, 92, 36]          # term | what it means | example value (or normal value)

# each eye / blink / head measurement in plain words; direction, label and normal value come from model.json
MEASURE_HELP = {
    "blink_rate": "Blinks per minute. People blink much less while their eyes are fixed on text.",
    "head_speed": "How fast the head turns (side-to-side plus up-down), in degrees per second. Someone reading "
                  "keeps the head still so the text stays in view.",
    "h_prog_frac": "Share of the time the eyes drift slowly in one direction across the screen, the way they do "
                   "when following a line of text.",
    "big_jump_rate": "Large, quick eye movements per minute: glancing around, looking up to think. A reader's eyes "
                     "stay on the text, so there are fewer of them.",
    "h_std": "How widely the eyes move left and right (eye-position units; bigger = wider). Eyes fixed on one "
             "part of the screen give a small spread.",
    "v_mean_listen": "How high the eyes are held compared with the same candidate while listening to the "
                     "question (+ = higher). Text in another part of the screen pulls the eyes to a different "
                     "height than when just listening.",
    "brow_down_mean": "How far the brows are pulled down, from 0 (not at all) to 1 (fully): a sign of "
                      "concentrating.",
}


def summary_help(M: dict, thr: float) -> dict:
    """Plain-language meaning of every tab-1 column (keyed by the un-renamed header)."""
    cheat = M["truth"] == "cheating"
    return {
        "#": "Row number. Rows are sorted from the highest v2 score (most like reading) to the lowest.",
        "IR ID": "The candidate's interview ID in the hiring system — use it to look the candidate up anywhere else.",
        "Candidate name": "The candidate's name, as in the video file name.",
        "Detail tab": "Click it to open this candidate's own tab with every detail.",
        "Folder": "Where the video came from. This is the known truth the software is graded against: "
                  "cheating_folder = confirmed cheater; not_cheating_folder / new_not_cheating = confirmed honest.",
        "Manually reviewed": "Yes when a person watched the video and wrote down the exact times the candidate was "
                             "reading. Only these candidates have the reviewer columns filled in.",
        "v2 verdict": "What the software decided for the whole video: CHEATING or NOT CHEATING.",
        "v2 result": ("Whether that decision was right. CAUGHT (green) = known cheater correctly marked CHEATING. "
                      "MISSED (red) = known cheater the software let through." if cheat else
                      "Whether that decision was right. CLEARED (green) = honest candidate correctly marked NOT "
                      "CHEATING. FLAGGED (red) = honest candidate wrongly marked CHEATING (a false alarm)."),
        "v2 score": "The number the verdict is based on: the average reading probability over the seconds in which "
                    "the candidate was talking. 0 = never looked like reading, 1 = looked like reading the whole "
                    "time. Yellow = borderline (within 5% of the threshold).",
        "Threshold": f"The bar the v2 score must reach to be called CHEATING ({thr:.4f}, about {thr:.2f}). "
                     f"The same for every candidate.",
        "Score − threshold": "How far above (+) or below (−) the bar the candidate is. A large + is a clear case; "
                             "a number close to 0 either way is a close call.",
        "Video length": "Length of the recording (minutes:seconds).",
        "Speaking time": "How long the candidate was talking, worked out from mouth movement (the sound is not used).",
        "Reading detected": "Total length of all the moments the software flagged as reading (minutes:seconds).",
        "Reading (seconds)": "The same total, in seconds.",
        "Reading time ÷ speaking time": "Reading detected divided by speaking time. 90% means that for almost all "
                                        "the time the candidate was talking, the software thought they were "
                                        "reading. (It can go over 100% if they also read while silent.)",
        "Moments": "How many separate moments were flagged as reading.",
        "Moments at": "Start–end of every flagged moment on the video clock (m:ss from the start of the "
                      "recording). Jump to these times in the video to see it yourself.",
        "First moment at": "When the first flagged moment starts.",
        "Longest moment": "The longest single moment, with its length in seconds.",
        "Strongest moment": "The single most reading-like second in the whole video, with its reading probability. "
                            "The best place to look first.",
        "High-confidence moments": "How many moments are High confidence (average reading probability 0.60 or more).",
        "Main reading signs": "The three behaviours that stood out most across this candidate's moments (weighted "
                              "by how long they lasted). Each one is explained in 'The 7 behaviour measurements' "
                              "below.",
        "Reviewer spans caught": "Reviewed candidates only. Of the stretches the reviewer marked as reading, how "
                                 "many the software caught (at least half of the stretch flagged). '28 / 29' = 28 of "
                                 "29 caught.",
        "Reviewer reading seconds flagged (recall)": "Reviewed candidates only. Of all the seconds the reviewer "
                                                     "marked as reading, the share the software also flagged. High "
                                                     "= it found nearly all the reading the reviewer saw.",
        "Flagged seconds the reviewer marked (precision)": "Reviewed candidates only. Of all the seconds the software "
                                                           "flagged, the share the reviewer had also marked. Low is "
                                                           "not always wrong: reviewers did not mark every reading "
                                                           "stretch (see 'Extra' under Reviewer check).",
        "v1 score": "Score from the older v1 model, shown for comparison. v1 has its own scale and bar, so "
                    "compare its result, not its number, with v2.",
        "v1 result": "Whether v1 got this candidate right (same words and colours as the v2 result).",
        "v2 @ 25 fps score": "Score from the previous v2 model, which analysed 25 frames per second instead of 15. "
                             "Shown for comparison; it had its own bar.",
        "v2 @ 25 fps result": "Whether that previous model got this candidate right.",
        "Processed fps": "Frames per second actually analysed. Normally 15; lower means the recording itself had "
                         "fewer frames (then every frame is used).",
        "Face found": "Share of the frames in which a face was found. Below 90% the face was often missing "
                      "(turned away, out of frame, poor light) and the numbers are less reliable.",
        "Frames processed": "How many video frames were analysed.",
        "Video file": "The recording's file name.",
        "Notes": "Automatic remarks: borderline score, which model caught or missed it (v1 vs v2), why a "
                 f"{M['wrong']} case went that way, recording problems, manual review.",
    }


def moment_help(M: dict, has_review: bool) -> dict:
    """Plain-language meaning of the moment columns (tab 2 and the candidate tabs)."""
    return {
        "IR ID": "Interview ID of the candidate the moment belongs to.",
        "Candidate": "Name of the candidate the moment belongs to.",
        "Video result": f"That candidate's overall result ({M['right']} / {M['wrong']}), repeated on every row "
                        f"so you can filter by it.",
        "Moment #": "'3 of 12' = the 3rd of this candidate's 12 flagged moments, in time order. '—' = not a "
                    "flagged moment: " + ("a stretch the reviewer marked that the software missed, or " if has_review
                                          else "") +
                    "for a candidate with no moments, their single most reading-like stretch.",
        "Time range": "When the moment happens, on the video clock (m:ss from the start of the recording).",
        "Start": "Moment start (m:ss).",
        "End": "Moment end (m:ss).",
        "Duration (s)": "Length of the moment in seconds.",
        "Strongest at": "The most reading-like second inside the moment — the best second to watch.",
        "Confidence": "How sure the software is, from the moment's average reading probability: High ≥ 0.60 "
                      "(dark orange), Medium 0.35–0.60 (light orange), Low below 0.35 (pale yellow). Low moments "
                      "are weak evidence on their own.",
        "Avg reading prob": "Average reading probability over the moment's seconds (0–1).",
        "Peak prob": "Highest reading probability of any second in the moment (0–1).",
        "Type": "'reading while answering' = the candidate was talking for at least half of the moment (reading "
                "the answer out). 'reading on screen' = mostly silent while reading. 'not flagged' / 'nothing "
                "flagged' = see Moment # '—'.",
        "Speaking": "Share of the moment during which the candidate was talking.",
        "What the model saw": "A plain sentence: what the candidate was doing and which behaviours stood out, each "
                              "measured value shown next to the normal value for an honest candidate talking.",
        "Reviewer check": "Reviewed candidates only. Confirmed (green) = the reviewer also marked this time as "
                          "reading. Extra (yellow) = flagged by the software only — a false alarm or reading the "
                          "reviewer did not note. MISSED (red) = the reviewer marked reading here but the software "
                          "flagged less than half of it. 'not reviewed' = this candidate has no manual review.",
        "Reviewer note": "The reviewer's own words for that stretch.",
        "Detail tab": "Click to open the candidate's tab.",
    }


def _lines(v, width: float) -> int:
    """Rough number of wrapped lines `v` needs in a column `width` wide."""
    if v is None:
        return 1
    per = max(1, int(width * 1.1))
    return max(1, sum(max(1, -(-len(p) // per)) for p in str(v).split("\n")))


def g_row(ws, x: int, term, text, extra=None, fmt: str | None = None) -> int:
    """One guide row: bold term | explanation | optional example / value."""
    for j, v in enumerate((term, text, extra), 1):
        if v is not None:
            ws.cell(x, j, v).alignment = TOP
    ws.cell(x, 1).font = Font(bold=True)
    if fmt:
        ws.cell(x, 3).number_format = fmt
    ws.row_dimensions[x].height = 15 * max(_lines(v, w) for v, w in zip((term, text, extra), GUIDE_W))
    return x + 1


def g_example(ws, x: int, term: str, text: str, src) -> int:
    """A guide row whose third cell copies a real cell of this workbook (value, format, colour)."""
    v = src.value if src is not None else None
    if isinstance(v, str) and len(v) > 200:
        v = v[:199] + "…"
    x = g_row(ws, x, term, text, v, src.number_format if src is not None else None)
    if src is not None and src.fill.fill_type:
        ws.cell(x - 1, 3).fill, ws.cell(x - 1, 3).font = copy(src.fill), copy(src.font)
    return x


def by_header(ws, row: int) -> dict:
    return {ws.cell(1, j).value: ws.cell(row, j) for j in range(1, ws.max_column + 1)}


def example(d: dict, row: int, M: dict) -> str:
    """One candidate's tab-1 row read out in plain sentences."""
    m, r = d["meta"], d["result"]
    right = d["flagged"] == (M["truth"] == "cheating")
    gap = d["score"] - d["thr"]
    out = (f"{m['name']} (ir{m['ir']}, # {row} in '{M['summary']}'). The video is "
           f"{mmss(d['stats']['video_seconds'])} long and the candidate was talking for "
           f"{mmss(r['speaking_seconds'])}. While talking, the average reading probability (the v2 score) was "
           f"{d['score']:.3f}, which is {abs(gap):.3f} {'above' if gap >= 0 else 'below'} the bar of "
           f"{d['thr']:.3f}, so v2 says {'CHEATING' if d['flagged'] else 'NOT CHEATING'}. The candidate is "
           f"{'a known cheater' if M['truth'] == 'cheating' else 'known to be honest'}, so the result is "
           f"{M['right'] if right else M['wrong']} ({'green' if right else 'red'}).")
    if not d["moments"]:
        return out + (" No stretch was reading-like enough to be flagged; the 'Strongest 5 moments' in the "
                      "candidate's tab show where the software came closest.")
    L, S = d["longest"], d["strongest"]
    out += (f" The software flagged {len(d['moments'])} {M['moments']} adding up to "
            f"{mmss(r['reading_seconds'])} ({r['reading_share_of_speech']:.0%} of the talking time). The first "
            f"starts at {mmss(d['moments'][0]['a'])}, the longest is {span(L['a'], L['b'])} ({L['b'] - L['a']} s) "
            f"and the most reading-like second is {mmss(S['peak_at'])} (probability {S['peak']:.2f}). Signs that "
            f"stood out most: {d['signs']}.")
    if not d["flagged"]:
        out += " These moments were too short or too few to lift the average over the bar."
    return out + f" To check it yourself, open {m['file']} and jump to {mmss(S['peak_at'])}."


def write_guide(ws, index: dict, art: dict, cases: list[dict], tabs: list[str], M: dict, has_review: bool,
                summary, timeline) -> None:
    cheat, n, thr = M["truth"] == "cheating", len(cases), cases[0]["thr"]
    n_right = sum(d["flagged"] == cheat for d in cases)
    clean_all = [c for c in index["candidates"] if c["actual"] == "clean"]
    cheat_all = [c for c in index["candidates"] if c["actual"] == "cheating"]
    n_fp = sum(c["verdict"] == "CHEATING" for c in clean_all)
    n_caught = sum(c["verdict"] == "CHEATING" for c in cheat_all)
    sec_thr, gap, min_len = art["second_threshold"], art["merge_gap_seconds"], art["min_segment_seconds"]
    rule = cases[0]["result"]["verdict_rule"].replace(">=", "≥")

    # example candidate for the column tables: a reviewed one (every column filled), else the first
    # correctly-judged one with moments; plus the first wrongly-judged one for a second worked example
    ex = next((i for i, d in enumerate(cases) if d["review"]), None)
    if ex is None:
        ex = next((i for i, d in enumerate(cases) if d["flagged"] == cheat and d["moments"]), 0)
    wrong = next((i for i, d in enumerate(cases) if d["flagged"] != cheat), None)
    d_ex = cases[ex]
    ir_ex = f"ir{d_ex['meta']['ir']}"
    t_rows = [r for r in range(3, timeline.max_row + 1) if timeline.cell(r, 1).value == ir_ex]
    t_ex = next((r for r in t_rows if by_header(timeline, r).get("Confidence").value == "High"),
                t_rows[0] if t_rows else None)

    ws.sheet_view.showGridLines = False
    widths(ws, GUIDE_W)
    ws.cell(1, 1, "How to read this report").font = TITLE_FONT
    ws.cell(2, 1, "Start here if you are new to the project. Every tab, column, number and colour in this "
                  "workbook is explained below.").font = REF_FONT
    link(ws.cell(3, 1), M["summary"], f"→ go to {M['summary']}")

    x = section(ws, 5, "In one minute")
    for a, b in [
        ("What is checked", "Candidates record a video interview. Some secretly read their answers off the "
                            "screen (notes, a chat bot, a web page) instead of answering from their own "
                            "knowledge. That reading is the 'cheating' this report is about."),
        ("How", "A piece of software — the 'v2 reading detector' — watches only the candidate's face in the "
                "recording: eyes, blinks, head and mouth. It does not use the sound and cannot see the "
                "candidate's screen."),
        ("What it gives", "For every second of the video, a reading probability from 0 to 1 (0 = clearly not "
                          "reading, 1 = clearly reading). Runs of reading-like seconds become 'moments' with a "
                          "start and end time you can look up in the video."),
        ("Verdict", f"It averages the probability over the seconds in which the candidate is talking. If that "
                    f"average — the 'v2 score' — is {thr:.4f} (about {thr:.2f}) or more, the video is marked "
                    f"CHEATING, otherwise NOT CHEATING."),
        ("Who is in this file",
         f"The {n} candidates people had already confirmed as cheaters (videos in cheating_folder). Because the "
         f"true answer is known, every row shows whether the software CAUGHT or MISSED that person." if cheat else
         f"The {n} candidates known to be honest (videos in not_cheating_folder and new_not_cheating). Every row "
         f"shows whether the software correctly CLEARED them or wrongly FLAGGED them."),
        ("Headline",
         f"v2 caught {n_right} of {n} known cheaters. On the {len(clean_all)} honest candidates (not in this "
         f"file) it raised {n_fp} false alarm{'' if n_fp == 1 else 's'}." if cheat else
         f"v2 cleared {n_right} of {n} honest candidates and wrongly flagged {n - n_right}. It caught {n_caught} "
         f"of {len(cheat_all)} known cheaters (see {MODES['cheating']['out']})."),
    ]:
        x = g_row(ws, x, a, b)

    x = section(ws, x + 1, "How the software reaches a verdict")
    for a, b in [
        ("1. Watch the face", f"The video is analysed at {art['target_fps']:g} frames per second. On every frame a "
                              f"face-tracking model (MediaPipe) places several hundred points on the face and "
                              f"measures where the eyes look, blinks, head turning, brows and mouth."),
        ("2. Detect talking", "Mouth opening and jaw movement show when the candidate is talking (from the "
                              "picture, not the sound). 'Speaking time' comes from this."),
        ("3. Score every second", f"For every second, the {art['read_window_seconds']:g} seconds around it are "
                                  f"summarised into {len(art['features'])} numbers (blink rate, head stillness, "
                                  f"eye movement, talking …) and a machine-learning model (XGBoost) turns them into "
                                  f"the reading probability for that second."),
        ("4. Mark reading seconds", f"A second counts as reading when its probability is {sec_thr} or more."),
        ("5. Build moments", f"Neighbouring reading seconds are joined into one moment: gaps shorter than {gap:g} s "
                             f"are bridged, and moments shorter than {min_len:g} s are dropped as noise."),
        ("6. Decide the video", f"{rule} (exact bar {thr:.4f}). The bar was tuned to avoid accusing honest people "
                                f"— a false accusation counts as worse than a miss — so some cheaters who read only "
                                f"briefly fall below it."),
        ("Note", f"Moments (step 5) and the verdict (step 6) use different rules. A candidate can have a few short "
                 f"moments and still be NOT CHEATING when those moments are too little to lift the average over "
                 f"the bar."),
    ]:
        x = g_row(ws, x, a, b)

    x = section(ws, x + 1, "Tabs in this workbook")
    moment_rows = timeline.max_row - 2
    for label, tab, b in [
        (GUIDE_TAB, None, "This guide."),
        (M["summary"], M["summary"],
         f"Start here. One row per candidate ({n}), from the highest v2 score to the lowest: the verdict, whether "
         f"it was right, how much reading was found and when. Click a name in 'Detail tab' to open that "
         f"candidate. The header row has filters."),
        (M["timeline"], M["timeline"],
         f"Every flagged moment of every candidate in one list ({moment_rows} rows): when, how long, how sure the "
         f"software was and what the eyes, blinks and head were doing. Filter 'Confidence' to High to see only "
         f"the strongest."
         + (" Stretches the reviewer marked that the software missed are listed too (red)." if has_review else "")
         + " A candidate with no moment gets one row with their most reading-like stretch."),
        (f"{tabs[0]} … (one tab per candidate)", tabs[0],
         "One tab per candidate, named 'ir<ID> <name>', from the third tab to the end: summary, every moment, "
         + ("the manual review comparison (reviewed candidates only), " if has_review else "")
         + "the 5 strongest moments, minute by minute and second by second. See 'Inside each candidate tab' "
         "below."),
    ]:
        x = g_row(ws, x, label, b)
        if tab:
            link(ws.cell(x - 1, 1), tab, label)
            ws.cell(x - 1, 1).font = Font(bold=True, color="0563C1", underline="single")

    x = section(ws, x + 1, "Worked example — one row read out in plain words")
    x = g_row(ws, x, d_ex["meta"]["name"], example(d_ex, ex + 1, M))
    if wrong is not None and wrong != ex:
        x = g_row(ws, x, cases[wrong]["meta"]["name"], example(cases[wrong], wrong + 1, M))

    # tab 1 columns, each with the example candidate's real value
    x = section(ws, x + 1, f"Tab '{M['summary']}' — every column",
                f"The right-hand column copies {d_ex['meta']['name']}'s value from that tab.")
    header(ws, x, ["Column", "What it means", f"Example: {d_ex['meta']['name']}"])
    x += 1
    renamed = {M["moments"].capitalize(): "Moments", f"Reading {M['verb']} at (every moment)": "Moments at"}
    help1 = summary_help(M, thr)
    for h, c in by_header(summary, ex + 2).items():
        x = g_example(ws, x, h, help1.get(renamed.get(h, h), ""), c)

    # moment columns (tab 2 and candidate tabs); the 7 measurements get their own table below
    measured = {h for _, h, _, _ in MEASURES}
    x = section(ws, x + 1, f"Tab '{M['timeline']}' and the moment tables in each candidate tab — every column",
                "The right-hand column copies one of " + d_ex["meta"]["name"] + "'s moments. The candidate tabs "
                "call the first column 'Moment' (number and time range).")
    header(ws, x, ["Column", "What it means", f"Example: {d_ex['meta']['name']}"])
    x += 1
    help2 = moment_help(M, has_review)
    ex_cells = by_header(timeline, t_ex) if t_ex else {}
    for j in range(1, timeline.max_column + 1):
        h = timeline.cell(1, j).value
        if h not in measured:
            x = g_example(ws, x, h, help2.get(h, ""), ex_cells.get(h))

    x = section(ws, x + 1, "The 7 behaviour measurements",
                "Shown for every moment. 'Normal' is the typical (median) value of honest candidates while they "
                "talk; it is repeated in the grey italic row under every moment table's header.")
    header(ws, x, ["Measurement", "What it measures and which way points to reading",
                   "Normal (honest candidate talking)"])
    x += 1
    refs = {r["feature"]: r for r in art.get("reason_reference", [])}
    for f, h, fmt, _ in MEASURES:
        r = refs.get(f, {})
        how = MEASURE_HELP[f]
        if r:
            how += (f" Points to reading when {'LOWER' if r['direction'] < 0 else 'HIGHER'} than normal. Called "
                    f"'{r['label']}' in 'Main reading signs'.")
        x = g_row(ws, x, h, how, r.get("clean_median"), fmt)
    x = g_row(ws, x, "No single sign is proof", f"Honest people can also blink little or sit still. The model weighs "
                                                f"all {len(art['features'])} numbers together; these 7 are simply the "
                                                f"easiest for a person to check in the video.")

    x = section(ws, x + 1, "Inside each candidate tab")
    rows = [
        ("Summary", f"The candidate's numbers from '{M['summary']}' as a list, plus the extras below."),
        ("Folder (ground truth)", "Which folder the video is in — the known truth."),
        ("Strongest second", "Same as 'Strongest moment' in tab 1."),
    ]
    if has_review:
        rows += [
            ("Reviewer reading time", "Total time the reviewer marked as reading."),
            ("Overlap (IoU)", "How well the software's reading time and the reviewer's line up: time both marked ÷ "
                              "time either marked. 1 = identical, 0 = no overlap."),
            ("Per-second AUC vs review", "How well the per-second probability separates the reviewer's reading "
                                         "seconds from the rest: 0.5 = no better than a coin flip, 1 = perfect."),
        ]
    rows += [
        (M["moment_title"], f"Every flagged moment, same columns as '{M['timeline']}'."),
        *([("Manual review vs model", "Reviewed candidates only. One row per stretch the reviewer marked as "
                                      "reading. Share flagged = how much of it the software flagged; Caught? = Yes "
                                      "when at least half; Model moment(s) = which flagged moments overlap it; "
                                      "Reviewer note = the reviewer's words.")] if has_review else []),
        (f"Strongest {TOP_K} moments", f"The {TOP_K} most reading-like seconds (at least {TOP_GAP_S} s apart), "
                                       f"±{TOP_HALF_S} s around each, whether or not they were flagged. 'Flagged as "
                                       f"reading?' says whether each falls inside a moment. For a candidate with "
                                       f"few or no moments, this shows where the software came closest."),
        ("Minute by minute", "One row per minute: seconds talking, seconds flagged as reading (shaded when any), "
                             "average and peak reading probability, and which moments fall in that minute"
                             + (" (plus seconds the reviewer marked)" if has_review else "")
                             + ". Shows when in the interview the reading happened."),
        ("Second by second", "Every second of the video: its reading probability, the moment it belongs to "
                             "(shaded rows) and how much of that second the candidate was talking (0–1)"
                             + ("; 'Reviewer marked reading' = Yes where the reviewer marked it" if has_review else "")
                             + ". A blank probability = the second could not be scored (e.g. no face visible)."),
        ("Scoring", "'out-of-fold' — see the glossary below."),
    ]
    for a, b in rows:
        x = g_row(ws, x, a, b)

    x = section(ws, x + 1, "Colours")
    lo = {lab: f"{t:.2f}" for t, lab in CONFIDENCE}
    for text, fill, font, b in [
        ("Column header", HEAD_FILL, HEAD_FONT, "Column headers."),
        (M["right"], GOOD_FILL, GOOD_FONT,
         ("Right: a known cheater marked CHEATING." if cheat else "Right: an honest candidate marked NOT CHEATING.")
         + (" Under Reviewer check / Caught?: the reviewer agrees." if has_review else "")
         + " Under 'Flagged as reading?': inside a flagged moment."),
        (M["wrong"], BAD_FILL, BAD_FONT,
         ("Wrong: a known cheater marked NOT CHEATING." if cheat else
          "Wrong: an honest candidate marked CHEATING (false alarm).")
         + (" Under Reviewer check / Caught?: reading the reviewer marked that the software missed." if has_review
            else "")
         + " Under 'Flagged as reading?': below the reading threshold (not a mistake, just not flagged)."),
        ("Borderline", WARN_FILL, None, "On a v2 score: within 5% of the threshold, a close call."
         + (" Under Reviewer check: Extra — flagged by the software, not marked by the reviewer." if has_review
            else "")),
        ("High", CONF_FILL["High"], None, f"Confidence High: average reading probability ≥ {lo['High']}."),
        ("Medium", CONF_FILL["Medium"], None, f"Confidence Medium: {lo['Medium']} to {lo['High']}."),
        ("Low", CONF_FILL["Low"], None, f"Confidence Low: below {lo['Medium']}. Weak evidence on its own."),
        ("Shaded time", READ_FILL, None, "Minute by minute / Second by second: time inside a flagged moment."),
        ("Grey italic row", None, REF_FONT, "Under each moment table's header: the normal value of each "
                                            "measurement for an honest candidate talking, to compare against."),
        ("Blue underlined", None, LINK_FONT, "A link — click to jump to another tab."),
    ]:
        x = g_row(ws, x, text, b)
        c = ws.cell(x - 1, 1)
        c.font = font or Font(bold=True)
        if fill:
            c.fill = fill

    x = section(ws, x + 1, "Glossary")
    rows = [
        ("v2", "The current reading detector, the one this report is about ('SHIPPED' = the version in use)."),
        *([] if M["v2_only"] else [
            ("v1", "The original detector v2 replaced. v1 judged whole 20-second chunks of video, so it could say "
                   "whether someone cheated but not when; v2 scores every second."),
            ("v2 @ 25 fps", "The previous v2, which analysed 25 frames per second. v2 now uses 15, the rate used "
                            "in production.")]),
        ("fps", "Frames per second: how many still pictures per second of video are analysed."),
        ("Reading probability", "A number from 0 to 1 saying how much a second looks like reading. It is not a "
                                "percentage chance that the candidate cheated."),
        ("Threshold", f"A cut-off. Two are used: per second (probability ≥ {sec_thr} → reading second) and per "
                      f"video (v2 score ≥ {thr:.4f} → CHEATING)."),
        ("Ground truth", "What is known to be true independently of the software — here, which folder the video "
                         "is in (cheating or honest), decided by people."),
        *([("Manual review", "A person watched the video and wrote down when the candidate was reading. "
                             "Available for 3 cheating candidates; it is what the software's moments are checked "
                             "against."),
           ("Recall", "Of everything the reviewer marked as reading, the share the software also flagged (how "
                      "much it found)."),
           ("Precision", "Of everything the software flagged, the share the reviewer had also marked (how much of "
                         "what it flagged was confirmed).")] if has_review else
          [("Manual review", "None of these candidates has one (the 3 manual reviews are of cheating "
                             "candidates), so there are no reviewer columns.")]),
        ("Out-of-fold", "Each video was scored by a version of the model that had not learned from that video. "
                        "So these numbers are a fair preview of how it does on brand-new candidates, not a "
                        "memory test."),
        ("False alarm", "An honest candidate marked CHEATING, or a moment flagged when nobody was reading."),
        ("Borderline", "A v2 score within 5% of the threshold — a close call either way."),
    ]
    for a, b in rows:
        x = g_row(ws, x, a, b)

    x = section(ws, x + 1, "Before you act on a result")
    for a, b in [
        ("A flag is a pointer", "It means the face behaved the way a reading face does — not proof. Always watch "
                                "the video at the listed times before concluding anything."),
        ("Weigh the moments", "Long, High-confidence moments are strong evidence. Short (4–6 s) or Low-confidence "
                              "moments are weak on their own."),
        ("Check the recording", "Below 90% 'Face found', or 'Processed fps' well under 15, makes the numbers less "
                                "reliable."),
        (f"{M['wrong']} candidates",
         (f"The Notes column says whether some short moments were flagged anyway, and the 'Strongest {TOP_K} "
          f"moments' in their tab show where the software came closest." if cheat else
          f"Their moments show exactly when and why the software was suspicious; watch those times to decide "
          f"whether it was a false alarm or real reading.")),
        ("Why the bar is where it is", f"It protects honest candidates: only {n_fp} of the {len(clean_all)} "
                                       f"honest videos {'was' if n_fp == 1 else 'were'} wrongly flagged. The price "
                                       f"is that some cheaters fall below it."),
        ("Times", "All times are on the video clock: minutes:seconds from the start of the recording."),
    ]:
        x = g_row(ws, x, a, b)


def build(truth: str, v2_only: bool = False) -> None:
    M = dict(MODES[truth], v2_only=v2_only)
    if v2_only:
        M["out"] = M["out"].replace(".xlsx", "_v2_only.xlsx")
    index, art, cases = load(truth, v2_only)
    has_review = any(d["review"] for d in cases)
    wb = Workbook()
    ws = wb.active
    ws.title = M["summary"]
    used = {M["summary"], M["timeline"], GUIDE_TAB}
    tabs = [sheet_name(d["meta"]["ir"], d["meta"]["name"], used) for d in cases]
    write_summary(ws, index, cases, tabs, M, has_review)
    timeline = wb.create_sheet(M["timeline"])
    write_timeline(timeline, cases, tabs, art, M, has_review)
    for d, tab in zip(cases, tabs):
        write_candidate(wb.create_sheet(tab), d, art, M)
    # the guide is written last (it quotes the other tabs) but opens first
    guide = wb.create_sheet(GUIDE_TAB, 0)
    write_guide(guide, index, art, cases, tabs, M, has_review, ws, timeline)
    for sh in wb.worksheets:
        sh.sheet_view.tabSelected = sh is guide
    wb.active = guide
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / M["out"]
    wb.save(out)
    right = sum(d["flagged"] == (truth == "cheating") for d in cases)
    print(f"wrote {out} — {len(cases)} {truth} candidates, {right} {M['right'].lower()}, "
          f"{sum(len(d['moments']) for d in cases)} {M['moments']}, {len(wb.sheetnames)} tabs")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", choices=list(MODES), help="build one workbook (default: both)")
    ap.add_argument("--v2-only", action="store_true",
                    help="leave out the v1 and 25 fps comparison; writes *_v2_only.xlsx")
    args = ap.parse_args()
    for truth in ([args.only] if args.only else list(MODES)):
        build(truth, args.v2_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
