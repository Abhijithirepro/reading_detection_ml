"""
all_videos_results.py (v2) — the verdict for every on-disk recording, side by side
with the previous v2 model (archive/25fps) and v1.

Every v2 score is OUT-OF-FOLD: it comes from the cross-validation run that shipped
the model (results/oof_video_scores__<experiment>.csv), so no video is scored by a
model that trained on it. v1 columns are v1's own out-of-fold scores
(../models/cv_video_scores.csv). The sampling rate and face ratio of each video come
from data/features/_extract_summary.csv, so the table also shows what was processed.

Writes results/ALL_VIDEOS_RESULTS.md and results/ALL_VIDEOS_RESULTS.csv.

Usage (from the project root):
    venv/bin/python v2/src/all_videos_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from common import PROJECT_ROOT, V1_ROOT, MODELS_DIR, RESULTS_DIR, FEATURES_DIR

ARCHIVE_MODELS = PROJECT_ROOT / "archive" / "25fps" / "models"
FOLDERS = {("cheating", "interviews"): "cheating_folder", ("clean", "interviews"): "not_cheating_folder",
           ("clean", "interviews_new"): "new_not_cheating"}
BORDERLINE = 0.05            # within 5 % of the verdict threshold


def _mmss(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def _tick(ok: bool) -> str:
    return "✓" if ok else "✗"


def load_v2(models_dir: Path):
    """(model.json, out-of-fold video scores) for the model in `models_dir`."""
    art = json.loads((models_dir / "model.json").read_text())
    path = RESULTS_DIR / f"oof_video_scores__{art['experiment']}.csv"
    # round_trip: the default parser can move the last bit, and a score that sits exactly
    # on the threshold (it was tuned on these scores) would flip
    return art, pd.read_csv(path, float_precision="round_trip").set_index("video")


def build() -> pd.DataFrame:
    art, cur = load_v2(MODELS_DIR)
    thr = art["video_threshold"]
    prev_art, prev = load_v2(ARCHIVE_MODELS) if (ARCHIVE_MODELS / "model.json").exists() else (None, None)
    v1 = pd.read_csv(V1_ROOT / "models" / "cv_video_scores.csv", float_precision="round_trip").set_index("group")
    v1_thr = json.loads((V1_ROOT / "models" / "model.json").read_text())["threshold"]
    ex = pd.read_csv(FEATURES_DIR / "_extract_summary.csv")
    ex.index = ex["out"].map(lambda p: Path(p).stem)

    rows = []
    for video, r in cur.iterrows():
        label, source = video.split("__")[:2]
        actual = "cheating" if r["label"] == 1 else "clean"
        flag = r["score"] >= thr
        row = dict(
            name=video.split("__")[-1].replace("combined_", ""),
            folder=FOLDERS.get((label, source), source), actual=actual,
            v2_score=round(r["score"], 3), v2_verdict="CHEATING" if flag else "clean",
            v2_correct=flag == (actual == "cheating"),
            v2_borderline=abs(r["score"] - thr) <= BORDERLINE * thr,
            reading_time=_mmss(r["reading_seconds"]), reading_segments=int(r["reading_segments"]),
        )
        if prev is not None and video in prev.index:
            pflag = prev.loc[video, "score"] >= prev_art["video_threshold"]
            row.update(v2_25fps_score=round(prev.loc[video, "score"], 3),
                       v2_25fps_verdict="CHEATING" if pflag else "clean",
                       v2_25fps_correct=pflag == (actual == "cheating"))
        if video in v1.index:
            vflag = v1.loc[video, "score"] >= v1_thr
            row.update(v1_score=round(v1.loc[video, "score"], 3), v1_verdict="CHEATING" if vflag else "clean",
                       v1_correct=vflag == (actual == "cheating"))
        if video in ex.index:
            row.update(sampled_fps=round(float(ex.loc[video, "sampled_fps"]), 2),
                       face_ratio=round(float(ex.loc[video, "face_ratio"]), 3))
        row["reviewed"] = bool(r["reviewed"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs.update(art=art, prev_art=prev_art, v1_thr=v1_thr)
    return df.sort_values(["actual", "v2_score"], ascending=[True, False]).reset_index(drop=True)


def _counts(df: pd.DataFrame, col: str) -> tuple:
    ch, cl = df[df.actual == "cheating"], df[df.actual == "clean"]
    return int(ch[col].sum()), len(ch), int(cl[col].sum()), len(cl), int(df[col].sum()), len(df)


def markdown(df: pd.DataFrame) -> str:
    art, prev_art, v1_thr = df.attrs["art"], df.attrs["prev_art"], df.attrs["v1_thr"]
    fps = art.get("target_fps", 25.0)
    has_prev = "v2_25fps_score" in df
    n_folder = df.folder.value_counts()

    out = [f"# All-video results — v2 ({fps:g} fps) vs the previous v2 (25 fps) and v1", ""]
    out += [f"Every on-disk recording ({n_folder.get('cheating_folder', 0)} in `cheating_folder/`, "
            f"{n_folder.get('not_cheating_folder', 0)} in `not_cheating_folder/`, "
            f"{n_folder.get('new_not_cheating', 0)} in `new_not_cheating/`), each scored by a model that "
            f"**never trained on it** (out-of-fold cross-validation).", ""]
    out += [f"- **v2 verdict** (`{art['experiment']}`): CHEATING when the average reading probability "
            f"while the candidate speaks is ≥ {art['video_threshold']:.3f}. *Reading time* = total length "
            f"of detected reading segments."]
    if has_prev:
        out += [f"- **v2 @ 25 fps** (`{prev_art['experiment']}`, archived in `archive/25fps/`): same rule, "
                f"threshold {prev_art['video_threshold']:.3f}."]
    out += [f"- **v1 verdict**: CHEATING when the top-k window mean is ≥ {v1_thr:.3f} "
            f"(v1's own out-of-fold scores).", ""]

    cols = [("v2_correct", f"v2 ({fps:g} fps)")] + ([("v2_25fps_correct", "v2 (25 fps, previous)")]
                                                     if has_prev else []) + [("v1_correct", "v1")]
    out += ["| | " + " | ".join(h for _, h in cols) + " |", "|---|" + "---|" * len(cols)]
    stats = [_counts(df, c) for c, _ in cols]
    out += ["| Cheating caught | " + " | ".join(f"{s[0]} / {s[1]}" for s in stats) + " |",
            "| Clean cleared | " + " | ".join(f"{s[2]} / {s[3]}" for s in stats) + " |",
            "| Overall | " + " | ".join(f"{s[4]} / {s[5]}" for s in stats) + " |", ""]

    if has_prev:
        changed = df[df.v2_verdict != df.v2_25fps_verdict]
        if len(changed):
            out += ["**Verdicts that changed with the switch to 15 fps:** " + "; ".join(
                f"{r['name']} ({r['actual']}: {r['v2_25fps_verdict']} → {r['v2_verdict']}, "
                f"{'now right' if r['v2_correct'] else 'now wrong'})" for _, r in changed.iterrows()), ""]
        else:
            out += ["**No verdict changed with the switch to 15 fps.**", ""]
    border = df[df.v2_borderline]
    if len(border):
        out += ["Near the v2 threshold (within 5 %, treat as borderline): " +
                ", ".join(f"{r['name']} ({r['actual']}, {r['v2_score']:.3f})" for _, r in border.iterrows()), ""]

    if "sampled_fps" in df:
        slow = df[(df.sampled_fps - fps).abs() >= 0.5]
        out += [f"**Processing:** {len(df)} videos, {len(df) - len(slow)} sampled at {fps:g} fps"
                + (f"; {len(slow)} at another rate — recordings slower than {fps:g} fps keep every frame (" +
                   ", ".join(f"{r['name']} {r['sampled_fps']:.1f} fps" for _, r in slow.iterrows()) + ")"
                   if len(slow) else "")
                + f". Face found in {df.face_ratio.median():.1%} of frames (median; lowest "
                  f"{df.face_ratio.min():.1%}).", ""]

    head = ["#", "video", "folder", "v2 score", "v2 verdict", "correct", "reading time", "segments"]
    if has_prev:
        head += ["25 fps score", "25 fps verdict"]
    head += ["v1 score", "v1 verdict", "v1 correct"]
    for actual, title in (("cheating", "Cheating videos"), ("clean", "Clean videos")):
        sub = df[df.actual == actual]
        out += ["", f"## {title} ({len(sub)})", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for i, (_, r) in enumerate(sub.iterrows(), 1):
            cells = [str(i), r["name"] + (" *(reviewed)*" if r["reviewed"] else ""), r["folder"],
                     f"{r['v2_score']:.3f}", r["v2_verdict"], _tick(r["v2_correct"]), r["reading_time"],
                     str(r["reading_segments"])]
            if has_prev:
                cells += [f"{r['v2_25fps_score']:.3f}", r["v2_25fps_verdict"]]
            cells += [f"{r['v1_score']:.3f}", r["v1_verdict"], _tick(r["v1_correct"])]
            out.append("| " + " | ".join(cells) + " |")
    out += ["", f"_Generated by `src/all_videos_results.py` from "
                f"`results/oof_video_scores__{art['experiment']}.csv`; CSV copy: `results/ALL_VIDEOS_RESULTS.csv`._", ""]
    return "\n".join(out)


def main() -> int:
    df = build()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "ALL_VIDEOS_RESULTS.md").write_text(markdown(df))
    csv = df.copy()
    for c in [c for c in csv.columns if c.endswith("_correct")]:
        csv[c] = csv[c].map({True: "yes", False: "no"})
    csv.to_csv(RESULTS_DIR / "ALL_VIDEOS_RESULTS.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'ALL_VIDEOS_RESULTS.md'} + .csv ({len(df)} videos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
