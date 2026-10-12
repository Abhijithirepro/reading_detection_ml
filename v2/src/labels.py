"""
labels.py (v2, new) — manual reading reviews -> per-second ground truth.

Reviewers mark the spans where a candidate reads from the screen, one CSV per
candidate, in manual_review_for_candidate/:

    start time (min:sec),End time (min:sec),supsicious/not suspsicious ,reason
    2:50,3:13,reading,giving introduction by reading ...

The file is matched to a video by its IR number (the `ir8266049` in both names).
Everything outside the marked spans counts as NOT reading — reviewers watched the
whole recording. A spreadsheet sometimes rewrites "24:25" as "24:25:00"; every
3-part value in these sheets is minutes:seconds:00, so it is read that way.

Usage:
    python src/labels.py                 # list the reviews it finds + reading totals
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common import LABELS_DIR

IR_RE = re.compile(r"ir(\d{5,})", re.IGNORECASE)


@dataclass
class Review:
    ir: str
    path: Path
    segments: list = field(default_factory=list)   # [(start_s, end_s, tag, reason)]

    @property
    def reading_seconds(self) -> float:
        return float(sum(e - s for s, e, *_ in self.segments))


def parse_timestamp(text: str) -> float:
    """'2:50' -> 170.0;  '24:25:00' (sheet-mangled mm:ss) -> 1465.0;  '95' -> 95.0."""
    parts = [p for p in str(text).strip().split(":") if p != ""]
    if not parts:
        raise ValueError(f"empty timestamp {text!r}")
    nums = [float(p) for p in parts]
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 3:
        return nums[0] * 60 + nums[1]          # mm:ss:00 (see module docstring)
    raise ValueError(f"unrecognised timestamp {text!r}")


def ir_of(name: str):
    m = IR_RE.search(name)
    return m.group(1) if m else None


def load_review(path: Path) -> Review:
    ir = ir_of(path.name)
    segs = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    for row in rows[1:]:
        if len(row) < 2 or not row[0].strip() or not row[1].strip():
            continue
        try:
            s, e = parse_timestamp(row[0]), parse_timestamp(row[1])
        except ValueError:
            continue
        if e <= s:
            continue
        tag = row[2].strip().lower() if len(row) > 2 else "reading"
        reason = row[3].strip() if len(row) > 3 else ""
        if tag.startswith("not"):                # an explicit "not suspicious" row
            continue
        segs.append((s, e, tag, reason))
    segs.sort()
    return Review(ir=ir, path=path, segments=segs)


def load_reviews(labels_dir: Path = LABELS_DIR) -> dict:
    """{ir: Review} for every review CSV in labels_dir."""
    out = {}
    for f in sorted(Path(labels_dir).glob("*.csv")):
        ir = ir_of(f.name)
        if ir:
            out[ir] = load_review(f)
    return out


def review_for(video_name: str, reviews: dict = None):
    """The Review whose IR number appears in this video/parquet name, or None."""
    reviews = load_reviews() if reviews is None else reviews
    ir = ir_of(video_name)
    return reviews.get(ir) if ir else None


def truth_mask(segments, n: int, dt: float, t0: float = 0.0) -> np.ndarray:
    """Per-sample reading mask on a uniform grid t = t0 + i*dt."""
    t = t0 + np.arange(n) * dt
    m = np.zeros(n, dtype=bool)
    for s, e, *_ in segments:
        m |= (t >= s) & (t < e)
    return m


def window_label_fraction(segments, start_s: float, end_s: float) -> float:
    """Fraction of [start_s, end_s) covered by reading segments."""
    cov = 0.0
    for s, e, *_ in segments:
        cov += max(0.0, min(e, end_s) - max(s, start_s))
    return cov / max(end_s - start_s, 1e-9)


def mmss(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def main():
    reviews = load_reviews()
    if not reviews:
        print(f"No review CSVs in {LABELS_DIR}")
        return 1
    for ir, r in reviews.items():
        print(f"ir{ir}: {len(r.segments)} reading spans, {r.reading_seconds / 60:.1f} min  ({r.path.name})")
        for s, e, tag, reason in r.segments[:3]:
            print(f"    {mmss(s)}-{mmss(e)}  {tag}  {reason[:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
