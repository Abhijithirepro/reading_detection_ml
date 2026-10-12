"""
results_log.py (v2, new) — append-only record of every evaluation run.

Each run is written twice:
  * results/<timestamp>__<name>.json   full machine-readable metrics
  * RESULTS_LOG.md                     one human-readable section appended at the end

Nothing is ever overwritten, so the log is the history of what was tried and
how it scored against the manual reading reviews.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from common import PROJECT_ROOT, RESULTS_DIR

LOG_PATH = PROJECT_ROOT / "RESULTS_LOG.md"

_HEADER = """# v2 results log

Append-only. Every evaluation run adds a section at the bottom; nothing is edited
afterwards. Machine-readable copies live in `results/`.

**Metrics**
- *per-second* (manual reviews): each second of a reviewed video is reading / not
  reading. `AUC` ranks the model's per-second reading probability; `P`/`R`/`IoU` are
  at the shipped operating threshold; `spans caught` = reviewer spans with >= 50% of
  their seconds flagged.
- *video-level*: CHEATING vs clean verdicts over the 100 interview recordings
  (21 cheating / 79 clean). `matched` = cheating vs the 28 clean videos recorded on
  the same platform/resolutions (the honest comparison).

"""


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def table(rows: list, cols: list = None) -> str:
    if not rows:
        return "_(none)_\n"
    cols = cols or list(rows[0].keys())
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        out += "| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |\n"
    return out


def append(name: str, summary: str, sections: dict, payload: dict) -> Path:
    """Append one run. `sections` maps a heading to markdown (e.g. from table())."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    fname = time.strftime("%Y%m%d-%H%M%S") + f"__{name}.json"
    (RESULTS_DIR / fname).write_text(json.dumps(dict(name=name, time=stamp, summary=summary,
                                                     **payload), indent=2, default=str))
    if not LOG_PATH.exists():
        LOG_PATH.write_text(_HEADER)
    md = f"\n---\n\n## {stamp} — {name}\n\n{summary.strip()}\n\n"
    for head, body in sections.items():
        md += f"**{head}**\n\n{body.strip()}\n\n"
    md += f"_json: `results/{fname}`_\n"
    with LOG_PATH.open("a") as fh:
        fh.write(md)
    return RESULTS_DIR / fname
