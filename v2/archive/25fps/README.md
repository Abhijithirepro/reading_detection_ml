# v2 at 25 fps (archived 2026-10-13)

The model v2 shipped before switching every stage to 15 fps (the production rate).
Experiment `v2-xgb-selftrain-hardneg-SHIPPED` in `../../RESULTS_LOG.md`.

- `models/` — `model.json`, `reading_xgb.json`, `config.json` (`target_fps` 25).
- The 25 fps per-frame features and windows it was trained on are in
  `v2/data/features_25fps/` and `v2/data/windows_25fps/` (git-ignored, local only).

To use it again, copy `models/*` back into `v2/models/` (the 25 fps `config.json`
with it; `predict.py` refuses a model whose `target_fps` differs from the config).
