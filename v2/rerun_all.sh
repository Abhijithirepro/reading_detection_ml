#!/usr/bin/env bash
# rerun_all.sh (v2) — rebuild every v2 result from the per-frame features:
# windows -> the experiment ladder (each run appended to RESULTS_LOG.md) -> the
# shipped model -> all-video table, config CSVs, evaluation report, web-app library,
# cheating / not-cheating workbooks.
#
# Run from the project root after extracting features (15 fps, see README):
#   venv/bin/python v2/src/extract_features.py --workers 7
#   bash v2/rerun_all.sh            # add --with-cnn for the CNN bake-off (~slow)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venv/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE
P=v2-15fps                                   # experiment-name prefix for this sampling rate

$PY v2/src/build_windows.py

# experiment ladder (same settings as the 25 fps runs in RESULTS_LOG.md)
$PY v2/src/train.py --exp $P-xgb-base --no-ship --drop-group ctx \
    --note "Strong labels from 3 reviews; 8 s/1 s windows; eye+mouth+head features without answer-context."
$PY v2/src/train.py --exp $P-xgb-context --no-ship \
    --note "Adds the answer-context group."
$PY v2/src/train.py --exp $P-xgb-context-selftrain --no-ship --pseudo-frac 0.3 \
    --note "Self-training on the 18 unreviewed cheaters (fold-specific, A/B cross-scored)."
# the shipped recipe -> v2/models/
$PY v2/src/train.py --exp $P-xgb-selftrain-hardneg-SHIPPED --pseudo-frac 0.15 --pseudo-weight 0.5 \
    --clean-speech-weight 1.0 --clean-budget 0.02 --video-stat speech_prob_mean \
    --note "Shipped recipe, re-run on 15 fps features (production rate)."

$PY v2/src/evaluate.py --v1-baseline          # v1's numbers on the same reviews / videos

if [[ "${1:-}" == "--with-cnn" ]]; then
  $PY v2/src/train_seq.py --exp $P-cnn-strong-labels --video-stat speech_prob_mean \
      --note "Same strong labels and folds as the XGBoost runs, no self-training."
  $PY v2/src/train_seq.py --exp $P-cnn-vs-xgb-selftrain --pseudo-frac 0.15 --pseudo-weight 0.5 \
      --clean-speech-weight 1.0 --video-stat speech_prob_mean --compare-xgb \
      --note "CNN with the recipe that fixed XGBoost, head-to-head on identical windows."
fi

$PY v2/src/all_videos_results.py              # results/ALL_VIDEOS_RESULTS.md + .csv
$PY v2/src/config_results_csv.py              # CONFIG_AND_RESULTS_v2.csv (+ ../CONFIG_AND_RESULTS_v1.csv)
$PY v2/src/report.py --eval                   # reports/evaluation_report.html
$PY v2/webapp/library.py --force              # pre-scored candidates for the web app
$PY v2/src/cheating_cases_xlsx.py             # results/{,NOT_}CHEATING_CASES_REPORT.xlsx (Google Sheets)
$PY v2/src/cheating_cases_xlsx.py --only clean --v2-only   # results/NOT_CHEATING_CASES_REPORT_v2_only.xlsx
echo "done — restart v2/webapp/server.py to serve the new model"
