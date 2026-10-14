# v2 results log

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


---

## 2026-10-12 19:31 — v1-baseline

Original model (`models/`, 1D-CNN, threshold 0.959) scored on the three reviewed videos. **In-sample**: v1 was trained on these very videos, so this is its best case. Per-second score = max probability of the 20 s windows covering that second. Video-level row uses v1's own out-of-fold CV scores.

**Per-second vs manual reviews**

| video | reading_s | flagged_s | auc | precision | recall | iou | spans_caught |
|---|---|---|---|---|---|---|---|
| combined_ir8308053_Muthakani_Venkateswarlu | 698 | 560 | 0.710 | 0.670 | 0.537 | 0.425 | 14/26 |
| combined_ir8244089_Marthala_Ashok_Reddy | 239 | 980 | 0.567 | 0.167 | 0.686 | 0.155 | 6/9 |
| combined_ir8266049_Sai_Koti_Reddy_Sai_Bhavanam (1) | 371 | 600 | 0.481 | 0.153 | 0.248 | 0.105 | 8/29 |

**Video level (v1 out-of-fold)**

| scope | auc | precision | recall | tp | fp | fn | tn | matched_auc | matched_fp |
|---|---|---|---|---|---|---|---|---|---|
| all 100 | 0.843 | 0.900 | 0.429 | 9 | 1 | 12 | 78 | 0.861 | 0/28 |

_json: `results/20261012-193125__v1-baseline.json`_

---

## 2026-10-12 20:50 — v2-xgb-base

XGBoost reading detector, 62 features, 5-fold CV (reviewed videos held out one per fold; unreviewed cheating never trained on). Second threshold 0.0111 (max F1, clean budget 2%); video rule: reading_seconds >= 37.00. Dropped groups: ['ctx']. Strong labels from 3 reviews; 8 s/1 s windows; eye+mouth+head features without answer-context.

**Per-second vs manual reviews (out-of-fold; pooled AUC = mean over videos)**

| video | reading_s | flagged_s | auc | precision | recall | iou | spans_caught |
|---|---|---|---|---|---|---|---|
| combined_ir8244089_Marthala_Ashok_Reddy | 239 | 37 | 0.858 | 0.324 | 0.050 | 0.045 | 0/9 |
| combined_ir8266049_Sai_Koti_Reddy_Sai_Bhavanam (1) | 371 | 119 | 0.782 | 0.529 | 0.170 | 0.148 | 3/29 |
| combined_ir8308053_Muthakani_Venkateswarlu | 698 | 53 | 0.947 | 1.000 | 0.076 | 0.076 | 0/26 |
| ALL reviewed (pooled) | 1308 | 209 | 0.862 | 0.612 | 0.098 | 0.090 | 3/64 |

Clean videos: 2652 s flagged of 142643 s (1.86%); 30/79 clean videos have any flagged span.

**Video level (out-of-fold)**

| stat | auc | precision | recall | tp | fp | fn | tn | matched_auc | matched_fp | unreviewed_caught | threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reading_seconds | 0.857 | 0.567 | 0.810 | 17 | 13 | 4 | 66 | 0.884 | 3/28 | 14/18 | 37.000 |

**Video-statistic AUCs**

| reading_seconds | reading_share | reading_share_of_speech | top10_mean | p95 | speech_prob_mean | speech_prob_p90 |
|---|---|---|---|---|---|---|
| 0.857 | 0.874 | 0.876 | 0.870 | 0.863 | 0.868 | 0.869 |

**Top features**

head_speed (0.138), speak_frac (0.125), mouth_act_vrel (0.114), blink_rate (0.085), jaw_act (0.058), mouth_act (0.032), roll_std (0.031), yaw_std_vrel (0.027), smile_mean (0.027), jaw_act_vrel (0.021), gx_std (0.021), jaw_mean (0.021)

_json: `results/20261012-205037__v2-xgb-base.json`_

---

## 2026-10-12 20:50 — v2-xgb-context

XGBoost reading detector, 78 features, 5-fold CV (reviewed videos held out one per fold; unreviewed cheating never trained on). Second threshold 0.005 (max F1, clean budget 2%); video rule: reading_seconds >= 97.00. Dropped groups: none. Adds the answer-context group (ctx_*: +-15 s speaking-only eye/head/blink stats, speaking-vs-listening contrasts, speech rhythm).

**Per-second vs manual reviews (out-of-fold; pooled AUC = mean over videos)**

| video | reading_s | flagged_s | auc | precision | recall | iou | spans_caught |
|---|---|---|---|---|---|---|---|
| combined_ir8244089_Marthala_Ashok_Reddy | 239 | 124 | 0.905 | 0.484 | 0.251 | 0.198 | 2/9 |
| combined_ir8266049_Sai_Koti_Reddy_Sai_Bhavanam (1) | 371 | 398 | 0.790 | 0.415 | 0.445 | 0.273 | 14/29 |
| combined_ir8308053_Muthakani_Venkateswarlu | 698 | 38 | 0.934 | 1.000 | 0.054 | 0.054 | 0/26 |
| ALL reviewed (pooled) | 1308 | 560 | 0.877 | 0.470 | 0.201 | 0.175 | 16/64 |

Clean videos: 2745 s flagged of 142643 s (1.92%); 29/79 clean videos have any flagged span.

**Video level (out-of-fold)**

| stat | auc | precision | recall | tp | fp | fn | tn | matched_auc | matched_fp | unreviewed_caught | threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reading_seconds | 0.816 | 0.600 | 0.571 | 12 | 8 | 9 | 71 | 0.848 | 2/28 | 10/18 | 97.000 |

**Video-statistic AUCs**

| reading_seconds | reading_share | reading_share_of_speech | top10_mean | p95 | speech_prob_mean | speech_prob_p90 |
|---|---|---|---|---|---|---|
| 0.816 | 0.828 | 0.828 | 0.872 | 0.864 | 0.875 | 0.867 |

**Top features**

speak_frac (0.162), ctx_head_speed_speak (0.151), ctx_head_vs_listen (0.119), head_speed (0.072), mouth_act (0.048), ctx_blink_speak (0.037), blink_rate (0.032), mouth_act_vrel (0.024), yaw_std_vrel (0.017), ctx_speak_frac (0.017), speak_act (0.016), head_speed_vrel (0.016)

_json: `results/20261012-205048__v2-xgb-context.json`_

---

## 2026-10-12 20:51 — v2-xgb-context-selftrain

XGBoost reading detector, 78 features, 5-fold CV (reviewed videos held out one per fold; unreviewed cheating never trained on). Second threshold 0.3 (max F1, clean budget 2%); video rule: reading_seconds >= 160.00. Dropped groups: none. Self-training: top 30% of each unreviewed cheating video's speaking windows as pseudo-reading (weight 0.5), A/B cross-scored. Self-training adds reading examples from the 18 unreviewed cheaters (fold-specific selection, A/B cross-scoring: leak-free).

**Per-second vs manual reviews (out-of-fold; pooled AUC = mean over videos)**

| video | reading_s | flagged_s | auc | precision | recall | iou | spans_caught |
|---|---|---|---|---|---|---|---|
| combined_ir8244089_Marthala_Ashok_Reddy | 239 | 397 | 0.921 | 0.544 | 0.904 | 0.514 | 9/9 |
| combined_ir8266049_Sai_Koti_Reddy_Sai_Bhavanam (1) | 371 | 874 | 0.885 | 0.419 | 0.987 | 0.416 | 28/29 |
| combined_ir8308053_Muthakani_Venkateswarlu | 698 | 512 | 0.947 | 0.928 | 0.681 | 0.646 | 20/26 |
| ALL reviewed (pooled) | 1308 | 1783 | 0.918 | 0.593 | 0.808 | 0.526 | 57/64 |

Clean videos: 2651 s flagged of 142643 s (1.86%); 29/79 clean videos have any flagged span.

**Video level (out-of-fold)**

| stat | auc | precision | recall | tp | fp | fn | tn | matched_auc | matched_fp | unreviewed_caught | threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reading_seconds | 0.789 | 0.750 | 0.571 | 12 | 4 | 9 | 75 | 0.780 | 1/28 | 9/18 | 160.000 |

**Video-statistic AUCs**

| reading_seconds | reading_share | reading_share_of_speech | top10_mean | p95 | speech_prob_mean | speech_prob_p90 |
|---|---|---|---|---|---|---|
| 0.789 | 0.798 | 0.802 | 0.832 | 0.821 | 0.823 | 0.817 |

**Top features**

speak_frac (0.196), mouth_act_vrel (0.070), ctx_head_vs_listen (0.069), ctx_head_speed_speak (0.054), ctx_blink_speak (0.050), blink_rate (0.049), mouth_act (0.044), ctx_speak_frac (0.024), jaw_mean (0.018), disp (0.018), ctx_v_std_speak (0.017), speak_act (0.017)

_json: `results/20261012-205121__v2-xgb-context-selftrain.json`_

---

## 2026-10-12 20:55 — v2-xgb-selftrain-hardneg-SHIPPED

XGBoost reading detector, 78 features, 5-fold CV (reviewed videos held out one per fold; unreviewed cheating never trained on). Second threshold 0.175 (max F1, clean budget 2%); video rule: speech_prob_mean >= 0.21. Dropped groups: none. Self-training: top 15% of each unreviewed cheating video's speaking windows as pseudo-reading (weight 0.5), A/B cross-scored. Clean speaking windows weighted as their own share (1.0). Chosen configuration, shipped to v2/models/. Explored first (unlogged): pseudo-frac 0.15/0.3 x clean-speech-weight 0/1, clean budget 1%/2%; 1% cost too much recall (Muthakani 11/26 spans).

**Per-second vs manual reviews (out-of-fold; pooled AUC = mean over videos)**

| video | reading_s | flagged_s | auc | precision | recall | iou | spans_caught |
|---|---|---|---|---|---|---|---|
| combined_ir8244089_Marthala_Ashok_Reddy | 239 | 382 | 0.932 | 0.568 | 0.908 | 0.537 | 9/9 |
| combined_ir8266049_Sai_Koti_Reddy_Sai_Bhavanam (1) | 371 | 821 | 0.882 | 0.441 | 0.976 | 0.436 | 28/29 |
| combined_ir8308053_Muthakani_Venkateswarlu | 698 | 457 | 0.964 | 0.976 | 0.639 | 0.629 | 18/26 |
| ALL reviewed (pooled) | 1308 | 1660 | 0.926 | 0.617 | 0.784 | 0.534 | 55/64 |

Clean videos: 2789 s flagged of 142643 s (1.96%); 33/79 clean videos have any flagged span.

**Video level (out-of-fold)**

| stat | auc | precision | recall | tp | fp | fn | tn | matched_auc | matched_fp | unreviewed_caught | threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speech_prob_mean | 0.847 | 0.900 | 0.429 | 9 | 1 | 12 | 78 | 0.832 | 1/28 | 6/18 | 0.205 |

**Video-statistic AUCs**

| reading_seconds | reading_share | reading_share_of_speech | top10_mean | p95 | speech_prob_mean | speech_prob_p90 |
|---|---|---|---|---|---|---|
| 0.818 | 0.821 | 0.825 | 0.844 | 0.835 | 0.847 | 0.837 |

**Top features**

speak_frac (0.252), ctx_head_vs_listen (0.099), mouth_act_vrel (0.064), blink_rate (0.056), ctx_blink_speak (0.050), ctx_head_speed_speak (0.049), mouth_act (0.026), ctx_speak_frac (0.018), ctx_v_std_speak (0.018), disp (0.017), speak_act (0.016), jaw_mean (0.016)

_json: `results/20261012-205542__v2-xgb-selftrain-hardneg-SHIPPED.json`_

---

## 2026-10-12 20:56 — v2-cnn-strong-labels

1D-CNN on raw per-frame channels (eye_h, eye_v, gaze_x, gaze_y, jaw, mouth, blink, yaw, pitch, speaking), 8 s @ 12.5 Hz, same folds and metrics as the XGBoost runs. Second threshold 0.0302. Same strong labels and folds as the XGBoost runs, no self-training.

**Per-second vs manual reviews (out-of-fold)**

| video | reading_s | flagged_s | auc | precision | recall | iou | spans_caught |
|---|---|---|---|---|---|---|---|
| combined_ir8244089_Marthala_Ashok_Reddy | 239 | 0 | 0.587 | 0.000 | 0.000 | 0.000 | 0/9 |
| combined_ir8266049_Sai_Koti_Reddy_Sai_Bhavanam (1) | 371 | 36 | 0.700 | 0.833 | 0.081 | 0.080 | 4/29 |
| combined_ir8308053_Muthakani_Venkateswarlu | 698 | 21 | 0.758 | 0.762 | 0.023 | 0.023 | 1/26 |

Clean videos: 2744 s flagged of 142643 s (1.92%).

**Video level (out-of-fold)**

| stat | auc | precision | recall | tp | fp | fn | tn | matched_auc | matched_fp | unreviewed_caught | threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speech_prob_mean | 0.722 | 0.538 | 0.333 | 7 | 6 | 14 | 73 | 0.772 | 1/28 | 7/18 | 0.024 |

**Video-statistic AUCs**

| reading_seconds | reading_share | reading_share_of_speech | top10_mean | p95 | speech_prob_mean | speech_prob_p90 |
|---|---|---|---|---|---|---|
| 0.670 | 0.719 | 0.718 | 0.723 | 0.780 | 0.722 | 0.765 |

_json: `results/20261012-205613__v2-cnn-strong-labels.json`_

---

## 2026-10-12 21:52 — v2-cnn-vs-xgb-selftrain

1D-CNN on raw per-frame channels (eye_h, eye_v, gaze_x, gaze_y, jaw, mouth, blink, yaw, pitch, speaking), 8 s @ 12.5 Hz, 10 epochs; self-training 15% (weight 0.5), clean-speech weight 1.0; verdict stat speech_prob_mean. Same folds, leak-free pseudo-labels, thresholds and metrics as the XGBoost runs — XGBoost re-run on the identical window set for a head-to-head. Fair re-run of the CNN with the recipe that fixed XGBoost.

**Head-to-head (out-of-fold)**

| model | spans_caught | sec_auc_mean | sec_precision | sec_recall | iou_mean | clean_flagged | video_auc | caught | false_alarms | matched_auc | Marthala auc/rec/spans | Sai auc/rec/spans | Muthakani auc/rec/spans |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 55/64 | 0.926 | 0.617 | 0.784 | 0.534 | 1.96% | 0.847 | 9/21 | 1 | 0.832 | 0.93 / 0.91 / 9/9 | 0.88 / 0.98 / 28/29 | 0.96 / 0.64 / 18/26 |
| CNN | 3/64 | 0.655 | 0.557 | 0.041 | 0.035 | 1.80% | 0.805 | 8/21 | 4 | 0.840 | 0.43 / 0.00 / 0/9 | 0.75 / 0.06 / 2/29 | 0.78 / 0.04 / 1/26 |
| XGB+CNN average | 39/64 | 0.920 | 0.597 | 0.537 | 0.402 | 1.75% | 0.886 | 10/21 | 0 | 0.900 | 0.92 / 0.68 / 7/9 | 0.88 / 0.91 / 25/29 | 0.96 / 0.29 / 7/26 |

**CNN per-second vs manual reviews**

| video | reading_s | flagged_s | auc | precision | recall | iou | spans_caught |
|---|---|---|---|---|---|---|---|
| combined_ir8244089_Marthala_Ashok_Reddy | 239 | 28 | 0.428 | 0.000 | 0.000 | 0.000 | 0/9 |
| combined_ir8266049_Sai_Koti_Reddy_Sai_Bhavanam (1) | 371 | 35 | 0.755 | 0.657 | 0.062 | 0.060 | 2/29 |
| combined_ir8308053_Muthakani_Venkateswarlu | 698 | 34 | 0.783 | 0.912 | 0.044 | 0.044 | 1/26 |

_json: `results/20261012-215211__v2-cnn-vs-xgb-selftrain.json`_

---

## 2026-10-13 23:13 — v2-15fps-xgb-base

XGBoost reading detector, 15 fps, 62 features, 5-fold CV (reviewed videos held out one per fold; unreviewed cheating never trained on). Second threshold 0.0111 (max F1, clean budget 2%); video rule: reading_seconds >= 27.00. Dropped groups: ['ctx']. Strong labels from 3 reviews; 8 s/1 s windows; eye+mouth+head features without answer-context.

**Per-second vs manual reviews (out-of-fold; pooled AUC = mean over videos)**

| video | reading_s | flagged_s | auc | precision | recall | iou | spans_caught |
|---|---|---|---|---|---|---|---|
| combined_ir8244089_Marthala_Ashok_Reddy | 239 | 32 | 0.846 | 0.188 | 0.025 | 0.023 | 0/9 |
| combined_ir8266049_Sai_Koti_Reddy_Sai_Bhavanam (1) | 371 | 103 | 0.726 | 0.330 | 0.092 | 0.077 | 0/29 |
| combined_ir8308053_Muthakani_Venkateswarlu | 698 | 34 | 0.940 | 1.000 | 0.049 | 0.049 | 0/26 |
| ALL reviewed (pooled) | 1308 | 169 | 0.837 | 0.438 | 0.057 | 0.050 | 0/64 |

Clean videos: 2675 s flagged of 142644 s (1.88%); 29/79 clean videos have any flagged span.

**Video level (out-of-fold)**

| stat | auc | precision | recall | tp | fp | fn | tn | matched_auc | matched_fp | unreviewed_caught | threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reading_seconds | 0.844 | 0.529 | 0.857 | 18 | 16 | 3 | 63 | 0.876 | 4/28 | 15/18 | 27.000 |

**Video-statistic AUCs**

| reading_seconds | reading_share | reading_share_of_speech | top10_mean | p95 | speech_prob_mean | speech_prob_p90 |
|---|---|---|---|---|---|---|
| 0.844 | 0.869 | 0.872 | 0.837 | 0.853 | 0.844 | 0.860 |

**Top features**

speak_frac (0.177), head_speed (0.117), mouth_act_vrel (0.080), blink_rate (0.080), yaw_std (0.034), smile_mean (0.033), mouth_act (0.032), jaw_act (0.032), roll_std (0.027), yaw_std_vrel (0.026), jaw_act_vrel (0.025), speak_ctx30 (0.021)

_json: `results/20261013-231313__v2-15fps-xgb-base.json`_

---

## 2026-10-13 23:13 — v2-15fps-xgb-context

XGBoost reading detector, 15 fps, 78 features, 5-fold CV (reviewed videos held out one per fold; unreviewed cheating never trained on). Second threshold 0.005 (max F1, clean budget 2%); video rule: reading_seconds >= 81.00. Dropped groups: none. Adds the answer-context group.

**Per-second vs manual reviews (out-of-fold; pooled AUC = mean over videos)**

| video | reading_s | flagged_s | auc | precision | recall | iou | spans_caught |
|---|---|---|---|---|---|---|---|
| combined_ir8244089_Marthala_Ashok_Reddy | 239 | 116 | 0.908 | 0.431 | 0.209 | 0.164 | 2/9 |
| combined_ir8266049_Sai_Koti_Reddy_Sai_Bhavanam (1) | 371 | 381 | 0.805 | 0.433 | 0.445 | 0.281 | 13/29 |
| combined_ir8308053_Muthakani_Venkateswarlu | 698 | 13 | 0.919 | 1.000 | 0.019 | 0.019 | 0/26 |
| ALL reviewed (pooled) | 1308 | 510 | 0.877 | 0.447 | 0.174 | 0.155 | 15/64 |

Clean videos: 2454 s flagged of 142644 s (1.72%); 32/79 clean videos have any flagged span.

**Video level (out-of-fold)**

| stat | auc | precision | recall | tp | fp | fn | tn | matched_auc | matched_fp | unreviewed_caught | threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reading_seconds | 0.829 | 0.591 | 0.619 | 13 | 9 | 8 | 70 | 0.866 | 1/28 | 11/18 | 81.000 |

**Video-statistic AUCs**

| reading_seconds | reading_share | reading_share_of_speech | top10_mean | p95 | speech_prob_mean | speech_prob_p90 |
|---|---|---|---|---|---|---|
| 0.829 | 0.851 | 0.854 | 0.855 | 0.851 | 0.863 | 0.851 |

**Top features**

ctx_head_speed_speak (0.143), speak_frac (0.140), ctx_head_vs_listen (0.103), head_speed (0.058), mouth_act (0.052), ctx_blink_speak (0.048), mouth_act_vrel (0.039), blink_rate (0.037), speak_act (0.022), ctx_aversion_vs_listen (0.021), ctx_speak_frac (0.021), ctx_v_std_speak (0.018)

_json: `results/20261013-231325__v2-15fps-xgb-context.json`_

---

## 2026-10-13 23:13 — v2-15fps-xgb-context-selftrain

XGBoost reading detector, 15 fps, 78 features, 5-fold CV (reviewed videos held out one per fold; unreviewed cheating never trained on). Second threshold 0.3 (max F1, clean budget 2%); video rule: reading_seconds >= 191.00. Dropped groups: none. Self-training: top 30% of each unreviewed cheating video's speaking windows as pseudo-reading (weight 0.5), A/B cross-scored. Self-training on the 18 unreviewed cheaters (fold-specific, A/B cross-scored).

**Per-second vs manual reviews (out-of-fold; pooled AUC = mean over videos)**

| video | reading_s | flagged_s | auc | precision | recall | iou | spans_caught |
|---|---|---|---|---|---|---|---|
| combined_ir8244089_Marthala_Ashok_Reddy | 239 | 347 | 0.925 | 0.539 | 0.782 | 0.469 | 8/9 |
| combined_ir8266049_Sai_Koti_Reddy_Sai_Bhavanam (1) | 371 | 849 | 0.883 | 0.425 | 0.973 | 0.420 | 28/29 |
| combined_ir8308053_Muthakani_Venkateswarlu | 698 | 771 | 0.945 | 0.840 | 0.928 | 0.789 | 26/26 |
| ALL reviewed (pooled) | 1308 | 1967 | 0.917 | 0.608 | 0.914 | 0.559 | 62/64 |

Clean videos: 2561 s flagged of 142644 s (1.80%); 33/79 clean videos have any flagged span.

**Video level (out-of-fold)**

| stat | auc | precision | recall | tp | fp | fn | tn | matched_auc | matched_fp | unreviewed_caught | threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reading_seconds | 0.771 | 0.769 | 0.476 | 10 | 3 | 11 | 76 | 0.767 | 1/28 | 7/18 | 191.000 |

**Video-statistic AUCs**

| reading_seconds | reading_share | reading_share_of_speech | top10_mean | p95 | speech_prob_mean | speech_prob_p90 |
|---|---|---|---|---|---|---|
| 0.771 | 0.786 | 0.784 | 0.804 | 0.805 | 0.805 | 0.805 |

**Top features**

speak_frac (0.216), ctx_head_vs_listen (0.076), mouth_act_vrel (0.067), ctx_blink_speak (0.063), mouth_act (0.046), blink_rate (0.041), ctx_head_speed_speak (0.033), ctx_speak_frac (0.021), jaw_mean (0.019), ctx_v_std_speak (0.019), speak_ctx30 (0.016), ctx_v_mean_speak (0.016)

_json: `results/20261013-231352__v2-15fps-xgb-context-selftrain.json`_

---

## 2026-10-13 23:14 — v2-15fps-xgb-selftrain-hardneg-SHIPPED

XGBoost reading detector, 15 fps, 78 features, 5-fold CV (reviewed videos held out one per fold; unreviewed cheating never trained on). Second threshold 0.175 (max F1, clean budget 2%); video rule: speech_prob_mean >= 0.17. Dropped groups: none. Self-training: top 15% of each unreviewed cheating video's speaking windows as pseudo-reading (weight 0.5), A/B cross-scored. Clean speaking windows weighted as their own share (1.0). Shipped recipe, re-run on 15 fps features (production rate).

**Per-second vs manual reviews (out-of-fold; pooled AUC = mean over videos)**

| video | reading_s | flagged_s | auc | precision | recall | iou | spans_caught |
|---|---|---|---|---|---|---|---|
| combined_ir8244089_Marthala_Ashok_Reddy | 239 | 342 | 0.929 | 0.567 | 0.812 | 0.501 | 9/9 |
| combined_ir8266049_Sai_Koti_Reddy_Sai_Bhavanam (1) | 371 | 797 | 0.883 | 0.445 | 0.957 | 0.437 | 28/29 |
| combined_ir8308053_Muthakani_Venkateswarlu | 698 | 606 | 0.953 | 0.901 | 0.782 | 0.720 | 23/26 |
| ALL reviewed (pooled) | 1308 | 1745 | 0.922 | 0.628 | 0.837 | 0.553 | 60/64 |

Clean videos: 2593 s flagged of 142644 s (1.82%); 37/79 clean videos have any flagged span.

**Video level (out-of-fold)**

| stat | auc | precision | recall | tp | fp | fn | tn | matched_auc | matched_fp | unreviewed_caught | threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speech_prob_mean | 0.827 | 0.909 | 0.476 | 10 | 1 | 11 | 78 | 0.825 | 1/28 | 7/18 | 0.166 |

**Video-statistic AUCs**

| reading_seconds | reading_share | reading_share_of_speech | top10_mean | p95 | speech_prob_mean | speech_prob_p90 |
|---|---|---|---|---|---|---|
| 0.804 | 0.823 | 0.819 | 0.823 | 0.816 | 0.827 | 0.816 |

**Top features**

speak_frac (0.230), ctx_head_vs_listen (0.091), blink_rate (0.087), mouth_act_vrel (0.072), ctx_blink_speak (0.070), ctx_head_speed_speak (0.031), mouth_act (0.026), ctx_speak_frac (0.020), jaw_mean (0.020), ctx_v_std_speak (0.018), speak_act (0.016), speak_ctx30 (0.014)

_json: `results/20261013-231419__v2-15fps-xgb-selftrain-hardneg-SHIPPED.json`_

---

## 2026-10-13 23:14 — v1-baseline

Original model (`models/`, 1D-CNN, threshold 0.959) scored on the three reviewed videos. **In-sample**: v1 was trained on these very videos, so this is its best case. Per-second score = max probability of the 20 s windows covering that second. Video-level row uses v1's own out-of-fold CV scores.

**Per-second vs manual reviews**

| video | reading_s | flagged_s | auc | precision | recall | iou | spans_caught |
|---|---|---|---|---|---|---|---|
| combined_ir8308053_Muthakani_Venkateswarlu | 698 | 560 | 0.710 | 0.670 | 0.537 | 0.425 | 14/26 |
| combined_ir8244089_Marthala_Ashok_Reddy | 239 | 980 | 0.567 | 0.167 | 0.686 | 0.155 | 6/9 |
| combined_ir8266049_Sai_Koti_Reddy_Sai_Bhavanam (1) | 371 | 600 | 0.481 | 0.153 | 0.248 | 0.105 | 8/29 |

**Video level (v1 out-of-fold)**

| scope | auc | precision | recall | tp | fp | fn | tn | matched_auc | matched_fp |
|---|---|---|---|---|---|---|---|---|---|
| all 100 | 0.843 | 0.909 | 0.476 | 10 | 1 | 11 | 78 | 0.861 | 0/28 |

_json: `results/20261013-231424__v1-baseline.json`_

---

## 2026-10-13 23:17 — v2-15fps-cnn-strong-labels

1D-CNN on raw per-frame channels (eye_h, eye_v, gaze_x, gaze_y, jaw, mouth, blink, yaw, pitch, speaking), 8 s @ 15 Hz, 10 epochs; no self-training; verdict stat speech_prob_mean. Same folds, leak-free pseudo-labels, thresholds and metrics as the XGBoost runs. Same strong labels and folds as the XGBoost runs, no self-training.

**Head-to-head (out-of-fold)**

| model | spans_caught | sec_auc_mean | sec_precision | sec_recall | iou_mean | clean_flagged | video_auc | caught | false_alarms | matched_auc | Marthala auc/rec/spans | Sai auc/rec/spans | Muthakani auc/rec/spans |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN | 6/64 | 0.701 | 0.725 | 0.067 | 0.064 | 1.94% | 0.742 | 10/21 | 9 | 0.787 | 0.59 / 0.00 / 0/9 | 0.77 / 0.17 / 4/29 | 0.74 / 0.03 / 2/26 |

**CNN per-second vs manual reviews**

| video | reading_s | flagged_s | auc | precision | recall | iou | spans_caught |
|---|---|---|---|---|---|---|---|
| combined_ir8244089_Marthala_Ashok_Reddy | 239 | 0 | 0.593 | 0.000 | 0.000 | 0.000 | 0/9 |
| combined_ir8266049_Sai_Koti_Reddy_Sai_Bhavanam (1) | 371 | 92 | 0.772 | 0.696 | 0.173 | 0.160 | 4/29 |
| combined_ir8308053_Muthakani_Venkateswarlu | 698 | 28 | 0.739 | 0.821 | 0.033 | 0.033 | 2/26 |

_json: `results/20261013-231741__v2-15fps-cnn-strong-labels.json`_

---

## 2026-10-13 23:22 — v2-15fps-cnn-vs-xgb-selftrain

1D-CNN on raw per-frame channels (eye_h, eye_v, gaze_x, gaze_y, jaw, mouth, blink, yaw, pitch, speaking), 8 s @ 15 Hz, 10 epochs; self-training 15% (weight 0.5), clean-speech weight 1.0; verdict stat speech_prob_mean. Same folds, leak-free pseudo-labels, thresholds and metrics as the XGBoost runs — XGBoost re-run on the identical window set for a head-to-head. CNN with the recipe that fixed XGBoost, head-to-head on identical windows.

**Head-to-head (out-of-fold)**

| model | spans_caught | sec_auc_mean | sec_precision | sec_recall | iou_mean | clean_flagged | video_auc | caught | false_alarms | matched_auc | Marthala auc/rec/spans | Sai auc/rec/spans | Muthakani auc/rec/spans |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | 60/64 | 0.922 | 0.628 | 0.837 | 0.553 | 1.82% | 0.827 | 10/21 | 1 | 0.825 | 0.93 / 0.81 / 9/9 | 0.88 / 0.96 / 28/29 | 0.95 / 0.78 / 23/26 |
| CNN | 4/64 | 0.696 | 0.613 | 0.075 | 0.066 | 1.81% | 0.791 | 17/21 | 15 | 0.821 | 0.53 / 0.04 / 0/9 | 0.74 / 0.09 / 3/29 | 0.82 / 0.08 / 1/26 |
| XGB+CNN average | 40/64 | 0.911 | 0.656 | 0.573 | 0.418 | 1.73% | 0.858 | 9/21 | 0 | 0.866 | 0.90 / 0.43 / 4/9 | 0.88 / 0.88 / 25/29 | 0.95 / 0.46 / 11/26 |

**CNN per-second vs manual reviews**

| video | reading_s | flagged_s | auc | precision | recall | iou | spans_caught |
|---|---|---|---|---|---|---|---|
| combined_ir8244089_Marthala_Ashok_Reddy | 239 | 42 | 0.527 | 0.214 | 0.038 | 0.033 | 0/9 |
| combined_ir8266049_Sai_Koti_Reddy_Sai_Bhavanam (1) | 371 | 61 | 0.740 | 0.574 | 0.094 | 0.088 | 3/29 |
| combined_ir8308053_Muthakani_Venkateswarlu | 698 | 57 | 0.823 | 0.947 | 0.077 | 0.077 | 1/26 |

_json: `results/20261013-232229__v2-15fps-cnn-vs-xgb-selftrain.json`_
