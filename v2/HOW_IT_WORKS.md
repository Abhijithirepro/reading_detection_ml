# How the v2 reading detector works — start to end, and the trained model's parameters

This explains, in simple words, what the v2 cheating (reading) detector takes in, how it
processes a video, which data and measurements it uses, how it reaches the final
CHEATING / NOT CHEATING result, how it was trained, and every parameter of the trained
XGBoost model and what it does.

All numbers come from the shipped model `v2-15fps-xgb-selftrain-hardneg-SHIPPED`
(`v2/models/model.json`, `v2/models/reading_xgb.json`, settings in `v2/src/train.py` and
`v2/src/common.py`, run record in `v2/results/20261013-231419__v2-15fps-xgb-selftrain-hardneg-SHIPPED.json`).

---

## In one paragraph

The software takes **only the interview video**. It looks at **15 pictures per second**, tracks
**478 points on the face**, and measures **eyes, blinks, head, mouth and brows**. It works out
**when the person is talking** from their lips (no sound). It summarises every **8-second slice**
of video into **78 numbers**, comparing the person with **their own listening behaviour** and with
**honest people**. A model (**XGBoost**, 400 small decision trees) trained on **100 videos
(21 cheaters, 79 honest, 3 with exact reviewer timings)** turns those numbers into a **reading
score for every second**. Seconds scoring **0.175 or more** become **reading moments** with times
and reasons. If the **average score while talking** is **0.1658 or more**, the video is marked
**CHEATING**.

There are two flows:

- **Flow A** — what happens to **one video** every time it is checked.
- **Flow B** — how the software **learned** to do this (done once, beforehand).

---

## The big picture

```
 INTERVIEW VIDEO (.mp4 / .webm)
        │
        ▼
 ① Take 15 pictures per second, shrink to ≤ 640 px wide
        │
        ▼
 ② Face tracking on every picture  → eye, head, blink, mouth, brow values
        │
        ▼
 ③ Detect when the candidate is TALKING (from the lips, no sound)
        │
        ▼
 ④ Cut into 8-second slices, one every second → 78 summary numbers per slice
        │
        ▼
 ⑤ Trained model (XGBoost) → reading score 0–1 for every second
        │
        ▼
 ⑥ Seconds ≥ 0.175 joined into "reading moments" + reasons
        │
        ▼
 ⑦ Average score while talking ≥ 0.1658 ?  → CHEATING / NOT CHEATING
        │
        ▼
 OUTPUT: verdict, score, moments with times and reasons → Excel report / web app
```

---

## Flow A – What happens to one video

### ① Input: the video

- **What goes in:** only the interview recording (.mp4 or .webm).
- **What is NOT used:** no sound, no screen recording, no keyboard or browser data. **Only the face.**
- The software takes **15 pictures (frames) per second**. A 12:36 video becomes about **11,337 pictures**.
- Big videos are shrunk to **at most 640 pixels wide**. Without this, the software could learn
  "sharp video = honest" by mistake, because many honest videos were recorded in higher quality
  than the cheating ones.

### ② Face tracking on every picture

Google's free face-tracking tool (**MediaPipe Face Landmarker**) finds the face and places
**478 points** on it, including the irises (the coloured part of the eyes). From those points,
**each picture** gets these values:

| Group | Values measured on every picture |
|---|---|
| **Eyes – where they look** | Iris position left-right and up-down (`gaze_x`, `gaze_y`), plus "looking in / out / up / down" scores for each eye |
| **Eyes – open or closed** | How open the eyes are (for counting **blinks**), blink and squint scores |
| **Head** | Turning left-right (yaw), nodding up-down (pitch), tilting (roll) |
| **Mouth** | Gap between the lips, jaw opening, mouth closed, smile |
| **Brows** | Brows pulled down, inner brow raised |
| **Distance** | Face size, which shows leaning in towards the screen |

If no face is found in a picture (the person turned away or left the frame), that picture is
marked as a gap. This is the **"Face found"** value in the report. Videos with a face in fewer
than 70% of pictures are flagged for a manual check.

### ③ Detect when the candidate is talking

- Uses **lip movement and jaw opening only**, with no sound.
- Every moment of the video becomes **talking** or **silent (listening)**.
- This matters for two reasons:
  - Cheaters in these videos mostly **read while answering out loud**.
  - The software compares each person's behaviour **while talking** with their own behaviour
    **while listening**.

### ④ 8-second slices → 78 numbers

The video is cut into **8-second slices, starting one every second** (0–8 s, 1–9 s, 2–10 s …).
Each slice is summarised into **78 numbers**, in these groups:

| Group | What the numbers describe | Why it helps |
|---|---|---|
| **Talking** | How much of the slice they talked, mouth and jaw activity, talking in the last 10–30 s | Reading here mostly happens *while answering* |
| **Eyes moving like reading** | Slow left-to-right drift followed by a quick jump back (like reading a line, then going to the next), how often that repeats, how wide the eyes move | "Eyes drifting along a line" and "side-to-side spread" in the report |
| **Where the eyes rest** | Eye height compared with the person's own **listening** position, eyes "parked" on one spot, eyes looking away to think | "Eye height vs listening" and "big eye jumps" in the report |
| **Head** | How still the head is, how much it turns / nods / tilts, whether eyes move without the head | "Head movement" in the report |
| **Blinks and face** | Blink rate, how open the eyes are, squint, **brow lowered**, leaning in | "Blinks per min" and "brow lowered" in the report |
| **Talking vs listening** | The same person's blinking, head and eye behaviour **while talking minus while listening**, over the 30 seconds around the slice | Every person is compared with **themselves** |
| **Compared with their own video** | Each key number relative to that person's own average in the video | Someone who always sits still or always blinks little isn't mistaken for a reader |

The full list of the 78 numbers is in [Appendix: the 78 input numbers](#appendix-the-78-input-numbers).

### ⑤ The model gives a reading score

- The 78 numbers of each slice go into the trained model (**XGBoost**). It answers: **"how much does
  this slice look like reading?"** as a number from 0 to 1.
- Slices overlap, so every second is covered by several of them. **Each second's score** is the
  average of the slices whose centre is within 2 seconds of it. It is then smoothed with a
  3-second median, so one odd second can't jump up or down on its own.
- Result: **one reading score for every second of the video** (the "Second by second" table in the report).

### ⑥ From seconds to "reading moments"

1. A second with a score **≥ 0.175** counts as a **reading second**.
2. Reading seconds next to each other are **joined into one moment**; gaps **under 3 seconds** are bridged.
3. Moments **shorter than 4 seconds** are **thrown away** as noise.
4. Each moment gets:
   - its start and end time
   - its average and peak score
   - a confidence label (High ≥ 0.60, Medium 0.35–0.60, Low < 0.35)
   - **reasons**: which of the 7 easy-to-understand signs (blinking, head, eye drift, eye jumps,
     eye spread, eye height, brow) differ from the **normal** values of honest people talking

### ⑦ The final result

1. Take all the seconds where the candidate was **talking** (mouth moving for at least half of that second).
2. **Average their reading scores.** That is the **v2 score**. (With fewer than 10 talking seconds the score is 0.)
3. **v2 score ≥ 0.1658 → CHEATING.** Below that → **NOT CHEATING.**

### ⑧ Output

The result contains:

- the verdict and the v2 score
- every reading moment, with times and reasons
- the score for every second

This feeds the **web app** (a clickable timeline) and the **Excel report**.

### A real example traced through: Kalaiselvan Karthikeyan (ir8241161)

```
Video 12:36  →  11,337 pictures at 15 per second  →  face found in 100% of them
→ talking detected for 9:27
→ ~750 slices of 8 s → 78 numbers each → a reading score for each of the 756 seconds
→ seconds ≥ 0.175 joined → 18 reading moments, 7:19 in total (9 High confidence)
   e.g. 6:15–6:41: still head, lowered brows, eyes moving along lines
→ average score while talking = 0.449  ≥  0.1658
→ CHEATING  →  he is a known cheater  →  CAUGHT ✅
```

---

## Flow B – How the software learned (done once, beforehand)

### The data it learned from

| Data | How much | Used for |
|---|---|---|
| **Cheating videos** (`cheating_folder`) | 21 videos | People confirmed by humans to have cheated |
| **Honest videos** (`not_cheating_folder`, `new_not_cheating`) | 79 videos | People confirmed honest |
| **Manual reviews** (`manual_review_for_candidate/`) | 3 CSV files (Sai Koti Reddy, Muthakani, Marthala Ashok Reddy) | A person watched the whole video and wrote **start time, end time, reason** for every stretch of reading |

All 100 videos together give **170,160 slices** of 8 seconds:

| Slices | Count | Label used in training |
|---|---|---|
| Reviewed videos, inside a reviewer's reading time | 1,238 | **reading (1)** |
| Reviewed videos, outside the reviewer's reading time | 3,034 | **not reading (0)** |
| Reviewed videos, half in / half out | 518 | skipped (unclear) |
| Honest videos, candidate talking | 89,915 | **not reading (0)** — the "hard" examples |
| Honest videos, candidate silent | 52,176 | **not reading (0)** |
| Unreviewed cheating videos (18 videos) | 23,279 | unknown — only their most reading-like ~15% of talking slices are used (self-training, below) |

### The learning steps

1. **Run Flow A steps ①–④ on all 100 videos**, giving the 170,160 slices with 78 numbers each.
2. **Label the slices (the "answer key"):**
   - Reviewed videos: a slice is **reading** if **≥ 60%** of it falls inside a reviewer's marked
     time, and **not reading** if **≤ 10%** does. Slices in between are skipped.
   - Honest videos: **every slice is "not reading".**
   - The 18 unreviewed cheating videos have no exact times, so they are **not labelled** at first.
3. **Teach the model:** XGBoost looks at the labelled slices and learns which combinations of the
   78 numbers mean "reading".
4. **Self-training:** the model scored the 18 unreviewed cheating videos. The **top 15% most
   reading-like talking slices** of each were added as extra "reading" examples, at half
   importance. This lets it learn from more cheaters than just the 3 reviewed ones.
5. **Hard examples:** **honest people while answering** get their own full share of importance, so
   the model learns that "talking normally" is **not** reading. This keeps false alarms low.
6. **Fair testing (out-of-fold):** the videos were split into 5 groups. Each video was scored by a
   model trained **without** it, like an exam with questions the student never saw. No video is
   ever scored by a model that saw its own labels or its own self-training examples.
7. **Choosing the two cut-offs:**
   - **0.175 per second** — chosen to match the reviewers' marked times as closely as possible,
     while flagging at most 2% of the honest videos' seconds.
   - **0.1658 per video** — chosen to **avoid accusing honest people**: a false accusation counts
     as worse than a miss.
8. **Save the model.** The final model is trained once more on all the labelled data with the same
   settings and saved (`v2/models/model.json` + `v2/models/reading_xgb.json`). Flow A uses it on
   every new video.

### How well it does (fair, out-of-fold test)

| Measure | Result |
|---|---|
| Cheaters caught | **10 of 21** |
| Honest people wrongly flagged | **1 of 79** |
| Overall correct | 88 of 100 |
| Of the videos it called CHEATING, really cheating (precision) | 91% (10 of 11) |
| Reviewer-marked reading stretches found (3 reviewed videos) | **60 of 64** |

---

## The trained XGBoost model — every decision parameter

### What XGBoost is, in simple words

XGBoost builds **many small decision trees**. Each tree asks a few yes/no questions about the 78
numbers of a slice (for example, "is the candidate talking more than 72% of the slice?") and ends in
a **leaf** that holds a small number: positive pushes towards *reading*, negative pushes towards
*not reading*.

The trees are built one after another. **Each new tree tries to fix the mistakes of the trees
before it.** To score a slice, the model:

1. starts from a fixed starting value,
2. adds up the leaf values of **all 400 trees**,
3. turns the total into a probability between 0 and 1.

```
 slice's 78 numbers
        │
        ├─► tree 1   → leaf +0.12
        ├─► tree 2   → leaf −0.03
        ├─► …
        └─► tree 400 → leaf +0.01
                          │
   starting value −0.51 + sum of all leaves  =  e.g. +1.19
                          │
          probability = 1 / (1 + e^(−1.19))  =  0.77   → "reading score" of that slice
```

### A real tree from the model (tree #1 of 400)

This is the first tree, exactly as stored in `reading_xgb.json`. "Yes" goes to the first branch.

```
Is the slice < 72% talking (speak_frac < 0.717)?
├─ YES (mostly not talking)
│   Is mouth activity, relative to this person's normal, very low (mouth_act_vrel < 0.006)?
│   ├─ YES → talking in the 30 s around it < 88%?  → … leaf −0.079 / −0.027 (not reading)
│   │                                               or blinks-while-talking < 2.3/min → +0.122 (reading)
│   └─ NO  → … leaves between −0.076 and +0.079
└─ NO (mostly talking)
    Does the head move < 2.0 °/s more while talking than while listening (ctx_head_vs_listen < 1.965)?
    ├─ YES (head stays as still as when listening)
    │   Blinking while talking < 36/min (ctx_blink_speak < 36.4)?
    │   ├─ YES → up-down eye movement while talking small (< 0.153)?  → leaf +0.117 (READING)
    │   │                                                     otherwise → leaf +0.037
    │   └─ NO  → …  leaf +0.128 / −0.068
    └─ NO (head moves a lot more while talking — natural speech)
        → squint checks → leaves −0.079 / +0.129 / −0.020 / −0.072 (mostly NOT reading)
```

In plain words, the strongest "reading" path of this tree is: **mostly talking + head as still as
when listening + blinking less than usual + eyes not moving up and down → push towards reading.**
That is exactly the pattern of someone reading an answer off the screen. The other 399 trees each
add their own small correction.

### XGBoost settings (hyperparameters)

Set in `v2/src/train.py` (`XGB_PARAMS`); confirmed in the saved model (400 trees, every tree at most
4 levels deep = at most 31 nodes, 78 inputs).

| Parameter | Value | What it does | Why this value helps here |
|---|---|---|---|
| `objective` | `binary:logistic` | The model answers a yes/no question (reading or not) and outputs a **probability 0–1**. | We need a score, not just yes/no, so moments and the video average can be built from it. |
| `n_estimators` | **400** | Number of trees that are added together. | Enough small steps to learn subtle patterns; with the low learning rate, fewer trees would under-learn. |
| `max_depth` | **4** | Each tree asks **at most 4 questions in a row** (at most 16 leaves). | Small trees can only learn simple combinations (e.g. "talking + still head + few blinks"), so they don't memorise individual people. Important with only 3 reviewed videos. |
| `learning_rate` | **0.05** | Each tree's vote is shrunk to 5% of its full size before it is added. | Many small careful steps instead of a few big ones → smoother scores and less over-fitting. |
| `subsample` | **0.8** | Each tree is built from a random **80%** of the training slices. | Every tree sees slightly different data, so the model doesn't depend on any one stretch of video. |
| `colsample_bytree` | **0.7** | Each tree may only use a random **70%** of the 78 numbers (~55). | Forces the model to use many different signs instead of leaning on one, which helps when a sign is missing for a person (e.g. someone who never blinks much). |
| `min_child_weight` | **5** | A leaf must be backed by enough (weighted) training examples, or the split is not made. | Stops rules built from a handful of unusual slices. |
| `reg_lambda` | **1.0** | L2 penalty: shrinks leaf values towards zero. | Keeps any single leaf from pushing too hard; makes the model more cautious. |
| `tree_method` | `hist` | Groups each number into bins (a histogram) before looking for split points. | Much faster training on 170,000 slices, with practically the same accuracy. |
| `eval_metric` | `logloss` | How training measures its error: punishes **confident wrong** answers heavily. | Matches the probability output; rewards honest, well-calibrated scores. |
| `random_state` | **42** | Fixed seed for the random choices (`subsample`, `colsample_bytree`, fold split). | Re-training gives the same model and the same results. |
| `base_score` | **0.375** (stored in the model) | The starting value before any tree (−0.51 on the log-odds scale). XGBoost sets it from the weighted share of "reading" examples. | Trees only need to learn the *difference* from this starting point. |
| `sample_weight` | see next table | How much each training slice counts. | Balances the very unequal groups (1,238 reading slices vs 142,091 honest ones). |

Everything else is XGBoost's default (for example `gamma = 0`, `reg_alpha = 0`,
`scale_pos_weight = 1`). Class balance is handled by `sample_weight` instead.

### Training weights (how much each kind of example counts)

The training slices are put into groups. **Each group gets a fixed share of the total importance,
and inside a group every video counts equally**, so one long video can't dominate.

| Group | Share | Parameter | Why |
|---|---|---|---|
| Reviewed **reading** slices | 1.0 | — | The most trusted examples of reading. |
| Self-training **reading** slices (unreviewed cheaters) | 0.5 | `pseudo_weight = 0.5` | Probably reading, but not confirmed by a person → half importance. |
| Reviewed **not reading** slices (cheaters, outside the marked times) | 0.5 | `reviewed_neg_weight = 0.5` | Teaches "a cheater is not reading all the time", at reduced importance. |
| Honest candidates **silent** | 1.0 | — | Normal listening behaviour. |
| Honest candidates **talking** ("hard negatives") | 1.0 | `clean_speech_weight = 1.0` | The hardest case: normal answering must not look like reading. Giving it its own share keeps false alarms low. |

### Other training parameters

| Parameter | Value | What it does |
|---|---|---|
| `label_pos_frac` | 0.6 | A reviewed slice is "reading" if ≥ 60% of it is inside a reviewer's marked time. |
| `label_neg_frac` | 0.1 | … and "not reading" if ≤ 10% is. In between it is skipped. |
| `pseudo_frac` | 0.15 | Self-training takes the top 15% most reading-like **talking** slices of each unreviewed cheating video. |
| `folds` | 5 | The 5 groups for fair (out-of-fold) testing. Each reviewed video is held out in its own group. |
| `seed` | 42 | Fixed randomness, so results can be repeated. |
| `clean_budget` | 0.02 | When choosing the per-second cut-off, at most 2% of honest seconds may be flagged. |
| `video_stat` | `speech_prob_mean` | The video score is the **average reading score while talking**. |

### Decision parameters after the model (stored in `model.json`)

These turn the model's slice scores into seconds, moments and the final verdict.

| Parameter | Value | What it does | How it helps |
|---|---|---|---|
| `target_fps` | 15 | Pictures per second analysed. | Same rate everywhere (training, web app, production browser), so scores match. |
| `max_frame_width` | 640 px | Wider frames are shrunk first. | Removes the "video quality" shortcut between honest and cheating videos. |
| `read_window_seconds` | 8 | Length of each slice. | About 1–2 lines of text read aloud. |
| `read_stride_seconds` | 1 | A new slice starts every second. | Gives one score per second, so moments have exact times. |
| centre radius | 2 s | Each second averages the slices whose centre is within 2 s. | An 8 s slice says most about its middle; keeps moment edges sharp. |
| `smooth_s` | 3 | 3-second moving median of the per-second score. | Removes one-second flicker. |
| `second_threshold` | **0.175** | A second with score ≥ 0.175 is a **reading second**. | Chosen as the best match with the reviewers' marked times (best F1) while flagging ≤ 2% of honest seconds. |
| `merge_gap_seconds` | 3 | Reading seconds less than 3 s apart are joined into one moment. | A short pause (breath, looking up once) doesn't split one reading stretch into many. |
| `min_segment_seconds` | 4 | Moments shorter than 4 s are dropped. | Removes blips that are too short to be real reading. |
| speaking cut-off | 0.5 | A second counts as "talking" when the mouth moves for at least half of it. | The video score only looks at answering time. |
| minimum talking | 10 s | With fewer than 10 talking seconds the video score is 0. | Avoids a verdict from almost no data. |
| `video_threshold` | **0.16577** (shown as 0.1658 / 0.17) | v2 score ≥ this → **CHEATING**. | Chosen on the fair out-of-fold scores with **F0.5**, which counts precision (not accusing honest people) twice as much as recall (catching every cheater). |
| confidence bands | 0.60 / 0.35 | High / Medium / Low label for a moment (report only). | Tells a reviewer how strong each moment is. |
| `min_face_ratio` | 0.70 | Videos with a face in < 70% of pictures are flagged for a manual check. | Warns when the numbers may be unreliable. |

### What the model pays most attention to (feature importance)

Share of the model's total decision power (gain-based importance from the shipped model). The top 10
numbers make up about two-thirds of it.

| # | Number | Importance | Simple meaning |
|---|---|---|---|
| 1 | `speak_frac` | 23.0% | How much of the 8-second slice the candidate is talking |
| 2 | `ctx_head_vs_listen` | 9.1% | Head movement while talking **minus** head movement while listening (the same person) |
| 3 | `blink_rate` | 8.7% | Blinks per minute in the slice |
| 4 | `mouth_act_vrel` | 7.2% | Mouth activity compared with this person's own normal in the video |
| 5 | `ctx_blink_speak` | 7.0% | Blink rate while talking, over the 30 s around the slice |
| 6 | `ctx_head_speed_speak` | 3.1% | Head speed while talking, over the 30 s around the slice |
| 7 | `mouth_act` | 2.6% | Mouth activity in the slice |
| 8 | `ctx_speak_frac` | 2.0% | How much of the 30 s around the slice is talking |
| 9 | `jaw_mean` | 2.0% | Average jaw opening |
| 10 | `ctx_v_std_speak` | 1.8% | Up-down eye movement while talking, over the 30 s around the slice |

In plain words, the model mainly asks: **Is the candidate answering? While answering, does the head
stay unusually still compared with when they listen? Do they blink much less than normal?** That
combination is the strongest sign of reading answers off the screen.

---

## Appendix: the 78 input numbers

| Group | Count | Numbers |
|---|---|---|
| Talking | 8 | `speak_frac`, `mouth_act`, `speak_act`, `jaw_act`, `jaw_mean`, `smile_mean`, `speak_ctx30`, `speak_prev10` |
| Eyes moving left-right / like reading lines | 10 | `h_std`, `h_range`, `gx_std`, `h_skew_dom`, `h_skew_abs`, `h_prog_frac`, `h_sweeps_dom`, `h_sweeps_max`, `gx_sweeps_dom`, `h_ac_peak` |
| Where the eyes rest | 8 | `v_mean_listen`, `gy_mean_listen`, `pitch_mean_listen`, `v_std`, `disp`, `aversion_frac`, `parked_frac`, `big_jump_rate` |
| Head | 5 | `yaw_std`, `pitch_std`, `roll_std`, `head_speed`, `corr_h_yaw` |
| Blinks and face | 5 | `blink_rate`, `ear_mean`, `squint_mean`, `brow_down_mean`, `face_scale_rel` |
| Talking vs listening (30 s around the slice) | 16 | `ctx_speak_frac`, `ctx_h_std_speak`, `ctx_v_std_speak`, `ctx_v_mean_speak`, `ctx_aversion_speak`, `ctx_blink_speak`, `ctx_head_speed_speak`, `ctx_jump_speak`, `ctx_h_std_vs_listen`, `ctx_v_std_vs_listen`, `ctx_aversion_vs_listen`, `ctx_blink_vs_listen`, `ctx_head_vs_listen`, `ctx_sweeps_speak`, `ctx_pause_rate`, `ctx_artic_cv` |
| Compared with their own video (`_vrel`) | 26 | `h_std_vrel`, `h_range_vrel`, `h_sweeps_dom_vrel`, `h_sweeps_max_vrel`, `h_ac_peak_vrel`, `h_skew_dom_vrel`, `gx_std_vrel`, `v_mean_listen_vrel`, `gy_mean_listen_vrel`, `v_std_vrel`, `disp_vrel`, `aversion_frac_vrel`, `parked_frac_vrel`, `yaw_std_vrel`, `pitch_std_vrel`, `head_speed_vrel`, `blink_rate_vrel`, `jaw_act_vrel`, `mouth_act_vrel`, `squint_mean_vrel`, `brow_down_mean_vrel`, `face_scale_rel_vrel`, `pitch_mean_listen_vrel`, `ear_mean_vrel`, `big_jump_rate_vrel`, `h_prog_frac_vrel` |
| **Total** | **78** | |

Names ending in `_listen` compare with the same person's listening position; names starting with
`ctx_` look at the 30 seconds around the slice (talking parts only, or talking minus listening);
names ending in `_vrel` are relative to that person's own median in the video.
