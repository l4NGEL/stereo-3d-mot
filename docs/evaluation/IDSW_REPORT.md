# Identity switches: cause and a controlled fix

Plan and selection rule, fixed before any run of the new variants: [`IDSW_PLAN.md`](IDSW_PLAN.md). Everything
below uses the unchanged P-K2D scorer of [`PROTOCOL.md`](PROTOCOL.md) (E = 0, score ≥ 0.25, min_hits 3, counts
pooled from raw counts) on the same detections as the published evaluation. Reproduce: `bash scripts/eval/run_idsw.sh`
(`--score-only` to rescore the dumps). Results: [`results/idsw/`](results/idsw/).

**Held-out caveat.** The hypothesis tested here was formed from an event-level look at the held-out dumps. The
held-out numbers are a confirmation on already-inspected data, not an untouched test. All thresholds were chosen
on dev (0000, 0003, 0017) only.

## 1. Baseline integrity

The rebuilt pipeline (`f3ab419` + this change) produced **byte-identical** `dets.txt` on all 11 sequences and
**byte-identical** S\* tracks (`A_iou_last` vs the published `iou2d_s0.25_h3` / `A5_plain_iou` dumps). Rescored,
they give exactly the published numbers: dev MOTA 0.530 / IDF1 0.617 / 35 IDSW; held-out 0.492 / 0.553 / 246.
SORT reproduces too (held-out 0.551 / 0.624 / 60).

**Input difference to SORT.** The project's tracker never sees detections without valid depth; SORT does. That is
5.4 % of the held-out detections at score ≥ 0.25 (6,148 → 5,817). Feeding SORT only the valid-depth detections
(`SORT_valid`) changes almost nothing (IDSW 60 → 61, IDF1 0.624 → 0.622), so this filter does not explain the
gap between the two trackers.

## 2. Variants

| | Changed from A | Changed components |
|---|---|---|
| A `A_iou_last` | — | S\*: IoU vs last observed box, one pass, max_age 30 |
| B `B_iou_last_age1` | max_age 30 → 1 | track lifetime only |
| C `C_iou_pred` | IoU box: last observed → predicted | box only (new 2D constant-velocity box filter per track) |
| D0 `D0_two_stage_none` | predicted box + two passes, lost tracks never re-matched | box + priority, no recovery |
| D1 `D1_two_stage_iou` | predicted box + two passes, lost tracks re-matched by the same predicted-box IoU | box + priority |
| D2 `D2_chi11.345_app0.5` | predicted box + two passes, lost tracks re-matched only if HSV appearance ≤ 0.5 AND 3D Mahalanobis² ≤ 11.345 | box + priority + stricter lost-track rule |
| E `SORT` / `SORT_valid` | — | `sort_baseline.py`; `_valid`: valid-depth detections only |

**Predicted box — why a separate 2D filter.** The track's 3D state is `[x, y, z, vx, vy, vz]` in the left-camera
frame; it has no box size and no ego-motion compensation. Projecting it would need a size model (scale by
`z_last / z_pred`) and would carry the ~1.5 m object-depth error of REPORT.md §3 into the image. Detection boxes are
the observed quantity, so each track now also keeps an image-plane constant-velocity filter on `[cx, cy, w, h]`
(`makeConstantVelocityBox2D`, SORT's noise constants, declared not tuned). It is predicted once per frame inside
`Track::predict()`, updated only with matched boxes, clamps sizes to ≥ 1 px, and falls back to the last box if the
state is non-finite. `lastBox()` (the reported box) and the E = 0 `isEmittable()` rule are unchanged.

**Two-stage** (`AssociationMethod::kTwoStage`): stage 1 = tracks matched on the previous frame (confirmed or
tentative; `stage1_max_tsu = 0`, not tuned) × all detections by predicted-box IoU ≥ 0.3; its matches are final.
Stage 2 = the remaining ("lost") tracks × the leftover detections. In D2 a missing descriptor or a non-finite
distance makes a pair infeasible (never a neutral fallback). The feasible cost is the mean of the two
gate-normalised terms. Confirmation, deletion and birth are unchanged.

## 3. Dev selection (rule fixed in advance)

D2 grid, 12 settings: IDF1 0.652–0.685. Selected `lost_chi2 = 11.345, lost_app = 0.5` (0.4 / 0.5 / 0.6 tied at
0.685; 0.5 and 0.6 have 22 IDSW; the stricter one wins). The recommended configuration across
{A, B, C, D0, D1, D2} is **C** (dev IDF1 0.718, next D1 0.696; no tie within 0.005).

| Dev (1,274 GT) | IDF1 | MOTA | IDSW | Frag | Recall | Precision |
|---|---|---|---|---|---|---|
| A | 0.617 | 0.530 | 35 | 51 | 0.819 | 0.758 |
| B | 0.598 | 0.526 | 31 | 42 | 0.798 | 0.763 |
| **C** | **0.718** | 0.551 | 19 | 47 | 0.827 | 0.760 |
| D0 | 0.620 | 0.514 | 33 | 36 | 0.779 | 0.765 |
| D1 | 0.696 | 0.551 | 18 | 47 | 0.827 | 0.760 |
| D2 (selected) | 0.685 | 0.531 | 22 | 46 | 0.830 | 0.746 |
| SORT | 0.610 | 0.526 | 22 | 30 | 0.781 | 0.767 |

Dev has 35 A-switches in total, so dev supports large differences only.

## 4. Held-out (8 sequences, 4,550 GT object-frames)

| System | IDF1 | MOTA | IDSW | Frag | Recall | Precision | FP | FN | IDTP / IDFP / IDFN |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.553 | 0.492 | 246 | 171 | 0.684 | 0.832 | 626 | 1439 | 2293 / 1444 / 2257 |
| B max_age 1 | 0.593 | 0.535 | 112 | 157 | 0.665 | 0.863 | 480 | 1525 | 2389 / 1116 / 2161 |
| **C predicted box** | **0.644** | 0.529 | 84 | 162 | 0.686 | 0.832 | 628 | 1429 | 2673 / 1076 / 1877 |
| D0 | 0.589 | 0.533 | 97 | 132 | 0.649 | 0.873 | 429 | 1598 | 2334 / 1047 / 2216 |
| D1 | 0.642 | 0.531 | 81 | 154 | 0.683 | 0.835 | 612 | 1443 | 2653 / 1066 / 1897 |
| D2 | 0.647 | 0.518 | 86 | 159 | 0.696 | 0.814 | 726 | 1381 | 2732 / 1163 / 1818 |
| SORT | 0.624 | 0.551 | 60 | 112 | 0.653 | 0.880 | 406 | 1578 | 2473 / 905 / 2077 |
| SORT_valid | 0.622 | 0.548 | 61 | 111 | 0.646 | 0.884 | 384 | 1611 | 2447 / 876 / 2103 |

Paired bootstrap over sequences (10,000 draws; 95 % interval; sequence wins a/b):

| Comparison | ΔIDF1 | ΔMOTA | ΔRecall | Reading |
|---|---|---|---|---|
| B − A | +0.040 [−0.000, +0.126] 5/3 | +0.043 [+0.007, +0.121] | **−0.019 [−0.033, −0.013] 0/8** | fewer IDSW (−134), paid for in recall on every sequence; IDF1 gain not demonstrated |
| **C − A** | **+0.091 [+0.044, +0.155] 8/0** | +0.037 [+0.010, +0.091] | +0.002 (noise) | demonstrated, no recall cost |
| D1 − C | −0.003 [−0.011, +0.011] | +0.001 (noise) | −0.003 (noise) | priority adds nothing measurable on top of the predicted box |
| D2 − D1 | +0.005 [−0.005, +0.015] | −0.013 (noise) | **+0.014 [+0.005, +0.042] 8/0** | stricter lost-track rule recovers more (recall up), more FP (726 vs 612); IDF1 not demonstrated |
| D0 − C | **−0.056 [−0.089, −0.000]** | +0.004 (noise) | **−0.037 [−0.056, −0.027] 0/8** | never recovering lost tracks costs identity and recall |
| C − SORT | +0.020 [−0.025, +0.057] 3/5 | −0.021 [−0.062, +0.003] | +0.033 [+0.023, +0.055] 8/0 | no demonstrated difference in IDF1/MOTA; C still has more IDSW (84 vs 60) |

**Is one sequence driving this?** 0001 held 86 of A's 246 switches. Without 0001, A → C is still 160 → 81 IDSW and
IDF1 0.525 → 0.595. C improves IDF1 on all 8 sequences. The exception to the pattern is 0008 (80 GT): C has 5
IDSW and IDF1 0.506 there, versus 0 IDSW and IDF1 0.738 for B and 0.713 for SORT. Not investigated further.

Per-sequence IDSW (A / B / C / D1 / D2 / SORT): 0001 86/6/3/3/3/2 · 0002 11/10/7/8/7/9 · 0004 50/21/9/8/10/6 ·
0007 17/15/12/12/15/4 · 0008 15/0/5/4/3/0 · 0010 5/5/4/3/5/2 · 0012 2/4/2/1/2/4 · 0016 60/51/42/42/41/33.

Tracking time is not reported: every run had diagnostics on, which adds work inside `Tracker::update()`.

## 5. What the identity switches are

Classes as defined in the plan (they sum to IDSW for every system). Internal state comes from the tracker's own
dump or from SORT's internal list, never from the scorer's output.

| Held-out | A: lost track took over (gap 1) | B: after a gap → existing track | C: after a gap → new identity | D: other (mostly swap between active tracks) | total |
|---|---|---|---|---|---|
| A | **148** | 47 | 28 | 23 | 246 |
| B | 10 | 30 | 54 | 18 | 112 |
| C | 19 | 31 | 14 | 20 | 84 |
| D0 | 0 | 17 | 68 | 12 | 97 |
| D1 | 13 | 29 | 20 | 19 | 81 |
| D2 | 19 | 34 | 10 | 23 | 86 |
| SORT | 0 | 14 | 37 | 9 | 60 |

**Class A in the baseline** (148 events, association diagnostics at the switch frame):
- The track that took the detection had been unmatched for a median of 5 frames (IQR 4–8). In 130 of the 148
  events it had earlier followed a *different* GT object.
- The track that lost the object was still eligible. In 135 of 141 events with diagnostics its IoU with the same
  detection was above the 0.3 gate (median 0.60), but the lost track's frozen box overlapped more (median 0.71;
  higher in 140/141). The Hungarian solver took the cheaper pair.
- By sequence: 0001 83, 0004 27, 0008 14, 0016 14, 0007 8, others 2.

**Visual check, 0001 frames 110–133** ([`figures/idsw_0001_frozen_boxes.jpg`](figures/idsw_0001_frozen_boxes.jpg)):
the camera drives forward along a street lined with parked cars on both kerbs. Unmatched tracks leave a trail of
frozen grey boxes along the image path the parked cars sweep out towards the image edge. The event log shows the
consequence: GT 36 is handed through tracks 73 → 54 → 28 → 12 → 19 → 64 between frames 116 and 121, and GT 38
through 80 → 73 → 54 → 28 → 12 → 19 between frames 123 and 128. That is the same chain of stationary lost tracks.
What the figure supports is limited: parked vehicles and forward camera motion in this segment. It does not show
this is the mechanism on the other sequences (0004 has 27 class-A events and was not inspected).

**Gap events (B + C) are not shown to be occlusion.** Across the 75 gap events of the baseline (313 scored
frames in which the object was missed):
- In 96 frames no detection overlapped the object (IoU ≥ 0.5).
- In 64 frames the only overlapping detection was below the 0.25 score.
- In 0 frames the detection was lost to the depth filter.
- In 153 frames a usable detection existed, but the object still counted as missed. For example it may have gone
  to a tentative track that is not yet emitted; this was not separated further.

KITTI's `occluded` label reached 2 ("largely occluded") in only 13 of the 75 gaps. So the data show detector
misses and low scores, not occlusion as such.

## 6. Verdicts

| Claim | Status | Evidence |
|---|---|---|
| Most baseline IDSW are a lost track with a frozen box winning a detection from the active track | **supported** | class A = 148/246; old track above the gate in 135/141; removing the frozen-box competition (B, C, D0) cuts class A to 10 / 19 / 0 |
| Shortening max_age alone brings IDSW near SORT's | **partly** | 246 → 112 (SORT 60); recall falls on 8/8 sequences; class C (new identity after a gap) doubles 28 → 54 |
| The predicted box alone fixes most of it | **supported** | C: IDSW 84, IDF1 +0.091 [+0.044, +0.155] 8/8, recall unchanged |
| Two-stage priority helps beyond the predicted box | **not demonstrated** | D1 − C inside the noise on every metric |
| A stricter appearance + Mahalanobis rule for lost tracks helps identity | **not demonstrated** | D2 − D1 IDF1 +0.005 (noise); recall +0.014 (8/8) but FP +114, MOTA −0.013 (noise) |
| IDSW are mainly caused by occlusion | **not supported** | 148/246 happen with no gap at all; gap events are mostly missed or low-score detections, `occluded = 2` in 13/75 gaps |
| The remaining gap to SORT (84 vs 60 IDSW) | **unexplained** | C and SORT differ in more than one respect: box parameterisation, output rule (SORT needs `hit_streak ≥ min_hits` again after a miss), lifetime; none of these was isolated |

## 7. Recommendation (from dev) and limits

Dev recommends **C** (`assoc=iou2d iou_pred=1`, max_age 30): the smallest change, and on held-out the largest
single gain (IDF1 0.553 → 0.644, IDSW 246 → 84, recall unchanged). It is still **not** demonstrated to beat SORT:
IDF1 +0.020 and MOTA −0.021 are both inside the noise, and it has more identity switches. The shipped default
(`configs/*.yaml`) is left unchanged, because C was not selected by the PROTOCOL.md §4 process. Adopting it is a
decision for the project owner.

Limits:
- Held-out was inspected before this work.
- 150-frame prefixes; one sequence (0016) is 46 % of the ground truth.
- Dev has 35 IDSW.
- Visual review covers one segment of one sequence and has one reviewer.
- 0008 goes the other way and was not investigated.
- Tracker timing was not measured.

## 8. Round 2: the remaining gap to SORT

Pre-registered in [`IDSW_PLAN.md`](IDSW_PLAN.md) round 2. Two of the three remaining differences between C and SORT are
tested as factors. The first is SORT's reporting rule: after a miss, a track is not reported again until `min_hits`
*consecutive* matches (`reconfirm_after_miss`, new, off by default). The second is track lifetime (`max_age`). Everything
else is as in C. The C-equivalent `R0_age30` reproduces C's tracks byte for byte on all 11 sequences.

**Dev** (12-point grid): the rule selects **C again** (`R0_age30`, IDF1 0.718). Reconfirmation lowers dev IDSW
(`R1_age30`: 12 vs 19) but costs recall (0.772 vs 0.827) and IDF1 (0.700). Every shorter `max_age` lowers IDF1.

**Held-out** (C plus the factor contrasts, run once):

| | IDF1 | MOTA | IDSW | Frag | Recall | Precision | FP | FN |
|---|---|---|---|---|---|---|---|---|
| C `R0_age30` (selected) | 0.644 | 0.529 | 84 | 162 | 0.686 | 0.832 | 628 | 1429 |
| `R0_age1` | 0.614 | 0.538 | 82 | 152 | 0.667 | 0.858 | 503 | 1515 |
| `R1_age30` (reconfirm) | 0.641 | 0.537 | **54** | 139 | 0.641 | 0.874 | 419 | 1634 |
| `R1_age1` | 0.612 | 0.535 | 72 | 134 | 0.645 | 0.872 | 431 | 1613 |
| SORT | 0.624 | 0.551 | 60 | 112 | 0.653 | 0.880 | 406 | 1578 |
| SORT, max_age 30 (diagnostic only) | 0.663 | 0.550 | 44 | 119 | 0.653 | 0.875 | 424 | 1580 |

- **Reconfirmation** (`R1_age30 − R0_age30`): the IDSW rate falls on 6/6 sequences where it differs, and 0008 is fixed
  (5 → 0 IDSW, IDF1 0.506 → 0.709). But recall drops **−0.045 [−0.060, −0.035] on 8/8** sequences, and IDF1
  changes by −0.003 (noise). Most of the IDSW reduction comes from reporting fewer frames: a track that is not
  reported cannot switch. It is not better association.
- **`R1_age30` vs SORT**: IDSW 54 vs 60, rate difference inside the noise. IDF1 +0.017 (noise). MOTA −0.014
  [−0.019, −0.005], worse on 6/8 sequences.
- **Short lifetime** hurts in both settings: `R0_age1 − R0_age30` has recall −0.019, excluding 0.
- **SORT with max_age 30** is the best held-out system on IDF1 and IDSW. On dev it is worse than C (0.653 vs 0.718),
  and it was not a candidate. It shows that the remaining difference sits in the parts not tested here: SORT's
  `[u, v, s, r]` box filter and its constants, and its first-frames birth exception. Lifetime and the reporting rule
  are not where the remaining difference is.

**Verdict:** the IDSW gap to SORT can be closed (54 vs 60) with SORT's reporting rule, but only by giving up recall,
with no IDF1 gain. The dev rule therefore keeps C. The third factor (box parameterisation) remains untested. Testing
it would need a new dev-only round; held-out has now been used for two rounds.

## 9. Round 3: SORT's box filter and assignment objective

Pre-registered in [`IDSW_PLAN.md`](IDSW_PLAN.md) round 3. Two new options, both off by default, and both have unit tests:
- `sort_box_model`: SORT's `[u, v, s, r]` filter and constants.
- `iou_max_total_assignment`: SORT's objective. It maximises total IoU and applies the gate afterwards. The default
  first maximises the number of gated pairs, which can prefer two weak matches over one strong one.

`T000` reproduces C byte for byte on all 11 sequences.

**Dev** (2 × 2 × 2 with reconfirm): the rule selects **C again** (`T000`, IDF1 0.718; `T001` 0.714 is inside the tie
but has one IDSW more). The SORT box filter lowers dev IDF1 in every combination (`T010` 0.679, `T011` 0.674).

**Held-out** (the 2 × 2 with reconfirm off; last use of held-out for this question):

| | IDF1 | MOTA | IDSW | Frag | Recall | Precision |
|---|---|---|---|---|---|---|
| C `T000` (selected) | 0.644 | 0.529 | 84 | 162 | 0.686 | 0.832 |
| `T010` SORT box | 0.632 | 0.531 | 84 | 167 | 0.687 | 0.834 |
| `T001` SORT assignment | 0.645 | 0.530 | 88 | 160 | 0.687 | 0.834 |
| `T011` both | 0.656 | 0.533 | 79 | 165 | 0.687 | 0.834 |
| SORT | 0.624 | 0.551 | 60 | 112 | 0.653 | 0.880 |
| SORT, max_age 30 (diagnostic) | 0.663 | 0.550 | 44 | 119 | 0.653 | 0.875 |

- **SORT box alone** (`T010 − T000`): IDF1 **−0.013 [−0.019, −0.002]**, worse on 7/8 sequences. It is also worse on dev.
- **SORT assignment alone** (`T001 − T000`): no difference on any metric.
- **Both together** (`T011`) is the best held-out variant of this round (IDF1 0.656, 79 IDSW). On dev it is second to
  last (0.674). The two splits disagree, so this is not evidence for it.

**Verdict:** neither the box filter nor the assignment objective explains the remaining difference to SORT, and C stays
the dev-selected tracker. The IDSW "gap" is also split-dependent. On dev, C has *fewer* switches than SORT (19 vs 22)
and a much higher IDF1 (0.718 vs 0.610); only held-out shows SORT ahead. The held-out sequences are now used up for this
question (three rounds), so the difference that remains is left unexplained rather than tuned against.

Differences between C and SORT that were never isolated:
- The box that is reported: SORT reports its filtered box, C the detection box.
- SORT's first-`min_hits`-frames birth exception.
- The 3D-state machinery the project's tracker still carries.
