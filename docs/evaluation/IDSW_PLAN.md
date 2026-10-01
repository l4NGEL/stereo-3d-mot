# ID-switch follow-up — plan, written before any dev or held-out run of the new variants

Date: 2026-09-30. Base commit: `f3ab419` (working tree clean before this work). The frozen
[`PROTOCOL.md`](PROTOCOL.md) is not changed: same P-K2D scorer (`scripts/eval/mot_eval.py`), same emission
policy E = 0, same sequences and frame prefixes (dev 0000/0003/0017 all frames; held-out 0001, 0002, 0004,
0007, 0008, 0010, 0012, 0016, first 150 frames, 0012 all 78), counts pooled from raw counts over sequences and
classes, never averaged.

## What motivated it, and a contamination statement

An event-level analysis of the **held-out** dumps of `fcde488` (§1 of [`IDSW_REPORT.md`](IDSW_REPORT.md))
found that most of S\*'s 246 identity switches happen on consecutive frames and involve a track that had not
been matched for several frames. That hypothesis was therefore formed **by looking at held-out data**. The
held-out sequences are no longer clean for this question: every held-out number below is a *confirmation on
already-inspected data*, not an untouched test. Thresholds are still chosen on dev only.

## Variants (all: score ≥ 0.25, min_hits 3, E = 0, same detections)

| Name | Changed from A | Components |
|---|---|---|
| A `A_iou_last` | — | the dev-selected S\*: IoU vs last observed box, one pass, max_age 30 |
| B `B_iou_last_age1` | max_age 30 → 1 | only the track lifetime |
| C `C_iou_pred` | IoU box: last → predicted (2D constant-velocity box filter) | only the box used for IoU |
| D0 `D0_two_stage_none` | predicted box + two passes; lost tracks never re-matched | priority of active tracks, no recovery |
| D1 `D1_two_stage_iou` | predicted box + two passes; lost tracks re-matched by the same predicted-box IoU | priority only (same cost in both passes) |
| D2 `D2_*` | predicted box + two passes; lost tracks re-matched only if appearance AND 3D Mahalanobis gates both pass | priority + stricter lost-track rule |
| E `SORT` | — | `sort_baseline.py`, unchanged |
| E′ `SORT_valid` | SORT fed only the detections with valid depth | input control: the project's tracker never sees `valid = 0` detections |

`stage1_max_tsu` is fixed at 0 (a track is "active" only if it was matched on the previous frame), not tuned.

## Dev selection rule (fixed now)

D2 grid: `lost_chi2 ∈ {4.108, 7.815, 11.345}` (χ² 3 dof at 75 / 95 / 99 %) × `lost_app ∈ {0.3, 0.4, 0.5, 0.6}`.

1. **D2 thresholds**: maximise pooled dev P-K2D **IDF1**; ties within 0.005 → fewer IDSW → higher MOTA → the
   stricter setting (smaller `lost_chi2`, then smaller `lost_app`).
2. **Recommended configuration**: the same rule over {A, B, C, D0, D1, selected D2}; a tie still left after
   IDSW and MOTA goes to the simpler variant in that listed order. SORT is a baseline, not a candidate.
3. Only the selected D2 (plus A, B, C, D0, D1) is run on held-out.

Dev has only 35 S\* identity switches (1274 GT object-frames), so dev can distinguish large effects only; this
is stated, not worked around.

## Event classes (for every IDSW event py-motmetrics emits; classes sum to IDSW)

Let GT object `o` be matched to track `h_old` at its last scored match (frame `f0`) and to `h_new` at frame
`f`. `gap = f − f0`. Internal track state comes from the tracker dump (every track, every frame, including
coasting and tentative) — never inferred from the scorer's output.

- **A** consecutive (gap = 1) and `h_new` was *internally* unmatched on frame `f − 1` (dump `tsu ≥ 1`):
  a lost track took over an actively followed object.
- **B** gap > 1 and `h_new` existed before `f0` … i.e. was born at or before `f0`: after a gap, the object
  went to another pre-existing track.
- **C** gap > 1 and `h_new` was born after `f0`: after a gap, the object continues under a new identity.
- **D** everything else (e.g. gap = 1 with `h_new` matched on `f − 1` to something else — a swap between two
  active tracks — or `h_new` absent from the dump).

For gap events, the gap frames are split by what the data can support: GT scored-and-missed, GT label present
but outside the evaluated set (occluded > 1, too small, truncated), label absent; for scored-and-missed
frames, whether a detection of the object existed (IoU ≥ 0.5, any score, any depth validity) and whether it
had valid depth. KITTI's `occluded` field is reported as is; "occlusion" is not claimed where only a missed
detection is observed.

## Round 2 — closing the remaining gap to SORT (added 2026-09-30, before any run of these variants)

After round 1, the dev-recommended variant C (predicted-box IoU, max_age 30) still has 84 held-out IDSW against SORT's 60.
C and SORT still differ in (1) SORT's reporting rule — after a miss a track is not reported again until `min_hits`
*consecutive* matches (`reconfirm_after_miss`) —, (2) track lifetime (SORT max_age 1), and (3) box parameterisation /
SORT's first-`min_hits`-frames exception. (1) and (2) are tested here as a 2 × 6 grid, everything else as in C:

`reconfirm ∈ {0, 1}` × `max_age ∈ {1, 2, 3, 5, 10, 30}` (`R{r}_age{a}`; `R0_age30` ≡ C).

Selection on dev with the same rule as round 1 (max pooled IDF1; ties within 0.005 → fewer IDSW → higher MOTA → closest to
C: reconfirm off before on, then larger max_age). Diagnostic only, not candidates: SORT with max_age 30 on the same dev
detections. The selected variant, plus `R0_age1`, `R1_age30` and `R1_age1` to separate the two factors, is then run once on
held-out. Held-out remains the already-inspected set of round 1; (3) is not addressed in this round.

## Round 3 — SORT's box filter and assignment objective (added 2026-09-30, before any run of these variants)

Round 2 left SORT with max_age 30 (diagnostic) as the best held-out system, so the remaining difference is not lifetime or
reporting. Two further differences to C are tested: (a) `sort_box` — SORT's `[u, v, s, r]` box filter and constants instead of
`[cx, cy, w, h]`; (b) `max_total` — SORT's assignment objective (maximise total IoU, gate applied afterwards) instead of
"maximise the number of gated pairs, then cost". Grid on dev, everything else as in C (predicted-box IoU, max_age 30):

`reconfirm ∈ {0, 1}` × `sort_box ∈ {0, 1}` × `max_total ∈ {0, 1}` (`T{r}{b}{m}`; `T000` ≡ C).

Selection on dev with the same rule (max pooled IDF1; ties within 0.005 → fewer IDSW → higher MOTA → fewest factors changed,
then reconfirm, sort_box, max_total switched on last in that order). Held-out, run once: `T000`, `T010`, `T001`, `T011`
(the 2 × 2 with reconfirm off) plus the selected variant if it is not among them. **This is the last use of the held-out
sequences for this question**; any further change needs new data.
