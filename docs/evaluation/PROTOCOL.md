# Evaluation protocol and acceptance criteria

**Status: frozen by the git commit that adds this file** (`git log -- docs/evaluation/PROTOCOL.md`).
It was written and committed *before* any measurement on the held-out data and before any new
metric code was run. Later changes are appended to §10 (Deviations log) with a reason; the text
above §10 is not edited afterwards.

The question this evaluation answers is not "is the project successful?" as one score. It is: for each of
stereo depth, detection, tracking, visual odometry and speed, **what was measured, on which data, with which
command, against which pre-stated target, and what was not measured at all.**

---

## 1. Disclosure: what the author already knew

I (the author of this protocol) had already seen these numbers from earlier phases when I wrote the
thresholds below, so I cannot claim to be blind to them:

- KITTI tracking 0000 / 0003 / 0017 under the *legacy* protocol (§3.2): MOTA −2.30 / IDF1 0.18 on 0000,
  IDF1 0.18–0.49 across the three, negative MOTA on all three.
- KITTI odometry 04: ATE 15.7 m, RPE 0.66 m per frame.
- Speed: 7.8 FPS compute-only, ~4.2 FPS with I/O on a Docker bind mount.
- Synthetic depth: disparity RMSE ~1.03–1.19 px, bad-2.0 ~0.7–0.85 %.

Mitigation: every threshold in §7 carries an explicit rationale that does not reference those numbers
(a repo-defined target, a budget derived from another target, the system's own configured assumption, or a
declared engineering judgement). The held-out data in §2 had not been downloaded when the thresholds were
written, so no threshold can have been fitted to it.

## 2. Data, splits and leakage controls

### 2.1 Tracking (KITTI tracking benchmark, `label_02`, left colour camera + right colour camera)

| Role | Sequences | Frames used | Notes |
|---|---|---|---|
| **Dev** (design and selection allowed) | 0000, 0003, 0017 | all (154 / 144 / 145) | Already used in Phases 4–6: `measurement_noise` 0.05→1.0 and `keep_classes` were chosen after looking at 0000. They are *not* clean test data and are labelled dev everywhere. |
| **Held-out test** (no design, no selection) | 0001, 0002, 0004, 0007, 0008, 0010, 0012, 0016 | first 150 (0012: all 78) | Chosen by a seeded draw: `numpy.random.RandomState(20260926).choice(<18 non-dev training sequences>, 8, replace=False)`, before any of them was downloaded or looked at. |

KITTI's 29 test-split sequences have no public labels and are not used.

### 2.2 Visual odometry (KITTI odometry benchmark, grayscale pair `image_0` / `image_1`, poses `poses/NN.txt`)

| Role | Sequences | Frames |
|---|---|---|
| Dev | 04 | all 271 (the only sequence used while VO was developed) |
| Held-out test | 00, 06, 07, 08, 10 | first 500 each (same seeded draw, 5 from the 10 non-dev sequences that have ground truth) |

Prefix-only runs cover roughly 200–500 m per sequence, so the official 100–800 m segment metric is computed for the
segment lengths that fit (100–400 m). This is stated wherever the number appears; it is not the full-sequence
leaderboard metric.

### 2.3 Stereo

- **Synthetic** (`SyntheticStereoSource`, exact ground truth): seeds 42, 1, 2, 3, 4; 30 frames each.
- **Real**: KITTI Stereo 2015 training set (`data_scene_flow.zip`): 200 pairs, frame `_10` only, ground truth
  `disp_occ_0` (all pixels with GT) and `disp_noc_0` (non-occluded subset). Nothing in this project was ever tuned on it.
- **Object-level, real**: the tracking sequences above, against KITTI's own 3D box labels (§3.5).

### 2.4 Leakage controls

- **Detector.** `models/yolov8n.onnx` is an Ultralytics YOLOv8n exported to ONNX (`scripts/export_yolov8.py`).
  This project never trains or fine-tunes it. Its training data is COCO; KITTI is not part of COCO. The inventory
  records the file hash so a swapped model would be detectable. No train/test split exists *inside* this project
  because there is no training here; the only leakage channel is hyperparameter selection, controlled below.
- **Hyperparameters.** Anything selected is selected on dev only (§4), the selection is committed before the held-out
  frontend is run, and the held-out sequences are never used to choose, only to report.
- **Previously seen choices** (`measurement_noise`, `keep_classes`, `fused_weight_*`, `gating_chi2`, `iou_gate`,
  `num_tiles`) are kept fixed and are declared as dev-influenced in the inventory. `fused_weight_*`
  (0.5 / 0.2 / 0.3) and the gates were fixed by reasoning before the KITTI numbers existed (per the Phase 5 commit
  history); `measurement_noise` and `keep_classes` were not.

## 3. Protocol definitions

### 3.1 Class mapping and ground-truth categories (used by every KITTI tracking / detection metric below)

| KITTI label type | Vehicle evaluation | Person evaluation |
|---|---|---|
| Car, Van, Truck | valid class | other class (not considered) |
| Pedestrian | other class | valid class |
| Person_sitting, Cyclist | not considered | **neutral** |
| Tram, Misc | **neutral** | **neutral** |
| DontCare | ignore region | ignore region |

Detector/tracker COCO class → evaluation class: `car`(2), `bus`(5), `truck`(7) → Vehicle; `person`(0) → Person;
`bicycle`(1), `motorcycle`(3) → two-wheeler, **excluded from both evaluations** (never counted as TP or FP). The
project's `kitti.yaml` keeps these classes; under the strict legacy protocol they are false positives, under the
KITTI-like protocol they are ignored. That difference is part of what is being measured (H-scope, §6).

A valid-class GT object is **evaluable** if it passes the KITTI object-detection benchmark's *moderate* difficulty
filter: box height ≥ 25 px, `occluded` ≤ 1, `truncated` ≤ 0.30. Valid-class GT that fails the filter is **ignored**
(a match to it is neither TP nor FP; if unmatched it is not an FN).

This is a "KITTI-like" protocol assembled from the published detection-benchmark difficulty definitions. I did not
consult or verify the official tracking devkit's exact ignore rules, so its numbers are **not** comparable to the
KITTI tracking leaderboard.

### 3.2 P-LEGACY (reproduces the numbers in README and roadmap)

Exactly what `apps/benchmark_kitti.cpp` does: every non-DontCare GT object (all types, all sizes, all occlusion levels)
versus every *confirmed* track (coasting tracks included, up to `max_age` = 30 frames), class-agnostic, matched by 3D
Euclidean distance ≤ 2.0 m between the GT box centre (`location.y − h/2`) and the track's filtered position,
CLEAR-MOT + IDF1 via `MotAccumulator`. Its role: (a) regression check that the new harness reproduces the published
numbers, (b) the strict reference the KITTI-like protocol is compared against.

### 3.3 P-K2D (primary tracking protocol)

Per sequence, per evaluation class (Vehicle, Person) separately, per frame:

1. **GT set**: evaluable GT of that class (§3.1).
2. **Hypothesis set**: tracks of that class group with `confirmed` and `time_since_update ≤ E`, where `E` is the emission
   policy (§4). A track with `time_since_update = 0` uses the matched detection's box exactly. A coasting track's box is
   the projection of its Kalman-predicted 3D position with the last matched box's size scaled by `z_last / z_now`.
3. **Pre-filter**: drop a hypothesis if it cannot be a TP (no evaluable GT box with IoU ≥ 0.5) **and** any of: IoU ≥ 0.5
   with an ignored or neutral GT box; intersection-over-hypothesis-area ≥ 0.5 with a DontCare box; box height < 25 px.
4. **Matching**: CLEAR-MOT with distance `1 − IoU`, match threshold IoU ≥ 0.5, previous correspondence kept when still
   valid (py-motmetrics semantics), MOTA / IDF1 / IDSW / FP / FN as usual.
5. **Aggregation**: raw counts (GT objects, FP, FN, IDSW, IDTP, IDFP, IDFN) are **summed** across sequences and classes,
   and the ratios are computed from the sums. Per-sequence MOTA values are never averaged into a "total MOTA".
   Vehicle and Person are reported separately and pooled.
6. **Also reported**: mean IoU of matches, and the mean 3D distance between matched track position and GT box centre
   (with the signed depth bias), because localisation quality in metres is a separate question from identity.

### 3.4 Detection metrics

Detections are the `Detector` output after class-aware NMS (0.45) with a score floor of 0.01 (so the whole PR curve is
available). Per class group, KITTI-style greedy matching in descending score order at a given IoU threshold: a detection
is a TP if it matches an unmatched evaluable GT, a duplicate of an already-matched GT counts as a FP, and is **ignored**
if its best match is an ignored or neutral GT, it lies in a DontCare region (§3.3 step 3) or it is shorter than 25 px.

- **Operating point**: score ≥ 0.25 (the shipped threshold): precision, recall, F1, TP/FP/FN.
- **AP**: 40-point interpolated (KITTI R40, recall 1/40 … 1) and all-point area. Vehicle at IoU 0.5 and 0.7, Person at 0.5.
- COCO mAP is **not** measured in this project. Published COCO mAP for YOLOv8n is not a KITTI number and is not used
  as one.

### 3.5 Depth metrics

- **Synthetic** (`benchmark_depth`, project ROI = image with the left `num_disparities` columns cropped): disparity
  RMSE, bad-1/2/4, density, depth RMSE/abs-rel. Additionally computed by an independent script from dumped disparity maps:
  the same errors **over the full image with missing estimates counted as errors** (the project's density figure hides
  that). Depth-binned by the known card depths.
- **KITTI Stereo 2015** (real, independent script over dumped disparity PNGs): **D1** error = pixel with
  |d − d_gt| > 3 px **and** > 5 % of d_gt (Menze & Geiger 2015). Reported as (i) D1-all with pixels lacking an estimate
  counted as errors (primary), (ii) D1 over pixels that have an estimate, (iii) the same on the non-occluded GT,
  plus end-point error, bad-1/2/3/5, density, and GT-depth bins (0–10, 10–20, 20–40, > 40 m). A baseline run of the
  project's own `type: BM` matcher (block 15, 128 disparities, other parameters as in `configs/default.yaml`) uses the same rules.
- **Object-level depth, real**: for each evaluable Vehicle/Person GT matched (IoU ≥ 0.5, best IoU) to a detection,
  compare the detection's promoted depth (median in the shrunken box, what the tracker actually sees) to the GT's
  **front-surface depth** — the ray through the detection-box centre intersected with the label's 3D cuboid — and also to the
  cuboid centre depth. Pairs whose ray misses the cuboid are counted and skipped, detections without valid depth are
  counted as missing. Bins: 5–15, 15–30, 30–50 m (by GT depth). KITTI's boxes come from LiDAR annotation, so this is a
  real-scene, object-level ground truth; it is not per-pixel and says nothing about textureless regions elsewhere.

### 3.6 Visual odometry metrics

- **KITTI segment metric** (Geiger et al. 2012): start frames every 10 frames, segment lengths 100, 200, 300, 400 m along
  the ground-truth path (those that fit), translation error / length (%) and rotation error / length (deg/m) from the relative
  pose difference, averaged over all segments. Per sequence and pooled over all held-out segments.
- **ATE** (rigid Kabsch/Umeyama alignment, no scale, and for reference with scale), **RPE** at 1, 10, 50 frames
  (Sturm et al. 2012), path length, ATE as a fraction of path length. Computed by an independent script from dumped full
  poses and cross-checked against the C++ `evaluateTrajectory`.
- Fraction of frames tracked (`ok`) vs coasted, mean matches per frame, recovered speed against ground-truth speed.

### 3.7 Speed methodology

- Sequence pipeline, frame loop strictly sequential: left+right `imread` → SGBM (tiled) → depth → detector →
  `promoteTo3D` + appearance → tracker update. Frames are 1242×375 (KITTI tracking) — the resolution is stated because it
  drives the cost.
- Build: CMake `Release`, Ubuntu 22.04 container (OpenCV 4.5.4, ONNX Runtime 1.19.2 CPU, all logical cores), on this
  machine (Intel i9-9880H 8C/16T laptop CPU, 32 GB, Windows 10 + Docker Desktop). The GPU is not used by the default path.
- **With I/O** = image files read from container-local storage (data copied into the container first), so the
  Docker-Desktop bind-mount penalty documented in Phase 6 is not in the number. The bind-mount figure is reported once for
  continuity and labelled as such. **Without I/O** = the same loop with decode excluded.
- Each run discards the first 10 frames as warm-up; 3 repetitions over dev sequences 0000, 0003, 0017; per-frame wall times
  are kept, from which median, mean, p95, p99, max and FPS (= 1000 / mean latency; median-based FPS also shown) are computed.
  System load and CPU clock are recorded before and after; results taken while another heavy job is running are discarded and
  re-run (Phase 6 showed a 3× distortion from exactly that).

### 3.8 Statistics

Sequences are the resampling unit. Differences between two systems are paired: per-sequence raw counts are resampled with
replacement (10,000 draws, seed 0), pooled metrics recomputed from the resampled sums, 95 % percentile interval reported,
plus the number of sequences on which each system wins. With 8 held-out sequences these intervals are wide, and that is the
honest width. A claimed improvement is "demonstrated" only if the interval excludes 0; otherwise it is "not demonstrated".

## 4. Systems under test and selection on dev

**Fixed for all systems** (not tuned here): `configs/kitti_tiled.yaml` — SGBM 128 disparities, block 5, `num_tiles` 4;
YOLOv8n at 640, NMS 0.45, `keep_classes` {0,1,2,3,5,7}; `measurement_noise` 1.0, `process_noise` 1.0, `max_age` 30,
`gating_chi2` 7.815, `iou_gate` 0.3, Hungarian assignment, fused weights 0.5 / 0.2 / 0.3.

**As shipped** (three systems, reported as-is): association ∈ {Mahalanobis-3D, IoU-2D, fused}, score threshold 0.25,
`min_hits` 3, emission `E` = 30 (every confirmed track, coasting included).

**Dev-selected** (one system): selected by exhaustive search over

- association ∈ {mahalanobis3d, iou2d, fused}
- emission `E` ∈ {0, 2, 5, 30} (0 = only tracks updated this frame; the usual SORT/AB3DMOT reporting rule)
- detector score threshold ∈ {0.25, 0.35, 0.50}
- `min_hits` ∈ {1, 2, 3}

using **only the three dev sequences**, criterion = pooled P-K2D MOTA (Vehicle + Person), ties within 0.005 broken
toward the configuration closest to shipped. The selection and its evidence are committed before the held-out frontend
is run. The held-out sequences then evaluate: the three as-shipped systems, the dev-selected system, the baseline and
the ablations of §5 — and nothing else is chosen from them.

## 5. Baseline and ablation matrix

Baseline **B1 — SORT-2D**: Bewley et al. 2016 reference algorithm (image-plane Kalman filter on box centre/scale/aspect,
IoU association at 0.3, `max_age` 1, `min_hits` 3, Hungarian), class-aware for comparability, fed the same detections
(same detector, same score threshold as the dev-selected system, and also at 0.25), **no stereo, no depth**, evaluated by
P-K2D only (it has no 3D output). Written from the published algorithm, not taken from a tracker package.

Ablations, each run against the dev-selected system on the held-out set (accuracy = P-K2D pooled MOTA / IDF1 / IDSW / FP / FN and
P-LEGACY; cost = per-frame milliseconds of the stage the variant changes):

| ID | Variant | Isolates |
|---|---|---|
| A1 | appearance cost off (γ = 0) | HSV-histogram cue |
| A2 | IoU cost off (β = 0) | 2D-overlap cue in the fused cost |
| A3 | 3D cost off (α = 0) | stereo geometry in the association cost |
| A4 | plain Mahalanobis-3D | fusion vs a single 3D cue |
| A5 | plain IoU-2D | fusion vs a single 2D cue |
| A6 | B1 SORT-2D | stereo/3D as a whole |
| A7 | greedy instead of Hungarian | assignment solver |
| A8 | emission `E` = 30 | the coasting-output rule |
| A9 | `num_tiles` 1 | tiling: accuracy neutrality and speed |
| A10 | CUDA detector | GPU: speed and output equality (GPU image, subset of sequences) |

Extra stage timers (`promote3d`, `appearance`, `track`, `stereo`) are added so each variant's cost is measured, not inferred.

## 6. Hypotheses about the negative MOTA, and how each is tested

The README explains the negative MOTA as "KITTI labels are not exhaustive". That is one hypothesis among several found by
reading the evaluation code. Each is stated with a falsification rule *before* looking.

| ID | Hypothesis | Test | Counts as supported if |
|---|---|---|---|
| H-label | Real objects that the detector correctly finds are simply not labelled | Manual review of unmatched detections in top-FP frames (below) | A + C categories ≥ 50 % of reviewed non-ghost FPs (25–50 % = partial, < 25 % = not supported) |
| H-ghost | Coasting confirmed tracks (up to 30 frames) inflate FP | P-LEGACY re-scored with `E` = 0 | FP falls ≥ 30 % |
| H-scope | Unevaluated categories (DontCare, small/occluded/truncated GT, neutral types, two-wheelers, class mixing) inflate FP and FN | P-LEGACY vs P-K2D with the ignore rules switched on | FP falls ≥ 20 % from the rules alone |
| H-bias | Median-in-box depth is on the visible surface while the GT centre is inside the object, so long vehicles fall outside the 2 m gate | Signed depth error (estimate − GT centre) for matched Vehicle pairs; P-LEGACY recall with gate 2 m vs 4 m | median bias ≤ −0.5 m **and** recall gains ≥ 5 points at 4 m |
| H-depth | Detections lose their depth (`valid = false`: left 128 columns, range < ~3 m, low texture) and the tracker never sees them | Stage attribution of every FN (below), especially 0003 | ≥ 10 % of FNs in a sequence are "detected, no depth" |

**FN stage attribution (automatic).** Each evaluable GT frame that ends as an FN is assigned to the first stage that lost it:
(1) no detection with IoU ≥ 0.3 at score ≥ 0.01; (2) a detection exists only below the operating score; (3) detected but of
an unmapped/incompatible class; (4) detected, `valid = false` (no depth); (5) valid but the track was not yet confirmed or
had been lost; (6) tracked but the 3D distance exceeded the gate (legacy only). The 0003 low-recall question is answered
with this table plus a visual review of its category-(1) misses.

**FP/FN visual review procedure.** Frames are picked by a rule, not by eye: the 12 frames with the most FP and the 12 with
the most FN (pooled, P-LEGACY), plus the 8 frames of 0003 with the most FN. For each, an image shows GT boxes (by type,
ignored ones dashed), DontCare regions and numbered unmatched detections. At most 12 unmatched detections per frame are
labelled, in descending score order, into:

- **A — visible, unlabelled real object**: the box tightly encloses a clearly recognisable object of the predicted class and no
  GT box or DontCare region covers it.
- **B — true false alarm**: the box does not enclose an object of that class (background, pole, sign, shadow, wrong object),
  or is a duplicate box on an object another detection already covers.
- **C — label-convention or class-mapping issue**: a real, labelled object that the protocol does not credit (Cyclist,
  Person_sitting, Tram, Misc, Van/Truck confusion, a person or part-object box on a labelled group).
- **D — ambiguous**: cannot be decided at this size/occlusion.

The labeller is a single reviewer (the author, an AI model) with no second annotator, so the distribution is one person's
judgement, not an agreement statistic. The images and the label table are committed so the review can be audited or redone.
Reported: how many frames and how many detections were reviewed, the category distribution with Wilson 95 % intervals, and the
counterfactual MOTA obtained by not charging A + C false positives. Review is done on dev sequences for diagnosis and repeated
on a held-out sample only to confirm the distribution; no rule is changed after looking at held-out images.

## 7. Acceptance criteria

Evidence source: **R** repo-defined target · **D** derived from another target by stated arithmetic · **S** the system's own
configured assumption · **J** engineering judgement (declared, not a standard). Tiers: *minimum* (must) and *target* (should).
The dev-selected system is the one judged unless "as shipped" is stated. Everything is for **this machine and 1242×375 KITTI frames**.

### Speed
| ID | Criterion | Src | Rationale |
|---|---|---|---|
| SP-1 | End-to-end ≥ 20 FPS with I/O from container-local storage (1000 / mean latency), and ≥ 20 in each of the 3 repetitions | R | `docs/roadmap.md` Phase 1: "end-to-end ≥ 20 FPS" |
| SP-2 | p95 latency ≤ 50 ms in every repetition | D | 20 FPS = 50 ms budget; a real-time claim has to hold on the tail, not only on average |
| SP-3 | Same as SP-1 excluding image decode | R | reported separately because Phase 6 excluded I/O |

### Detection (IoU 0.5, moderate, score ≥ 0.25, held-out pooled)
| ID | Criterion | Src | Rationale |
|---|---|---|---|
| DE-1 | Vehicle precision ≥ 0.80 **and** recall ≥ 0.70 | D | with FN = 1−R and FP = R(1−P)/P, MOTA ≥ 0.50 (TR-2) needs (P, R) ≈ (0.80, 0.68) even with zero identity errors and no confirmation delay; this is a necessary budget |
| DE-2 | Vehicle AP50 (R40) ≥ 0.56 | D | area under the PR curve up to any point on it is ≥ P·R = 0.80 × 0.70; necessary, not sufficient |
| DE-3 | Person precision ≥ 0.75 **and** recall ≥ 0.50; AP50 (R40) ≥ 0.375 | D | same arithmetic for MOTA ≥ 0.30 (TR-3) |
| DE-4 | COCO mAP | — | **not measured in this project; no verdict.** Only KITTI-protocol numbers above are claimed |

### Tracking (P-K2D, held-out, pooled from raw counts)
| ID | Criterion | Src | Rationale |
|---|---|---|---|
| TR-1 | *minimum*: pooled MOTA > 0 | J | MOTA ≤ 0 means more errors than objects |
| TR-2 | *target*: Vehicle MOTA ≥ 0.50 and IDF1 ≥ 0.60 | J | the level at which a classical tracking-by-detection system is generally considered usable on KITTI cars |
| TR-3 | *target*: Person MOTA ≥ 0.30 and IDF1 ≥ 0.50 | J | pedestrians are smaller and more occluded than cars in KITTI |
| TR-4 | Identity stability: IDSW ≤ 2 % of GT objects (pooled) | J | ≤ 1 switch per 50 object-frames |
| TR-5 | Beats B1 SORT-2D on pooled MOTA **and** IDF1 with the paired-bootstrap interval excluding 0 | D | "stereo/3D helps" is claimed only against a depth-free baseline on identical detections |
| TR-6 | Metric integrity: the harness reproduces the published P-LEGACY counts on dev (|ΔMOTA|, |ΔIDF1| ≤ 0.001), and C++ `MotAccumulator`, py-motmetrics and an independent reference implementation agree exactly on FP/FN/IDSW/GT | J | otherwise none of the numbers can be trusted |
| TR-7 | Ablation claims: each contribution in §5 is called "demonstrated" only under §3.8 | D | |

### Visual odometry (KITTI segment metric, held-out pooled, segments 100–400 m)
| ID | Criterion | Src | Rationale |
|---|---|---|---|
| VO-1 | *minimum* t_rel ≤ 5 %; *target* ≤ 2.5 % | J | frame-to-frame stereo VO without bundle adjustment is usually reported in the low single-digit %; 2.5 % is the level of well-tuned classical stereo VO (published ~2–3 %, not re-verified here) |
| VO-2 | *minimum* r_rel ≤ 0.03 deg/m; *target* ≤ 0.015 deg/m | J | same reasoning |
| VO-3 | ≥ 95 % of frames `ok` (not coasted) on every held-out sequence | J | a VO that coasts often is not tracking |
| VO-4 | No held-out sequence with t_rel > 10 % | J | catches a catastrophic failure hidden by a good average |

### Stereo depth
| ID | Criterion | Src | Rationale |
|---|---|---|---|
| ST-1 | Synthetic, project ROI, 5 seeds: bad-2.0 ≤ 2 %, disparity RMSE ≤ 1.5 px, density ≥ 95 % | J | a **regression gate** on a benign scene whose right image is a shifted copy of the left; low evidential value, not a real-scene claim |
| ST-2 | KITTI Stereo 2015: D1-all (missing = error) ≤ 25 % **and** better than the BM baseline under the same rules | J | a quarter of pixels wrong at 3 px / 5 % is still workable for median-in-box depth; SGBM must at least beat block matching |
| ST-3 | Object-level: for evaluable Vehicles at 5–30 m, RMSE of (detection depth − GT front-surface depth) ≤ 1.0 m | S | `configs/kitti.yaml` sets `measurement_noise` = 1.0 m; if the measured error is larger, the Kalman filter's own assumption is violated |
| ST-4 | Tiling neutrality: |D1-all(4 tiles) − D1-all(untiled)| ≤ 0.5 pp on Stereo 2015 | J | the optimisation must not cost accuracy |

### Integrity and reproducibility
| ID | Criterion | Src |
|---|---|---|
| IN-1 | Every headline number regenerates from one documented command on the documented data | J |
| IN-2 | Determinism: two identical runs give bit-identical tracker dumps (timing excluded) | J |
| IN-3 | No leakage: the detector is unmodified pretrained weights (hash recorded), every selection used dev only, held-out never used to choose | J |

## 8. Verdict vocabulary

For each ID: **GEÇTİ** (measured, meets it), **KALDI** (measured, fails it), **ÖLÇÜLEMEDİ** (data or capability missing —
with what is missing and how to obtain it), **KANITLANMADI** (measured, the effect is inside the noise; used for relative
claims). A "GEÇTİ" for a *minimum* tier does not imply the *target* tier. Where a criterion passes only for the dev-selected
configuration and not as shipped, both are stated.

## 9. Scope limits, stated in advance

- Numbers are for this hardware; nothing here says anything about an embedded or target platform.
- Held-out tracking sequences are the first 150 frames only; identities that start later are not scored.
- One detector (YOLOv8n, COCO-pretrained) at one input size; a fine-tuned or larger model is a different system.
- Appearance cue is an HSV histogram, not a learned ReID embedding.
- The protocol is KITTI-like, not an official-leaderboard reproduction.
- The manual review is single-annotator.

## 10. Deviations log

*(appended after the freeze; each entry gives the date, what changed, why, and whether any result had been seen)*
