# Roadmap

Phases are ordered so each one produces a demonstrable, tested capability and the
next one slots into an interface that already exists.

## Phase 1 — Stereo depth core  ✅ done

- [x] `CameraModel`, `StereoRig` (baseline, `doffs`, `Q`, triangulation)
- [x] `StereoMatcher` over `cv::StereoBM` / `cv::StereoSGBM`
- [x] disparity → metric depth, dense 3D reprojection, PLY export
- [x] `DepthMetrics` (RMSE, MAE, abs-rel, bad-N %, δ<1.25ᵏ, density)
- [x] `SyntheticStereoSource` with analytically exact ground truth
- [x] `MiddleburySource` (calib.txt + PFM ground truth)
- [x] `stereo_depth_demo`, `benchmark_depth`
- [x] GoogleTest suite, Dockerfile, docker-compose, Makefile, GitHub Actions CI

## Phase 2 — Detection in the loop  ✅ done

- [x] `OnnxDetector` implementing `Detector` (ONNX Runtime CPU, auto-detects the
      YOLOv8 `[1,4+nc,N]` and YOLOv5 `[1,N,5+nc]` output layouts)
- [x] `letterbox` pre-processing + inverse mapping, class-aware `nms`, COCO names
      — each a separately unit-tested unit
- [x] wired `detector → promoteTo3D` into `stereo_depth_demo` (3D box overlays,
      per-detection `Z` + camera-frame position, `detect` in the profiler)
- [x] `benchmark_detect` app: latency p50/p90/p99 + throughput
- [x] `scripts/export_yolov8.py` (Ultralytics → ONNX) — training/export in
      Python, inference in C++
- [x] hermetic test: `scripts/make_test_model.py` builds a tiny ONNX fixture so
      `test_onnx_detector` runs with no download and no real weights
- [x] optional build (`-DS3M_WITH_ONNX`): the geometry path never depends on ORT

## Phase 3 — 3D multi-object tracking

- [x] `Tracker` manager: predict → gate → associate (greedy / Hungarian) →
      update → birth/death/coasting, pluggable association strategy
- [x] `solveAssignmentHungarian` / `solveAssignmentGreedy` (`assignment.hpp`) --
      generic O(n^3) Kuhn-Munkres over a gated cost matrix, plus the greedy
      baseline, both unit-tested (incl. a hand-verified regression: an
      over-gate-but-finite cost must be exactly as forbidden as +inf)
- [x] Two interchangeable association strategies on the *same* Tracker:
      `kMahalanobis3D` (chi-square-gated squared Mahalanobis distance in the
      track's 3D state) vs `kIou2D` (1 - IoU on the last matched 2D box,
      the classic depth-blind baseline) -- `--track` / `configs/default.yaml`
      `tracking.association`
- [x] `benchmark_track`: the two strategies head-to-head (+ Hungarian vs
      greedy) on a hermetic synthetic scene engineered so a near (2 m) and a
      far (9 m) object's 2D boxes visibly cross -- ID switches, ID
      consistency, fragmentation, false-track rate. `TrackerTest.
      DepthSeparatesOccludingBoxesIou2DGetsItWrong` pins down the single-frame
      mechanism with exact, hand-verified numbers.
- [x] `TrackState` trajectory export: CSV and a PLY polyline
      (`trajectory_io.hpp`), wired into `stereo_depth_demo --track
      --trajectories/--trajectories-ply`

## Phase 4 — KITTI + MOTA/MOTP/IDF1  ✅ done

- [x] `MotAccumulator` (`mot_metrics.hpp`): CLEAR-MOT (Bernardin & Stiefelhagen,
      2008) + IDF1 (Ristani et al., 2016) -- MOTA, MOTP, IDF1, ID switches,
      fragmentation, precision/recall. Ported from a from-scratch Python
      implementation validated against **py-motmetrics** (the reference
      library) across 400+ randomised trials plus hand-built edge cases
      (gaps, id switches, crossings, gating) *before* writing the C++; that
      pass caught two real divergences from my first-draft algorithm (the
      previous-correspondence map must persist across any gap, not just reset
      each frame that an object is briefly absent; fragmentation is scoped to
      each object's own `[first match, last match]` span, not its whole
      presence window) -- see `tests/test_mot_metrics.cpp`.
- [x] `readKittiLabels` / `readKittiCalib` / `KittiTrackingSource`
      (`kitti_loader.hpp`): parses the tracking-benchmark label format (17
      fields, DontCare handling, tolerant of malformed lines) and calibration
      (`P2`/`P3` → `StereoRig`, including a non-zero `doffs` when the
      rectified principal points differ -- not assumed away), and reads
      `image_02`/`image_03` pairs as a `FrameSource`. Tested against a
      hand-built hermetic fixture (self-consistent with the documented
      rectified-`P` convention), and cross-checked against sequence 0000's
      real `calib/0000.txt`: the derived baseline comes out to 0.5327 m,
      matching KITTI's documented ~0.54 m rig baseline.
- [x] `benchmark_track` now also reports MOTA/MOTP/IDF1/IDSW/Frag (via the
      same `MotAccumulator`, scored against the synthetic scene's *exact*
      ground truth) alongside its own identity-preservation metrics -- one
      more end-to-end exercise of the accumulator before trusting it on real
      data.
- [x] `benchmark_kitti` app: runs the actual pipeline (stereo depth →
      detector → `promoteTo3D` → `Tracker`) on a KITTI sequence, once, per
      association method, and scores each against the sequence's ground
      truth -- the `2D IoU` vs `3D Mahalanobis` × `MOTA`/`IDF1`/`IDSW`/`Frag`
      table this phase is for.
- [x] `scripts/download_kitti.sh` (calib + labels always; images opt-in via
      `--with-images`, since the official archives bundle all sequences at
      ~15 GB each with no per-sequence download).
- [x] **Actually run it**, on sequence 0000 (154 frames). Downloaded via HTTP
      range requests against KITTI's S3 bucket instead of the ~15 GB/camera
      full archives (`scripts/fetch_kitti_sequence.py`: reads the zip central
      directory remotely, pulls only this sequence's ~265 MB) --
      `scripts/download_kitti.sh` now uses this path automatically when
      `requests` is available. Real result:
      `MOTA=-2.30 MOTP=1.00m IDF1=0.182 IDSW=33 Frag=11` (3D Mahalanobis) vs
      `MOTA=-2.14 MOTP=1.02m IDF1=0.187 IDSW=35 Frag=23` (2D IoU) --
      `configs/kitti.yaml` restricts the detector to road-relevant COCO
      classes and retunes `measurement_noise` for real (much larger) stereo
      error. Full numbers and the honest read on them (negative MOTA is a
      detector/GT domain-mismatch artifact, not a bug; the 2D/3D gap mostly
      closes on real noisy depth, unlike the clean synthetic-scene win) are
      in the README's [KITTI evaluation](../README.md#kitti-evaluation)
      section -- not repeated here to avoid the two copies drifting apart.
- [ ] connects to prior MOT / ReID work -- Phase 5.

## Phase 5 — Appearance-aware association (ReID)  ✅ done

- [x] A third association cue, fused with the existing geometric ones as
      `C = α·C_3D + β·C_IoU + γ·C_ReID` (`kFusedAppearance`, `tracker.hpp`),
      α/β/γ = 0.5/0.2/0.3 by default (`TrackerParams::fused_weight_*`).
      **Scope choice, stated up front**: the "ReID" cue is a classical HSV
      color histogram (`appearance.hpp`, Bhattacharyya distance), not a
      learned embedding -- no model file, export step, or new dependency,
      and histogram-based appearance matching is a real, long-established MOT
      technique for exactly this role (one fusable cue, not the sole
      association signal a real ReID network is designed to be). A learned
      embedding would very likely separate appearance better, especially
      between same-colored objects; that's future work, not something this
      phase claims to have done.
- [x] Gating: the *union* of the Mahalanobis and IoU gates, not their
      intersection -- deliberately looser than either alone, since the whole
      point of fusing in a second geometric cue plus appearance is staying
      robust when one cue (here: depth, per Phase 4's real-data finding) is
      unreliable. Gating on the intersection would just inherit whichever
      cue is currently worse.
- [x] Hand-verified unit test (`TrackerTest.
      FusedAppearanceRecoversIdentityMahalanobisAloneGetsWrong`): two tracks
      at close depths (2.00 m red, 2.10 m blue), a red disambiguating
      detection sitting geometrically *closer* to the wrong (blue) track --
      plain `kMahalanobis3D` picks blue, `kFusedAppearance` correctly picks
      red. Paired with a no-descriptor fallback test proving the neutral
      0.5 appearance cost doesn't change the ranking when nothing has a
      descriptor yet (i.e. fused degrades gracefully to plain geometry, it
      doesn't just add noise).
- [x] Weights set by reasoning + those unit tests, **not** tuned against
      either benchmark's real numbers below -- same discipline as every
      earlier phase's Python-first validation, applied here as
      validate-before-you-look-at-the-target-number instead.
- [x] Re-ran both Phase 3/4 benchmarks with the third row.
      `benchmark_track` (synthetic scene, cards have distinct colors by
      construction): fused **ties or beats** plain 3D Mahalanobis (0 ID
      switches vs 2, 100% vs 99.5% ID consistency, IDF1 0.989 tied) at the
      cost of a higher false-track rate (50% vs 25% -- the looser OR-gate
      lets through a false positive the tighter single gate would have
      rejected; an honest, explainable trade, not hidden).
- [x] **Not left at one KITTI sequence.** `benchmark_kitti --sweep` re-run on
      two more real sequences chosen for a different occlusion profile than
      0000 (30% occluded, dense) -- 0003 (49% occluded, sparse) and 0017
      (62% occluded, dense). Result is *not* "fusion wins": 0003's best IDF1
      is plain 2D IoU (0.301), 0017's is plain 3D Mahalanobis (0.486), and
      fusion is second-best on both. What holds across all three: **fusion
      is never the worst of the three methods on IDF1, IDSW, or
      fragmentation on any tested sequence**, and it's the outright winner on
      the one sequence (0000) where neither pure cue dominated. A hedge
      against not knowing which cue fails on a given scene, not a universal
      upgrade -- a materially more defensible claim than the single-sequence
      version of this result. Weight-sensitivity (5 settings) and a cue
      ablation (3D-only / 3D+IoU / 3D+appearance / full) both re-run per
      sequence too: the ranking above holds across the whole weight grid, and
      the ablation confirms both added cues pull real weight (with a stated
      caveat -- the union gate means "no IoU" still benefits from IoU's
      *gating*, just not its *cost*). Full numbers and interpretation in the
      README's [Phase 5](../README.md#phase-5--appearance-aware-reid-association)
      section.
- [x] **Found and fixed a real bug while building `--sweep`.** Running many
      association variants over one cached, reused frame set (instead of
      each method recomputing detections independently, as before) surfaced
      a `cv::Mat` aliasing bug: `Track`'s constructor shallow-copied its
      initial appearance descriptor, so a newly-born track's descriptor
      *aliased* the cached detection's buffer, and `correct()`'s in-place EMA
      blend silently corrupted that supposedly-read-only cached data for
      every *later* run() call in the same process. Caught by a determinism
      check (identical settings gave different results depending on what had
      already run first); fixed by `.clone()`-ing on ingestion
      (`track.cpp`); re-verified with a three-way cross-check that the same
      weight setting now agrees everywhere it's computed. Confirmed this did
      **not** affect any previously-reported number -- every earlier run
      (including this phase's original single-sequence result) recomputed
      detections fresh per call with nothing shared to alias.
- [x] This directly validates the motivation written into the Phase 4
      write-up before any of Phase 5 was built: real stereo depth noise
      erodes 3D Mahalanobis's synthetic-scene advantage, and appearance is a
      cue that doesn't degrade with depth noise the way Mahalanobis gating
      does. The three-sequence result narrows that to its defensible form:
      fusing it in doesn't always beat a single well-matched cue, but it
      reliably avoids the worst case.

## Phase 6 — Real-time optimisation

- [ ] profile hotspots (the existing `ProfileRegistry` sections plus detector
      pre/post-processing); tile + parallelise the matcher
- [ ] single-thread → multi-thread, then optionally CUDA / TensorRT behind
      the same `Detector` interface
- [ ] fixed frame-time budget with a report (target: end-to-end ≥ 20 FPS),
      broken down per stage (capture → preprocess → inference → postprocess →
      stereo → 3D → tracking → output), not just a single aggregate number
- [ ] memory + allocation audit

## Phase 7 — Visual odometry

- [ ] feature extraction + matching (ORB), essential-matrix pose
- [ ] frame-to-frame pose, scale from the stereo baseline
- [ ] trajectory vs. dataset ground truth (ATE / RPE)
- [ ] touches the VI-SLAM / camera-pose requirements

## Phase 8 — ROS2 integration (optional / bonus)

- [ ] `/stereo/left`, `/stereo/right` in; `/perception/detections`,
      `/perception/tracks`, `/perception/pointcloud` out
- [ ] only after Phases 4-7 give the perception stack itself something worth
      wrapping in a node graph

## Non-goals (for now)

- Full bundle adjustment / global SLAM back-end.
- Learned stereo (RAFT-Stereo etc.) — the ONNX seam would allow it later.
- Multi-camera (>2) rigs.
