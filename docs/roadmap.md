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

## Phase 5 — Appearance-aware association (ReID)

- [ ] a third association cue: appearance embedding distance (cosine/L2 on a
      ReID feature vector), fusable with the existing geometric cues as
      `C = α·C_3D + β·C_IoU + γ·C_ReID`
- [ ] draws on prior ReID / MOT17 / occlusion and ID-switch-forensics work --
      the natural place that experience plugs into this pipeline
- [ ] re-run the Phase 4 KITTI table with a third row; the interesting result
      isn't "ReID wins" in isolation, it's *how much* it helps once depth is
      already in the cost function
- [ ] do this only after Phase 4 has real baseline numbers to improve on

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
