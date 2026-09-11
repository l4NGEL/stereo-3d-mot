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
- [ ] KITTI tracking loader + MOTA / MOTP / IDF1 evaluation against real
      detector output -- benchmark_track is a synthetic proxy for this, not a
      replacement; this is the immediate next step
- [ ] connects to prior MOT / ReID work: association features beyond
      geometry (appearance embeddings) once a real detector/dataset is in the
      loop

## Phase 4 — Visual odometry

- [ ] feature extraction + matching (ORB), essential-matrix pose
- [ ] frame-to-frame pose, scale from the stereo baseline
- [ ] trajectory vs. dataset ground truth (ATE / RPE)
- [ ] touches the VI-SLAM / camera-pose requirements

## Phase 5 — Optimisation

- [ ] profile hotspots; tile + parallelise the matcher pre/post-processing
- [ ] optional CUDA / TensorRT inference path behind the same interfaces
- [ ] fixed frame-time budget with a report (target: end-to-end ≥ 20 FPS)
- [ ] memory + allocation audit

## Non-goals (for now)

- Full bundle adjustment / global SLAM back-end.
- Learned stereo (RAFT-Stereo etc.) — the ONNX seam would allow it later.
- Multi-camera (>2) rigs.
