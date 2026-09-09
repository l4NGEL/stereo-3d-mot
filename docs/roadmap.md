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

## Phase 2 — Detection in the loop

- [ ] `OnnxDetector` implementing `Detector` (ONNX Runtime, YOLO-style export)
- [ ] letterbox pre-processing, NMS, class map
- [ ] wire `detector → promoteTo3D` into `stereo_depth_demo` with overlays
- [ ] a Python `scripts/export_model.py` (train/convert → ONNX) — Python for
      training, C++ for inference
- [ ] detection latency added to the profiler report

## Phase 3 — 3D multi-object tracking

- [ ] `Tracker` manager: predict → gate (Mahalanobis) → associate
      (greedy / Hungarian) → update → birth/death
- [ ] `TrackState` trajectory export (CSV / PLY polyline)
- [ ] KITTI tracking loader + MOTA / MOTP / IDF1 evaluation
- [ ] occlusion handling / coasting on missed detections
- [ ] connects to prior MOT / ReID work: association features beyond IoU

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
