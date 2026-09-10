# Architecture

## Goals

- A **modular** stereo perception pipeline where each stage is independently
  testable and swappable.
- **Metric** output (metres), not just pretty disparity maps.
- **Ground truth from day one**: a synthetic scene generator so the whole
  pipeline is verifiable offline, plus a real-dataset loader.
- **Linux-first, reproducible builds**: the Docker image is the source of truth
  for the toolchain; CI builds the same way.

## Layers

```
            ┌─────────────────────────────────────────────┐
  apps/     │  stereo_depth_demo   benchmark_depth         │
            │  benchmark_detect                            │
            └───────────────┬─────────────────────────────┘
                            │  uses
            ┌───────────────▼─────────────────────────────┐
  s3m::s3m  │  io  →  depth  →  geometry  →  detection  →  │
   library  │       ↑        ↖ camera ↗      tracking      │
            │     viz       core (types, config, timer)    │
            └───────────────┬─────────────────────────────┘
                            │  links
            ┌───────────────▼─────────────────────────────┐
            │  OpenCV 4 (core imgproc calib3d imgcodecs    │
            │  objdetect)   Eigen 3   ONNX Runtime (opt.)  │
            └─────────────────────────────────────────────┘
```

### `core`
- `types.hpp` — the vocabulary types that cross module boundaries:
  `StereoFrame`, `Detection2D`, `Detection3D`, `TrackState`.
- `Config` — plain structs + a `cv::FileStorage` YAML loader. Missing keys keep
  their defaults, so partial config files are fine.
- `Stopwatch`, `FpsMeter`, `ProfileRegistry` — timing. `ProfileRegistry::scope()`
  returns an RAII object that accumulates wall time per named section; the
  benchmark prints the table.

### `camera`
- `CameraModel` — pinhole intrinsics (`fx, fy, cx, cy`), `project` / `backProject`
  / `ray`, and `scaled()` for resized images. Distortion coefficients are stored
  for future undistortion but the projection helpers are the ideal model.
- `StereoRig` — two `CameraModel`s + baseline + `doffs`. Owns the depth ↔
  disparity relation, `triangulate()`, and the reprojection matrix `Q`. The
  single source of truth for stereo geometry.

### `depth`
- `StereoMatcher` — one interface over `cv::StereoBM` and `cv::StereoSGBM`.
  Returns `CV_32F` disparity in pixels (the internal `/16` fixed point undone),
  with unmatched pixels set to a sentinel.
- `DepthMetrics` / `evaluate()` — RMSE, MAE, abs-rel, sq-rel, "bad-N" pixel
  fractions at configurable thresholds, `δ < 1.25ᵏ` accuracy, and density. Works
  for both disparity maps (px) and depth maps (m); only pixels valid in both the
  estimate and the ground truth contribute.

### `geometry`
- `reproject()` — disparity + `Q` → `CV_32FC3` point cloud, plus a validity mask.
- `disparityToDepthMap()` — per-pixel `Z`.
- `robustDepthInRoi()` — median depth over a shrunk box, with a validity-fraction
  guard; the primitive behind 2D → 3D promotion.
- `promoteTo3D()` — `Detection2D` + depth map → `Detection3D` with a metric
  position at the box centre.

### `io`
- `FrameSource` — abstract iterator of `StereoFrame`s. `rig()`, `size()`,
  `next() -> optional<StereoFrame>`, `reset()`.
- `SyntheticStereoSource` — textured background plane + fronto-parallel "cards"
  at known depths. The right image is synthesised by depth-dependent horizontal
  shift, so ground-truth disparity/depth are analytically exact (away from
  occlusion borders). Cards drift between frames to exercise tracking.
- `MiddleburySource` — one Middlebury 2014 scene: parses `calib.txt`, reads
  `im0/im1.png` and `disp0.pfm`, rescales intrinsics if the images were
  downsized, and derives a metric ground-truth depth map.
- `readPfm` / `writePfm` — the float image format Middlebury ships GT in
  (endianness + bottom-to-top rows handled).
- `writePointCloudPly` — coloured ASCII PLY for MeshLab / CloudCompare.

### `detection`
- `Detector` interface; `NullDetector`; `HogPeopleDetector` (OpenCV HOG+SVM, no
  model file).
- `OnnxDetector` — ONNX Runtime (CPU EP), PIMPL so the ORT API never leaks into
  the rest of the tree. Auto-detects the output layout: YOLOv8 `[1, 4+nc, N]`
  (channels-first, no objectness) or YOLOv5 `[1, N, 5+nc]`. Built only when
  `-DS3M_WITH_ONNX=ON` **and** ORT is found; otherwise the target silently omits
  it and `S3M_WITH_ONNX` stays undefined.
- `letterbox` — aspect-preserving resize + centre pad, with `toOriginal()` to map
  detections back. `nms` / `nmsClassAware` — greedy NMS; the class-aware variant
  offsets boxes into per-class lanes so one pass suffices. `coco.hpp` — the 80
  class names. Each is a separately unit-tested unit (`test_letterbox`,
  `test_nms`, `test_onnx_detector`).

### `tracking` (Phase 3 seam)
- `KalmanFilter` — generic linear KF (dynamic Eigen matrices), plus
  `makeConstantVelocity3D(dt, accel_std, meas_std)` giving a 6-state
  `[x y z vx vy vz]` filter with a white-noise-acceleration `Q`.
- `Track` — wraps a KF with age / hits / misses / confirmation and a
  Mahalanobis `gatingDistanceSq()` for association.

### `viz`
- `colorizeDisparity`, `colorizeDepth` (invalid → black), `tile` (grid montage),
  `drawDetections` / `drawDetections3D`.

## Data flow (one frame)

```
FrameSource::next() ──► StereoFrame{ left, right, [gt_disparity, gt_depth] }
      │
      ├─ StereoMatcher::computeDisparity(left, right) ──► disparity  (CV_32F px)
      │
      ├─ disparityToDepthMap(disparity, rig) ───────────► depth      (CV_32F m)
      ├─ reproject(disparity, rig, &mask) ──────────────► cloud      (CV_32FC3)
      │
      ├─ detector.detect(left) ────────────────────────► {Detection2D}
      ├─ promoteTo3D(dets, depth, rig) ─────────────────► {Detection3D}
      └─ [Phase 3] tracker.update({Detection3D}) ───────► {TrackState}

  if gt present: evaluate(estimate, gt, …) ─────────────► DepthMetrics
```

## Testing strategy

- **Geometry** is checked against closed-form expectations
  (`test_camera_model`, `test_stereo_rig`, `test_reprojection`) — round-trips,
  known disparities, `Q`-vs-`triangulate` consistency.
- **Matcher** is checked with a deterministically shifted image pair: recovered
  disparity must equal the known shift (`test_stereo_matcher`).
- **Synthetic source** is checked for internal GT consistency and for the full
  matcher→depth path staying within generous error bounds (`test_synthetic_source`).
- **Detection** — `test_letterbox` (aspect ratio, padding, inverse-map
  round-trip), `test_nms` (suppression, thresholds, class-awareness),
  `test_onnx_detector` runs the real ORT session against a 99 KB hand-built ONNX
  fixture (`scripts/make_test_model.py`) with detections baked into the output
  tensor, so parse → NMS → letterbox inversion is verified end-to-end offline.
- **Metrics**, **PFM**, **config**, **Kalman/Track** have focused unit tests.
- `benchmark_depth` / `benchmark_detect` are the quantitative harnesses.
