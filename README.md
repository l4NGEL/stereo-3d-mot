# stereo-3d-mot

**Real-time stereo vision pipeline in modern C++: rectified stereo → disparity → metric depth → 3D reprojection → (multi-object) tracking.**

Built as a from-scratch C++17 systems project on top of OpenCV and Eigen, developed
Linux-first inside Docker, with unit tests, a quantitative benchmark, and CI.

[![ci](https://github.com/l4NGEL/stereo-3d-mot/actions/workflows/ci.yml/badge.svg)](https://github.com/l4NGEL/stereo-3d-mot/actions/workflows/ci.yml)

> Status: **Phases 1–2 complete.** Calibrated stereo geometry, BM/SGBM disparity,
> metric depth, dense 3D reprojection + PLY export, a synthetic scene with exact
> ground truth, a Middlebury 2014 loader, a depth-accuracy benchmark — and an
> ONNX Runtime detector (YOLOv8/v5) wired through `promoteTo3D` so detections
> come out with a metric `Z` and a camera-frame position. 50+ unit tests pass;
> all three apps run end to end. 3D Kalman tracking is interfaced and tested;
> the tracker manager is Phase 3. See [docs/roadmap.md](docs/roadmap.md).

---

## Pipeline

```
        left camera        right camera
             |                  |
             +--------+---------+
                      v
             rectified stereo pair            (StereoFrame)
                      v
          StereoMatcher   (BM / SGBM)         -> disparity map  [px]
                      v
          disparity -> depth                  Z = f * B / (d + doffs)
                      v
          reprojection (Q matrix)             -> point cloud  (X,Y,Z) [m]
                      v
   +------------------+------------------+
   v                                     v
 detector (2D)  ----> promote to 3D ---> 3D constant-velocity
 HOG / ONNX           (median Z in box)  Kalman tracks*  (track id, X,Y,Z, v)
 (YOLOv8 / v5)

 * Phase 3 — the Kalman filter + Track are implemented and tested; the
   multi-object tracker manager (gating + association) is next.
```

Everything except the tracker manager is implemented and covered by tests.

## Why this project

It pairs a real computer-vision pipeline with the C++ / build / tooling side that
job specs in robotics and defense perception ask for:

| Requirement                        | Where it lives                                             |
| ---------------------------------- | --------------------------------------------------------- |
| Stereo / mono depth estimation     | `depth/`, `camera/`, `geometry/`                          |
| Geometric computer vision          | `StereoRig` (Q matrix, triangulation), `CameraModel`      |
| Deep model inference in C++        | `detection/` (ONNX Runtime, `OnnxDetector`)               |
| C++17, OpenCV, Eigen, ONNX Runtime | throughout                                                |
| Linux, CMake, Git, Docker          | `Dockerfile`, `docker-compose.yml`, `Makefile`, CMake     |
| Unit testing & verification        | `tests/` (GoogleTest), `benchmark_depth`, `benchmark_detect` |
| CI                                 | `.github/workflows/ci.yml`                                |
| Real-time / performance mindset    | `ProfileRegistry`, `FpsMeter`, latency percentiles        |

## Quick start (Docker — recommended)

No local C++ toolchain needed; you build in the same Ubuntu image CI uses. Each
target builds the image and compiles on first run, then reuses both.

```bash
make test     # build the image, compile, run the unit tests (ctest)
make demo     # end-to-end demo on the synthetic scene -> ./out/ (image board + PLY per frame)
make bench    # quantitative depth benchmark on the synthetic scene (exact ground truth)
make shell    # interactive shell in the dev image, source bind-mounted
```

Docker Desktop must be running.

## Quick start (native Linux)

```bash
sudo apt-get install -y build-essential cmake ninja-build pkg-config \
    libopencv-dev libeigen3-dev libgtest-dev

# optional: ONNX Runtime for the OnnxDetector (prebuilt CPU release)
ORT=1.19.2
curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT}/onnxruntime-linux-x64-${ORT}.tgz" \
  | sudo tar -xz -C /opt --one-top-level=onnxruntime --strip-components=1

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DONNXRUNTIME_ROOT=/opt/onnxruntime           # omit to build without ONNX
cmake --build build --parallel
LD_LIBRARY_PATH=/opt/onnxruntime/lib ctest --test-dir build --output-on-failure

./build/apps/stereo_depth_demo --source synthetic --out out --cloud
./build/apps/benchmark_depth   --source synthetic
```

Only `core imgproc calib3d imgcodecs objdetect` are actually linked, so the
individual `libopencv-<module>-dev` packages work too; CMake also accepts a
pkg-config `opencv4` if the CMake package is absent. `-DS3M_WITH_ONNX=OFF`
builds the geometry pipeline with no ONNX dependency at all.

Windows: use the Docker workflow, or WSL2 + the native steps above.

## Running on Middlebury 2014

```bash
# grab one scene (im0.png, im1.png, calib.txt, disp0.pfm)
scripts/download_middlebury.sh Motorcycle        # -> data/middlebury/Motorcycle

./build/apps/benchmark_depth --source data/middlebury/Motorcycle
./build/apps/stereo_depth_demo --source data/middlebury/Motorcycle --out out --cloud
```

The loader parses `calib.txt` (per-view `cx`, `doffs`, `baseline` in mm), reads
the `.pfm` ground-truth disparity, and converts it to a metric depth map so both
disparity and depth accuracy are reported. See [data/README.md](data/README.md).

## Detection → 3D

`OnnxDetector` runs a YOLO-family ONNX model on the left image (ONNX Runtime,
CPU). It auto-detects the output layout — YOLOv8 `[1, 4+nc, N]` or YOLOv5
`[1, N, 5+nc]` — does letterbox pre-processing, class-aware NMS, and maps boxes
back to original pixels. Each surviving box is promoted to 3D with the median
depth inside it (`promoteTo3D`), giving a metric `Z` and an `(X, Y, Z)` position
in the left-camera frame.

```bash
# training / export stays in Python, inference is C++
pip install ultralytics
python scripts/export_yolov8.py --weights yolov8n.pt          # -> models/yolov8n.onnx

./build/apps/stereo_depth_demo --source synthetic --detector onnx \
    --model models/yolov8n.onnx --out out
./build/apps/benchmark_detect  --detector onnx --model models/yolov8n.onnx --frames 50
```

`--detector hog` uses OpenCV's built-in pedestrian HOG (no model file).
ONNX Runtime is optional: without it (`-DS3M_WITH_ONNX=OFF`, or not installed)
the project still builds and `--detector onnx` falls back to a warning + no-op.
The `test_onnx_detector` suite runs against a 99 KB hand-built ONNX fixture
(`scripts/make_test_model.py`) — no download, deterministic.

## Example benchmark output

Measured on the built-in synthetic scene (640×480, default config: SGBM,
`num_disparities = 128`, `block_size = 5`), background plane at 12 m plus three
closer cards:

```
$ ./build/apps/benchmark_depth --source synthetic --frames 5

matcher = SGBM   num_disparities = 128   block_size = 5
frames  = 5      match: mean ~120 ms  (single-threaded SGBM, RelWithDebInfo)

disparity (px):  RMSE 1.03   bad-2.0 0.7%    density 99.2%
depth (m):       RMSE 0.66   abs-rel 1.2%    delta<1.25 99.3%
```

Dropping to `num_disparities = 64` roughly halves the matcher time; Phase 5
targets a real-time budget. `--out <dir>` also writes a Markdown results table.
Middlebury numbers will be lower — real photographs have textureless regions and
occlusions the synthetic scene doesn't.

Numbers depend on your machine; `--out <dir>` also writes a Markdown table for
pasting into a results section.

## Project layout

```
include/s3m/            public headers                 src/            implementation
  core/    types, config (YAML), timing/profiling
  camera/  CameraModel (pinhole), StereoRig (baseline, doffs, Q, triangulation)
  depth/   StereoMatcher (BM/SGBM wrapper), DepthMetrics (RMSE / bad-px / delta)
  geometry/ reprojection: disparity -> point cloud / depth map / 3D detections
  io/      FrameSource, SyntheticStereoSource, MiddleburySource, PFM, PLY export
  detection/ Detector interface, HOG, OnnxDetector (ORT), letterbox, NMS, COCO
  tracking/  KalmanFilter (generic linear), constant-velocity 3D, Track
  viz/     depth / disparity colourisation, image tiling, detection overlays

apps/    stereo_depth_demo, benchmark_depth, benchmark_detect
tests/   GoogleTest suites (12 files) + tests/data/ ONNX fixture
cmake/   warning flags, FindONNXRuntime        configs/ default.yaml
docs/    architecture.md, roadmap.md           scripts/ datasets, model export
```

## The geometry, briefly

Rectified horizontal stereo, left camera as the reference frame (OpenCV axes:
X right, Y down, Z forward). For a point at depth `Z`:

```
d = fx * B / Z - doffs                 doffs = cx_right - cx_left  (0 if perfectly rectified)
Z = fx * B / (d + doffs)
```

`StereoRig::reprojectionMatrix()` returns the 4×4 `Q` with
`[X Y Z W]ᵀ = Q · [u v d 1]ᵀ`, consistent with `cv::reprojectImageTo3D` and with
`StereoRig::triangulate()` — a property checked directly in
[`tests/test_stereo_rig.cpp`](tests/test_stereo_rig.cpp).

## Roadmap

1. **Stereo depth core — done.** geometry, BM/SGBM, depth, reprojection, PLY,
   synthetic + Middlebury sources, benchmark, tests, Docker, CI.
2. **Detection in the loop — done.** ONNX Runtime detector (YOLOv8/v5), letterbox
   + class-aware NMS, 2D → 3D promotion in the demo, `benchmark_detect`.
3. 3D multi-object tracking: data association + track lifecycle, KITTI tracking
   metrics (MOTA / IDF1), trajectory export.
4. Visual odometry: feature tracks, essential matrix, camera pose — touches the
   VI-SLAM side.
5. Optimisation: profiling, tiling/parallelism, optional CUDA/TensorRT path.

Details and rationale in [docs/roadmap.md](docs/roadmap.md).

## License

MIT — see [LICENSE](LICENSE).
