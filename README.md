# stereo-3d-mot

**Real-time stereo vision pipeline in modern C++: rectified stereo → disparity → metric depth → 3D reprojection → detection → 3D multi-object tracking.**

Built as a from-scratch C++17 systems project on top of OpenCV and Eigen, developed
Linux-first inside Docker, with unit tests, a quantitative benchmark, and CI.

[![ci](https://github.com/l4NGEL/stereo-3d-mot/actions/workflows/ci.yml/badge.svg)](https://github.com/l4NGEL/stereo-3d-mot/actions/workflows/ci.yml)

> Status: **Phases 1–3 complete.** Calibrated stereo geometry, BM/SGBM disparity,
> metric depth, dense 3D reprojection + PLY export, a synthetic scene with exact
> ground truth, a Middlebury 2014 loader, a depth-accuracy benchmark, an ONNX
> Runtime detector (YOLOv8/v5) wired through `promoteTo3D`, and a 3D multi-object
> tracker (Hungarian/greedy association, gated on either chi-square Mahalanobis
> distance in 3D or 2D IoU — pick per run and compare) with trajectory export.
> 70+ unit tests pass; all five apps run end to end. KITTI MOTA/IDF1 against a
> real detector is next. See [docs/roadmap.md](docs/roadmap.md).

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
 detector (2D)  ----> promote to 3D ---> Tracker  (predict -> gate -> Hungarian/
 HOG / ONNX           (median Z in box)   greedy associate -> update -> birth/death)
 (YOLOv8 / v5)                            gate: chi-sq Mahalanobis(3D) or IoU(2D)
                                                     v
                                          {track id, X,Y,Z, v, confirmed}
                                                     v
                                          CSV / PLY-polyline trajectory export
```

Every stage above is implemented and covered by tests.

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

## Tracking → 3D MOT

`Tracker` runs each frame's `Detection3D`s through predict → associate →
update/coast → birth/death. Two association methods read from the exact same
`Track` state, so they're a fair, swappable comparison — not two different
trackers:

| `tracking.association` | cost function | uses depth for the *decision*? |
|---|---|---|
| `mahalanobis3d` (default) | chi-square-gated squared Mahalanobis distance in 3D | yes |
| `iou2d` | 1 − IoU on the last matched 2D box | no (the classic baseline) |

Both respect `tracking.use_hungarian` (optimal Kuhn-Munkres, default) vs a
greedy nearest-first baseline (`solveAssignmentGreedy`), and a hard class-id
gate. Enable it in the demo and export the trajectories:

```bash
./build/apps/stereo_depth_demo --source synthetic --detector onnx \
    --model models/yolov8n.onnx --track \
    --trajectories out/tracks.csv --trajectories-ply out/tracks.ply --out out
```

**Why 3D association matters** — `benchmark_track` runs both methods on a
hermetic synthetic scene: a near object (2 m) and a far object (9 m) cross
paths on screen. The renderer already draws nearer cards over farther ones, so
the simulated detector realistically stops firing on the far object while it's
occluded — a real gap to re-acquire across, not just noisy boxes.

```bash
$ ./build/apps/benchmark_track --frames 120

method                        ID switch   ID consist.%   fragm. false trk%     tracks    update (us)
3D Mahalanobis + Hungarian            2          99.5%        1      25.0%          4          12.02
2D IoU        + Hungarian             2          58.3%        2      60.0%          5           5.47
```

(Full output has greedy rows too.) Same detections, same noise, same gap —
depth-aware association keeps its identity through the occlusion, image-overlap
association mostly doesn't. Robust across seeds: sweeping `--seed 1..5` keeps
Mahalanobis at 98–100% ID consistency and IoU2D at 58–61% (occasionally both
hit 100% when the run's noise never seriously tests either). `--seed` and the
noise flags change the exact counts.

`TrackerTest.DepthSeparatesOccludingBoxesIou2DGetsItWrong` pins the mechanism
down to a single frame with hand-verified numbers: one new detection sitting
exactly on a far track's last box (IoU = 1) but carrying the near object's true
depth — the IoU-only tracker takes the box overlap and assigns it to the far
track; the Mahalanobis tracker takes the depth and assigns it to the near one.
This is a hermetic proxy for identity-preservation quality, not KITTI
MOTA/IDF1 against real detector output — that's the next step (see
[docs/roadmap.md](docs/roadmap.md)).

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
  io/      FrameSource, SyntheticStereoSource, MiddleburySource, PFM, PLY,
           trajectory CSV/PLY export
  detection/ Detector interface, HOG, OnnxDetector (ORT), letterbox, NMS, COCO
  tracking/  KalmanFilter, Track, assignment (Hungarian/greedy), Tracker
  viz/     depth / disparity colourisation, image tiling, detection/track overlays

apps/    stereo_depth_demo, benchmark_depth, benchmark_detect, benchmark_track
tests/   GoogleTest suites (15 files) + tests/data/ ONNX fixture
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
3. **3D multi-object tracking — done.** `Tracker` (Hungarian/greedy, chi-square
   Mahalanobis-3D or IoU-2D association, class gate, birth/death), trajectory
   export, `benchmark_track` head-to-head comparison. KITTI MOTA/MOTP/IDF1
   against a real detector is next.
4. Visual odometry: feature tracks, essential matrix, camera pose — touches the
   VI-SLAM side.
5. Optimisation: profiling, tiling/parallelism, optional CUDA/TensorRT path.

Details and rationale in [docs/roadmap.md](docs/roadmap.md).

## License

MIT — see [LICENSE](LICENSE).
