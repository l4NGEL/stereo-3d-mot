# stereo-3d-mot

**Real-time stereo vision pipeline in modern C++: rectified stereo → disparity → metric depth → 3D reprojection → detection → 3D multi-object tracking.**

Built as a from-scratch C++17 systems project on top of OpenCV and Eigen, developed
Linux-first inside Docker, with unit tests, a quantitative benchmark, and CI.

[![ci](https://github.com/l4NGEL/stereo-3d-mot/actions/workflows/ci.yml/badge.svg)](https://github.com/l4NGEL/stereo-3d-mot/actions/workflows/ci.yml)

> Status: **Phases 1–5 built and run end to end, including on real KITTI
> data.** Calibrated stereo geometry, BM/SGBM disparity, metric depth, dense
> 3D reprojection + PLY export, a synthetic scene with exact ground truth, a
> Middlebury 2014 loader, a depth-accuracy benchmark, an ONNX Runtime detector
> (YOLOv8/v5) wired through `promoteTo3D`, a 3D multi-object tracker with
> three interchangeable association strategies — chi-square Mahalanobis
> distance in 3D, 2D IoU, and a Phase 5 fusion of both plus an appearance
> (color-histogram) cue — Hungarian or greedy, pick per run and compare, with
> trajectory export, and a CLEAR-MOT + IDF1 evaluator (`MotAccumulator`,
> validated against py-motmetrics) with a KITTI tracking-benchmark loader.
> 106 unit tests pass (1 skipped outside its expected working-directory
> context); all 5 apps run end to end, including `benchmark_kitti`
> against a real downloaded KITTI tracking sequence — see
> [KITTI evaluation](#kitti-evaluation) and
> [Phase 5](#phase-5--appearance-aware-reid-association) below for the real
> numbers and the honest read on what they mean (short version: absolute
> MOTA is negative because of a real, explained detector/GT domain mismatch;
> on real noisy stereo depth the 2D-vs-3D gap from the synthetic benchmark
> mostly closes; and fusing in an appearance cue recovers a real, if partial,
> identity-consistency win on that same real sequence — not a cherry-picked
> "ReID wins" claim).

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
update/coast → birth/death. Three association methods read from the exact
same `Track` state, so they're a fair, swappable comparison — not different
trackers:

| `tracking.association` | cost function | uses depth for the *decision*? |
|---|---|---|
| `mahalanobis3d` (default) | chi-square-gated squared Mahalanobis distance in 3D | yes |
| `iou2d` | 1 − IoU on the last matched 2D box | no (the classic baseline) |
| `fused` | weighted blend of both plus an appearance cue — [Phase 5](#phase-5--appearance-aware-reid-association) | yes, plus appearance |

All three respect `tracking.use_hungarian` (optimal Kuhn-Munkres, default) vs
a greedy nearest-first baseline (`solveAssignmentGreedy`), and a hard
class-id gate. Enable one in the demo and export the trajectories:

```bash
./build/apps/stereo_depth_demo --source synthetic --detector onnx \
    --model models/yolov8n.onnx --track \
    --trajectories out/tracks.csv --trajectories-ply out/tracks.ply --out out
```

**What `benchmark_track` measures, and what it doesn't.** It runs both methods
on a hermetic synthetic scene — a near object (2 m) and a far object (9 m)
cross paths on screen; the renderer already draws nearer cards over farther
ones, so the simulated detector realistically stops firing on the far object
while it's occluded, a real gap to re-acquire across, not just noisy boxes.
This is a controlled, reproducible way to exercise one specific claim (does
gating on depth reduce identity errors through an occlusion where 2D overlap
alone can't tell the objects apart), **not** a general "3D beats 2D" result —
it's one scene, two objects, synthetic noise. Raw ID-switch counts alone are
also a misleading headline number here: a handful of switches that each
resolve back to the right id within a couple of frames (typical of the
Mahalanobis runs) hurts identity quality far less than the same count from
switches that stick (typical of the IoU runs), which is why **ID consistency**
— the fraction of a track's lifetime spent under its dominant id — is the
number to read first:

```bash
$ ./build/apps/benchmark_track --frames 120

-- this project's identity-preservation metrics --
method                        ID switch   ID consist.%   fragm. false trk%     tracks    update (us)
3D Mahalanobis + Hungarian            2          99.5%        1      25.0%          4          14.83
2D IoU        + Hungarian             2          58.3%        2      60.0%          5           4.79

-- CLEAR-MOT + IDF1 (MotAccumulator, gate=1.5 m) --
method                           MOTA     MOTP     IDF1     IDSW     Frag
3D Mahalanobis + Hungarian      0.979    0.074    0.989        0        0
2D IoU        + Hungarian       0.929    0.204    0.543        2        2
```

Same switch count, very different consistency — exactly the "raw count lies,
consistency doesn't" case. The standard metrics agree, more starkly: MOTA
barely moves (0.979 vs 0.929 — both methods detect the objects fine, MOTA is
dominated by presence, not identity) but **IDF1**, the metric built
specifically to measure identity correctness, is 0.989 vs 0.543 — the IoU
tracker gets barely half credit for keeping the right label on the right
object. Swept across `--seed 1..5`: Mahalanobis holds 98–100% ID consistency
every time, IoU2D lands at 58–61% in 4 of 5 seeds and ties Mahalanobis at
100% in the fifth, when that particular run's noise never seriously tests
either method. The honest claim from this benchmark is scoped: *in this
simulated occlusion scenario, depth-aware association substantially improved
identity consistency over the 2D IoU baseline.* Whether that holds on real
detections and real occlusions is exactly what the KITTI evaluation below is
for.

`TrackerTest.DepthSeparatesOccludingBoxesIou2DGetsItWrong` pins the mechanism
down to a single frame with hand-verified numbers: one new detection sitting
exactly on a far track's last box (IoU = 1) but carrying the near object's true
depth — the IoU-only tracker takes the box overlap and assigns it to the far
track; the Mahalanobis tracker takes the depth and assigns it to the near one.

## KITTI evaluation

`MotAccumulator` (`mot_metrics.hpp`) is a from-scratch CLEAR-MOT
(Bernardin & Stiefelhagen, 2008) + IDF1 (Ristani et al., 2016) implementation
— MOTA, MOTP, IDF1, ID switches, fragmentation, precision/recall — validated
against **py-motmetrics**, the reference library, on 400+ randomised trials
plus hand-built edge cases (gaps, switches, crossings, gating) before being
ported to C++. `benchmark_track`'s table above now reports it too, scored
against the synthetic scene's exact ground truth, alongside its own
identity-preservation metrics — one more full exercise of the accumulator
before trusting it on real data.

`kitti_loader.hpp` parses the KITTI tracking-benchmark's label and calibration
formats (`readKittiLabels`, `readKittiCalib`) and reads `image_02`/`image_03`
pairs as a `FrameSource` (`KittiTrackingSource`), tested against a hermetic
hand-built fixture. `benchmark_kitti` runs the *actual* pipeline — stereo
depth, the ONNX detector, `promoteTo3D`, `Tracker` — once per association
method on a real sequence, and scores each against its ground truth:

```bash
scripts/download_kitti.sh 0000 --with-images   # calib+labels always (~10 MB); one sequence's
                                                # images via HTTP range requests (~265 MB, not
                                                # the ~30 GB full archives -- see the script)
./build/apps/benchmark_kitti --kitti-root data/kitti --sequence 0000 \
    --detector onnx --model models/yolov8n.onnx --config configs/kitti.yaml
```

Run on sequence 0000 (154 frames, the standard first KITTI tracking sequence —
a short residential drive), 711 ground-truth object-frames after dropping
`DontCare` regions (292 Van, 243 Car, 154 Cyclist, 22 Pedestrian):

```
sequence 0000   detector=onnx (yolov8n, COCO)   frames=all   gate=2 m

3D Mahalanobis: MOTA=-2.304  MOTP=0.995 m  IDF1=0.182  IDSW=33  Frag=11  FP=2195  FN=121  Prec=0.212  Rec=0.830
2D IoU        : MOTA=-2.135  MOTP=1.020 m  IDF1=0.187  IDSW=35  Frag=23  FP=1980  FN=214  Prec=0.201  Rec=0.699
```

`configs/kitti.yaml` restricts the detector to COCO's road-relevant classes
(person, bicycle, car, motorcycle, bus, truck) and raises `measurement_noise`
from the synthetic scene's 0.05 m to 1.0 m — real SGBM stereo error on KITTI
ranges is far larger than the synthetic scene's simulated jitter (confirmed
by this very run: MOTP, the average position error over *matched* pairs
only, comes out to ~1 m), and the tighter synthetic-tuned default
over-rejected genuinely correct matches in the Mahalanobis gate. Same class
of bug as the Phase 3 measurement-noise mistuning, re-surfacing at a
different noise scale on real sensor data instead of simulated data.

**Two honest findings here, not one clean headline:**

- **Absolute MOTA is negative, and that's a detector/GT domain-mismatch
  artifact, not a pipeline bug.** Spot-checking frame 0: YOLOv8n correctly
  finds 4 cars + 3 bicycles + 2 people in the image; KITTI's tracking GT for
  that same frame labels exactly 3 objects (1 Van, 1 Cyclist, 1 Pedestrian).
  Averaged over the whole run, confirmed tracks outnumber GT objects roughly
  4:1. KITTI's *tracking* ground truth is not an exhaustive census of every
  visible instance of its classes the way the detection benchmark is — a
  general-purpose COCO detector, with no KITTI-specific fine-tuning, finds
  plenty of real cars, bicycles and people the tracking labels simply don't
  include, and every one of those scores as a false positive under CLEAR-MOT.
  That's a known characteristic of scoring an off-the-shelf detector against
  KITTI tracking GT directly, not something this project's tracker or metrics
  implementation got wrong — confirmed by re-deriving MOTA/MOTP/IDF1 against
  py-motmetrics earlier in this same phase.
- **On real data, 2D IoU and 3D Mahalanobis land close together — this is not
  a rerun of the synthetic-scene result.** The synthetic benchmark above
  shows Mahalanobis clearly ahead (IDF1 0.989 vs 0.543) because its simulated
  depth was clean by construction. Real stereo depth is noisier — MOTP ~1 m
  here, and growing with range — so on this sequence the two methods come out
  close (IDF1 0.182 vs 0.187; Mahalanobis fragments less, 11 vs 23; IoU edges
  it narrowly on IDF1). The honest reading: depth-aware association's
  advantage is real but conditional on depth quality — decisive when depth is
  accurate, and eroded once stereo noise grows large enough to rival the
  gate size. That's a more useful and more defensible result than "3D always
  wins," and it's exactly the kind of gap the ReID/appearance-fusion work in
  Phase 5 is meant to help close, by giving the associator a cue that doesn't
  degrade with depth noise.

## Phase 5 — Appearance-aware (ReID) association

A third association cue, fused with the existing two as
`C = α·C_3D + β·C_IoU + γ·C_ReID` (`AssociationMethod::kFusedAppearance`,
`tracker.hpp`), α/β/γ = 0.5/0.2/0.3 by default
(`TrackerParams::fused_weight_*`, `configs/*.yaml`).

**Scope, stated up front**: the "ReID" cue here is a classical HSV
color histogram (`tracking/appearance.hpp`, Bhattacharyya distance via
`cv::compareHist`), not a learned ReID embedding. That's deliberate, not a
shortcut taken by accident — it needs no model file, export step, or new
dependency, and histogram-based appearance matching is a real,
long-established MOT technique for exactly the role it plays here: one
fusable cue among several, not the sole association signal a real ReID
network is designed to be. A learned embedding would likely separate
appearance better, especially between similarly-colored objects — that's
future work this project doesn't claim to have done.

Gating uses the *union* of the Mahalanobis and IoU gates, not their
intersection: deliberately looser than either alone, because the whole point
of fusing in a second geometric cue plus appearance is staying robust when
one cue is unreliable — and Phase 4 just showed real stereo depth is exactly
that unreliable cue on noisy real data. Gating on the intersection would
just inherit whichever cue is currently worse. A detection or track with no
appearance descriptor yet falls back to a neutral 0.5 appearance cost rather
than being excluded.

`TrackerTest.FusedAppearanceRecoversIdentityMahalanobisAloneGetsWrong` pins
the mechanism down with hand-verified numbers, mirroring Phase 3's
`DepthSeparatesOccludingBoxesIou2DGetsItWrong`: two tracks settle at close
depths (2.00 m red, 2.10 m blue); a red disambiguating detection sits
geometrically *closer* to the wrong (blue) track (0.07 m from red vs 0.03 m
from blue — plain Mahalanobis prefers blue by a clear margin). Fusing in
appearance correctly recovers red. A paired test confirms the neutral 0.5
fallback doesn't change anything when neither side has a descriptor yet —
fused degrades gracefully to plain geometry, it doesn't add noise.

The fusion weights were set by reasoning plus those unit tests, **not**
tuned against either benchmark's numbers below — the same discipline this
project has used since Phase 3 (validate against a trusted reference before
trusting a result), applied here as validate-before-you-look-at-the-target
instead of validate-against-a-reference-library.

**Synthetic scene** (`benchmark_track` — cards have distinct colors by
construction, tints already baked into `SyntheticStereoSource`):

```
method                        ID switch   ID consist.%   fragm. false trk%
3D Mahalanobis + Hungarian            2          99.5%        1      25.0%
2D IoU        + Hungarian             2          58.3%        2      60.0%
3D+IoU+ReID    + Hungarian            0         100.0%        0      50.0%
```

Fused ties or beats plain 3D Mahalanobis on every identity metric (IDF1 tied
at 0.989) — but at a higher false-track rate (50% vs 25%). That's the union
gate's honest cost: it lets through a false-positive detection the tighter
single gate would have rejected, in exchange for never losing a genuinely
correct match to an unreliable single cue. A real, explainable trade, not
hidden in the writeup.

**Real KITTI sequence 0000** (`benchmark_kitti`, same run as the [KITTI
evaluation](#kitti-evaluation) section above, third method added):

```
3D Mahalanobis: IDF1=0.182  IDSW=33  Frag=11  FP=2195  MOTA=-2.304
2D IoU        : IDF1=0.187  IDSW=35  Frag=23  FP=1980  MOTA=-2.135
3D+IoU+ReID   : IDF1=0.203  IDSW=27  Frag=13  FP=2145  MOTA=-2.270
```

On real, noisy data, fusion **wins outright on IDF1** (0.203 vs 0.182 vs
0.187 — the identity-focused metric this whole project has centered on) and
**wins outright on ID switches** (27 vs 33 vs 35, the fewest of all three).
Not a clean sweep, and reported as such: plain 3D still fragments least (11
vs fused's 13), and plain 2D still has the fewest raw false positives (1980
vs fused's 2145) — both the same union-gate trade-off seen on the synthetic
scene, just smaller in absolute effect here.

This directly validates the motivation written into the Phase 4 section
above *before* any of Phase 5 was built: real stereo depth noise erodes 3D
Mahalanobis's clean synthetic-scene win, and appearance is a cue that
doesn't degrade with depth noise the way Mahalanobis gating does — fusing it
back in recovers real ground on the exact same real sequence, without
tuning a single weight against this number.

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
  io/      FrameSource, SyntheticStereoSource, MiddleburySource, KITTI loader,
           PFM, PLY, trajectory CSV/PLY export
  detection/ Detector interface, HOG, OnnxDetector (ORT), letterbox, NMS, COCO
  tracking/  KalmanFilter, Track, assignment (Hungarian/greedy), Tracker,
             MotAccumulator (CLEAR-MOT + IDF1)
  viz/     depth / disparity colourisation, image tiling, detection/track overlays

apps/    stereo_depth_demo, benchmark_{depth,detect,track,kitti}
tests/   GoogleTest suites (18 files) + tests/data/ ONNX fixture
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
   export, `benchmark_track` head-to-head comparison.
4. **KITTI + MOTA/MOTP/IDF1 — done.** `MotAccumulator` (validated against
   py-motmetrics), the KITTI tracking loader, and `benchmark_kitti` all run
   against a real downloaded sequence; see [KITTI evaluation](#kitti-evaluation).
5. **Appearance-aware association (ReID) — done.** A third cost term
   (classical color-histogram appearance, not a learned embedding — scoped
   deliberately) fused with the existing geometric cues; real win on IDF1 and
   ID switches on the same KITTI sequence. See
   [Phase 5](#phase-5--appearance-aware-reid-association).
6. Real-time optimisation: profiling, tiling/parallelism, single→multi-thread,
   optional CUDA/TensorRT path.
7. Visual odometry: feature tracks, essential matrix, camera pose — touches the
   VI-SLAM side.
8. ROS2 integration (optional / bonus), once Phases 6–7 give the stack
   something worth wrapping in a node graph.

Details and rationale in [docs/roadmap.md](docs/roadmap.md).

## License

MIT — see [LICENSE](LICENSE).
