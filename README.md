# stereo-3d-mot

**Real-time stereo vision pipeline in modern C++: rectified stereo → disparity → metric depth → 3D reprojection → detection → 3D multi-object tracking.**

Built as a from-scratch C++17 systems project on top of OpenCV and Eigen, developed
Linux-first inside Docker, with unit tests, a quantitative benchmark, and CI.

[![ci](https://github.com/l4NGEL/stereo-3d-mot/actions/workflows/ci.yml/badge.svg)](https://github.com/l4NGEL/stereo-3d-mot/actions/workflows/ci.yml)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## What this project actually does

Feed it a pair of stereo camera images (or a recorded stereo sequence) and it will:

1. Turn the left/right pair into a metric **depth map** — how far away every pixel is, in meters, not just a disparity image.
2. **Detect objects** in the left image (cars, pedestrians, cyclists, …) with a YOLO model, running in C++ through ONNX Runtime.
3. Lift every 2D detection into **3D** using the depth at its location.
4. **Track every object through time in 3D**, giving each one a stable ID that survives brief occlusion, using a Kalman filter and a choice of three interchangeable matching strategies (3D distance, 2D box overlap, or a fusion of both plus appearance).
5. Export the tracked trajectories (CSV / PLY) and score them against ground truth with the same metrics (MOTA / MOTP / IDF1) the multi-object-tracking research literature uses.

This is the same class of problem an autonomous vehicle or ground robot's perception stack has to solve: turn two camera feeds into *"here is everything around me, in 3D, and here is where it's going."* Everything is written from scratch in C++17 — no ROS, no off-the-shelf SLAM/tracking framework — on OpenCV, Eigen, and ONNX Runtime, and it's validated on the real KITTI self-driving dataset, not just synthetic data.

## Results at a glance

| | |
| --- | --- |
| **Unit tests** | 109/109 passing (GoogleTest, enforced in CI) |
| **Real-time throughput** (Phase 6, tiled stereo + CPU detection) | 4.9 → **7.8 FPS**, a real 1.6x over the pre-optimisation baseline |
| **Stereo matching speedup** from tiling | **1.95x** (152.8 → 78.5 ms/frame), tracking output bit-identical before/after |
| **Identity tracking quality**, synthetic occlusion test (IDF1) | depth-aware 3D association **0.989** vs 2D-only baseline **0.543** |
| **Identity tracking quality**, real KITTI driving data, 3 sequences | appearance-fused association is never the worst of 3 methods; wins outright on 1 of 3 |
| **MOT metrics implementation** (CLEAR-MOT + IDF1, from scratch) | cross-validated against `py-motmetrics` (the reference library) on 400+ trials |
| **GPU (RTX 2080) vs. 16-thread CPU** for detection | measured, and honestly reported: **no speedup** for this model at batch size 1 |

Full numbers, methodology, and the honest caveats behind every one of these are in [KITTI evaluation](#kitti-evaluation), [Phase 5](#phase-5--appearance-aware-reid-association), and [Phase 6](#phase-6--real-time-optimisation) below.

---

## Pipeline

```mermaid
flowchart LR
    LCam[Left camera] --> Rect[Rectified stereo pair]
    RCam[Right camera] --> Rect

    Rect --> SM["StereoMatcher (BM / SGBM)"]
    SM --> Disp["Disparity map (px)"]
    Disp --> Depth["Depth Z = f·B / (d + doffs)"]
    Depth --> Reproj["Reprojection (Q matrix)"]
    Reproj --> Cloud["3D point cloud (X,Y,Z), meters"]

    LCam --> Det["Detector: HOG or ONNX (YOLOv8 / v5)"]
    Det --> Promote["promoteTo3D (median depth in box)"]
    Depth --> Promote
    Promote --> Track["Tracker: predict -> gate -> associate -> update -> birth/death"]
    Track --> State["track id, X, Y, Z, velocity, confirmed"]
    State --> Export["CSV / PLY trajectory export"]

    classDef stage fill:#1f6feb,color:#fff,stroke:#1f6feb;
    class SM,Det,Promote,Track stage;
```

Association inside the tracker is gated by chi-square Mahalanobis distance in 3D, 2D IoU, or a fusion of both plus an appearance cue — pick per run, see [Tracking → 3D MOT](#tracking--3d-mot). Every stage above is implemented and covered by tests.

## Example output

![Demo output: left camera frame, colorized estimated disparity, colorized depth, and ground-truth disparity](docs/images/demo_board.png)

Left → right: the left camera frame (three cards at known depths in front of a textured background; overlaid with detections/tracks when `--track` is on), colorized **estimated** disparity, colorized **depth** (metres), and **ground-truth** disparity for comparison. Generated with `make demo` on the built-in synthetic scene, which is used throughout this README as a controlled, ground-truth-exact scenario before trusting results on real KITTI data.

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
```

**Identity-preservation metrics** (this project's own):

| method | ID switch | ID consist.% | fragm. | false trk% | tracks | update (µs) |
| --- | --- | --- | --- | --- | --- | --- |
| 3D Mahalanobis + Hungarian | 2 | **99.5%** | 1 | 25.0% | 4 | 14.83 |
| 2D IoU + Hungarian | 2 | 58.3% | 2 | 60.0% | 5 | 4.79 |

**CLEAR-MOT + IDF1** (`MotAccumulator`, gate = 1.5 m):

| method | MOTA | MOTP | IDF1 | IDSW | Frag |
| --- | --- | --- | --- | --- | --- |
| 3D Mahalanobis + Hungarian | 0.979 | 0.074 | **0.989** | 0 | 0 |
| 2D IoU + Hungarian | 0.929 | 0.204 | 0.543 | 2 | 2 |

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

`sequence 0000` — detector = ONNX (yolov8n, COCO), frames = all, gate = 2 m:

| method | MOTA | MOTP | IDF1 | IDSW | Frag | FP | FN | Prec | Rec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D Mahalanobis | -2.304 | 0.995 m | 0.182 | 33 | 11 | 2195 | 121 | 0.212 | 0.830 |
| 2D IoU | -2.135 | 1.020 m | **0.187** | 35 | 23 | 1980 | 214 | 0.201 | 0.699 |

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

| method | ID switch | ID consist.% | fragm. | false trk% |
| --- | --- | --- | --- | --- |
| 3D Mahalanobis + Hungarian | 2 | 99.5% | 1 | 25.0% |
| 2D IoU + Hungarian | 2 | 58.3% | 2 | 60.0% |
| 3D+IoU+ReID + Hungarian | 0 | **100.0%** | 0 | 50.0% |

Fused ties or beats plain 3D Mahalanobis on every identity metric (IDF1 tied
at 0.989) — but at a higher false-track rate (50% vs 25%). That's the union
gate's honest cost: it lets through a false-positive detection the tighter
single gate would have rejected, in exchange for never losing a genuinely
correct match to an unreliable single cue. A real, explainable trade, not
hidden in the writeup.

**Real KITTI, three sequences, not one.** A single-sequence result invites
exactly the kind of cherry-picking this project has tried to avoid
throughout, so `benchmark_kitti --sweep` (see below) was re-run on two more
sequences chosen for a genuinely different occlusion profile — 0003 (49% of
object-frames have some occlusion, sparse: 2.7 objects/frame) and 0017 (62%
occluded, dense: 6.1 objects/frame) — against 0000's own 30%/4.6:

| sequence | occlusion | IDF1 (3D / 2D / Fused) | IDSW (3D / 2D / Fused) | Frag (3D / 2D / Fused) |
| --- | --- | --- | --- | --- |
| 0000 | 30% | 0.182 / 0.187 / **0.203** | 33 / 35 / 27 | 11 / 23 / 13 |
| 0003 | 49% | 0.223 / **0.301** / 0.283 | 0 / 2 / 0 | 2 / 3 / 2 |
| 0017 | 62% | **0.486** / 0.386 / 0.462 | 5 / 13 / 8 | 1 / 6 / 2 |

**This is not "fusion wins."** It wins outright on 0000, but on 0003 plain
2D IoU has the best IDF1 (0.301, fusion 0.283 second), and on 0017 plain 3D
Mahalanobis has the best IDF1 (0.486, fusion 0.462 second) — sparser or
more-occluded scenes apparently suit one pure cue better than the blend.
The honest, three-sequence-wide pattern is narrower than "always best" but
still real: **across all three sequences, fused is never the worst of the
three on IDF1, IDSW, or fragmentation.** It's a hedge, not a universal
upgrade — it never loses outright to both alternatives, and on the one
sequence where neither pure cue dominated (0000), fusing them won. That is
a materially more defensible claim than the single-sequence version of this
section used to make, and a more useful one: it tells you fusion is a safe
default when you don't know in advance which failure mode (bad depth vs.
box-overlap confusion) a given scene will hit, not that it beats a
well-matched single cue on its own turf.

**Weight sensitivity** (`--sweep`, five (α,β,γ) settings from 0.4/0.3/0.3 to
0.6/0.1/0.3, plus two appearance-heavier settings): IDF1 moves by at most
~0.04 across the whole grid on any one sequence, and the three-sequence
ranking above (fused beats one pure cue, loses to the other, never worst)
holds at every tested setting — the qualitative finding isn't an artifact of
picking exactly 0.5/0.2/0.3.

**Cue ablation** (3D-only / 3D+IoU / 3D+appearance / full fused, same
`--sweep`): confirms appearance and IoU are each pulling real weight rather
than one silently doing nothing — 3D+IoU and 3D+appearance land at
different points on all three sequences, and neither alone matches the full
fused row. Caveat stated in the code and repeated here: because
`kFusedAppearance`'s *gate* is always the union of the Mahalanobis and IoU
gates regardless of weight, "no IoU" still benefits from IoU admitting
candidates Mahalanobis alone wouldn't — a fully independent ablation would
need separate gating per combination, which this doesn't do.

**A real bug, caught by a determinism check.** Building `--sweep` (many
association variants run back-to-back over one cached frame set, instead of
each method recomputing detections independently) surfaced a genuine bug:
`Track`'s constructor stored its initial appearance descriptor via
`cv::Mat`'s shallow, reference-counted copy, so a newly-born track's
descriptor *aliased* the same buffer as the cached detection it was born
from. `Track::correct()`'s EMA blend then reassigned that descriptor in
place — silently mutating the cached, supposedly-read-only frame data a
*later*, supposedly-independent `run()` call would read. Symptom: the exact
same (association, weights) setting gave different IDF1/IDSW depending on
how many other variants had already run first in the same process — caught
by literally running the identical command three times and noticing the
*single-method* runs agreed with each other but not with the *same* setting
embedded inside `--sweep`. Fixed by `.clone()`-ing on ingestion
(`track.cpp`), re-verified full test suite (106/106 non-skipped) plus a
three-way cross-check that the same weight setting now gives byte-identical
output whether read from the plain row, the weight-sweep, or the ablation
table. This did **not** affect any previously-reported number — the code
path used for the original single-sequence Phase 5 result (and everything
in `benchmark_track`) recomputes detections fresh per run with no data
shared across calls, so there was nothing to alias — but it would have
silently corrupted every number in this section had `--sweep` shipped
without the determinism check.

This still connects back to the motivation written into the Phase 4 section
*before* any of Phase 5 was built: real stereo depth noise erodes 3D
Mahalanobis's clean synthetic-scene win, and appearance is a cue that
doesn't degrade with depth noise the way Mahalanobis gating does. The
three-sequence result narrows that to its defensible form: fusing it in
doesn't always beat a single well-matched cue, but it reliably avoids the
worst case, which is the property that actually matters when you can't
choose your scene in advance.

## Phase 6 — Real-time optimisation

Profiled the real pipeline on real KITTI data before optimising anything —
`benchmark_kitti --profile` wires `ProfileRegistry` through every stage:

`sequence 0000`, 154 frames, `cv::getNumThreads() = 16`:

| stage | cost |
| --- | --- |
| stereo | 152.8 ms/frame |
| capture (imread) | 137.9 ms/frame |
| detect | 49.5 ms/frame |
| depth + promote3d + appearance | <2 ms/frame combined |
| track | 0.19 ms/frame |

Stereo matching dominates — roughly 3x detection's cost, and everything
else is noise by comparison. `capture (imread)` is almost as expensive as
stereo matching itself; that's very likely a Docker-Desktop-on-Windows
bind-mount filesystem artifact (WSL2 cross-boundary I/O for many small PNG
reads), not real algorithmic cost or something a native-Linux deployment
would pay — reported honestly rather than folded into or dropped from the
headline number.

**Threading isn't uniform, and that's the whole reason tiling needed new
code.** `--threads 1` vs the default (16 cores): `detect` goes 121.9 → 47.0 ms
(2.6x — ONNX Runtime's own intra-op parallelism already uses every core, no
code required), but `stereo` barely moves (145.9 → 143.0 ms — OpenCV's
StereoSGBM in this build doesn't meaningfully parallelise internally). So
detection's "multi-thread" roadmap item was already done by the library;
stereo's wasn't, and needed real tiling work.

**Tiled the matcher** (`StereoMatcherParams::num_tiles`,
`StereoMatcher::computeTiledRaw`): horizontal strips, one independent
matcher per strip (OpenCV's BM/SGBM keep internal scratch state unsafe to
share across concurrent `compute()` calls), `cv::parallel_for_` across
strips, a small overlap margin cropped off before stitching so seams don't
show. `num_tiles <= 1` — the default everywhere except the new
`configs/kitti_tiled.yaml` — is byte-identical to the pre-Phase-6 code path;
nothing published before this phase changed.

The first overlap margin (a conservative `max(32, 8·block)` = 40 rows) was
too generous for KITTI's short (~375-row) images: past `num_tiles=8` the
margin competes directly with tile height, and at 16 tiles the *result was
net slower than not tiling at all* (468 ms vs 163 ms) — more tiles just
means redoing more of your neighbours' work. Tightened to `max(16, 3·block)`
= 16 rows after confirming accuracy held
(`StereoMatcherTest.TiledMatchesUntiledClosely`; `benchmark_depth` on the
synthetic scene's exact ground truth: RMSE 1.192 vs 1.191, bad-2.0% 0.854%
vs 0.853%). Re-measured the tile-count sweep with the tighter margin and the
sweet spot moved to **`num_tiles=4`, a real 1.95x stereo speedup** (152.8 →
78.5 ms/frame on the full 154-frame sequence); `num_tiles=8` is barely ahead
of untiled and 16 is still a net loss. The finding that matters isn't the
number 4, it's that **overlap-to-tile-height ratio, not core count, sets the
practical tile-count ceiling** on a given image size — a wider image, or a
tighter margin, would push the sweet spot higher.

**MOT-metric impact: unmeasurable.** Re-ran the full Phase 4/5 KITTI table
with tiling on (`configs/kitti_tiled.yaml`) against the published
(`configs/kitti.yaml`) numbers — IDF1, IDSW, fragmentation, false
positives/negatives, MOTA, precision and recall are **bit-identical** for
all three association methods; only MOTP (mean position error over matched
pairs) shifts in the 5th decimal place. Not one track identity or
match/mismatch decision changed on this sequence. The Phase 4/5 numbers
elsewhere in this README stay measured with `kitti.yaml`, untiled — this
result confirms they'd have looked the same either way, it doesn't quietly
change them.

**GPU-accelerated detection: built for real, and it doesn't help — reported
as such.** A real NVIDIA GPU (RTX 2080, 8 GB) is available on this machine
and confirmed reachable from Docker (`--gpus all`), so this wasn't left as
an "optional, untested" line. `Dockerfile.gpu` builds a separate image on a
CUDA 12.4 + cuDNN 9 base (matching ONNX Runtime 1.19.x's own documented
requirement, not guessed) with ORT's GPU tarball instead of the CPU one;
`OnnxDetector::Options::use_cuda` (`--cuda` / `configs/*.yaml`
`detector.use_cuda`) registers the CUDA execution provider on the ORT
session — no silent CPU fallback, it throws if the EP genuinely isn't
available, so a timing comparison can't accidentally compare CPU against
CPU. Detection results are **bit-identical** to CPU (same MOTA/IDF1/IDSW/
FP/FN for all three association methods on the full sequence), so this is
a fair, correctness-verified comparison, not just "it ran without crashing."

Measured on the real sequence: **`detect` on CUDA EP is 48.4 ms/frame vs
CPU's ~49.5–54.3 ms/frame (16 threads) — essentially no speedup.** Not a
bug, not a measurement error (re-confirmed after ruling out a system-load
confound — the first measurement attempt coincided with unrelated heavy
background CPU/GPU use from other work on this machine, which slowed
*every* stage including the CPU-only ones, and was re-measured once that
cleared). YOLOv8n is a genuinely tiny model (~3.2 M parameters) run one
image at a time (batch size 1): at that scale, fixed per-call overhead
(kernel launches, host↔device transfer, no batching to amortize it across)
competes directly with the actual compute, and a 16-thread CPU already
running well-optimized kernels (ORT's own MLAS backend) turns out to be
genuinely competitive rather than an easy target. Concrete, named
next steps that could still move this (not attempted here, correctly out of
scope for "does naive CUDA EP help" rather than silently left vague):
FP16 inference, `IOBinding` to cut host↔device copies, CUDA graph capture
to amortize launch overhead, or batching multiple frames — any of which
could change this result and are worth trying if GPU throughput becomes
the actual bottleneck later (right now stereo isn't on the GPU at all, so
it wouldn't be, yet).

**End to end**: 342.2 → 238.4 ms/frame (2.92 → 4.19 FPS) including the
imread artifact; 204.2 → 128.2 ms/frame (4.90 → **7.80 FPS**) excluding it
(stereo + detect + depth + promote, tiled stereo, CPU detect) — GPU
detection doesn't move this further, per the finding above. A real, measured
~1.6x over the Phase-6-start baseline, short of the 20 FPS target, reported
as such rather than rounded up. The honest ceiling on this pipeline, on this
hardware, without deeper GPU-specific work (FP16/IOBinding/graphs, or
porting stereo matching itself to `cv::cuda::StereoSGBM`, which would need
an OpenCV built with CUDA support — this project's apt-installed OpenCV
isn't) is CPU-bound tiled-SGBM plus CPU-or-GPU-equivalent detection, not a
GPU win waiting to be unlocked by more casual effort.

**Memory/allocation audit — answered by the profile above, not a separate
pass.** `depth`, `promote3d+appearance` and `track` combined cost under 2 ms
per frame, so there's no meaningful copy/allocation overhead hiding in this
project's own glue code — the two real costs (`stereo`, `detect`) are
library-internal compute (SGBM, ONNX Runtime), not inefficiencies in code
written here. Chasing allocations in the 2 ms that's left wouldn't move the
frame-time budget.

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
   deliberately) fused with the existing geometric cues; validated on three
   real KITTI sequences at different occlusion levels — never the worst of
   the three methods, wins outright on one. See
   [Phase 5](#phase-5--appearance-aware-reid-association).
6. **Real-time optimisation — done.** Profiled the real pipeline on real
   KITTI data first: stereo matching dominates (~3x detection's cost),
   detection already scales across cores for free (ONNX Runtime's own
   threading), stereo didn't and needed real tiling work — a measured 1.95x
   stereo speedup with a bit-identical tracking result. GPU detection built
   and measured for real on an actual RTX 2080 — and turned out not to help
   at batch size 1 for a model this small, a genuine, counter-intuitive
   finding reported honestly rather than assumed away. ~1.6x end to end,
   short of the 20 FPS target. See
   [Phase 6](#phase-6--real-time-optimisation).
7. Visual odometry: feature tracks, essential matrix, camera pose — touches the
   VI-SLAM side.
8. ROS2 integration (optional / bonus), once Phases 6–7 give the stack
   something worth wrapping in a node graph.

Details and rationale in [docs/roadmap.md](docs/roadmap.md).

## License

MIT — see [LICENSE](LICENSE).
