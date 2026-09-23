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

## Phase 6 — Real-time optimisation  ✅ done

- [x] **Profiled hotspots for real, on real KITTI data**
      (`benchmark_kitti --profile`, `ProfileRegistry` wired through
      `collectFrames`/`run`). Sequence 0000, 154 frames, baseline
      (`configs/kitti.yaml`, `num_tiles` unset -> 1): `stereo` 152.8 ms/frame,
      `capture (imread)` 137.9 ms/frame, `detect` 49.5 ms/frame, `depth` +
      `promote3d+appearance` under 2 ms/frame combined, `track` 0.19 ms/frame
      -- stereo matching dominates, roughly 3x the cost of detection, and the
      `depth`/`promote3d`/`track` stages are negligible next to either.
- [x] **`capture (imread)` measured, not guessed, and flagged honestly.**
      137.9 ms/frame to read one stereo PNG pair is comparable to the stereo
      match itself -- almost certainly a Docker-Desktop-on-Windows bind-mount
      filesystem artifact (WSL2 cross-boundary I/O), not real algorithmic
      cost or representative of a native-Linux/embedded deployment. Reported
      both ways (with and without it) rather than silently included in or
      excluded from the headline number.
- [x] **Single-thread -> multi-thread: measured, and it's not uniform.**
      `cv::setNumThreads(1)` vs the default (16 cores, `--threads`, new CLI
      flag): `detect` 121.9 -> 47.0 ms (2.6x -- ONNX Runtime's own intra-op
      parallelism already exploits every core, no code change needed);
      `stereo` 145.9 -> 143.0 ms (no change at all -- OpenCV's StereoSGBM in
      this build does not meaningfully parallelise internally, confirming
      the roadmap's original suspicion instead of assuming it). This is *why*
      tiling (next item) was the one piece that needed new code -- threading
      alone was already fully exploited for detection and did nothing for
      stereo.
- [x] **Tiled + parallelised the matcher** (`StereoMatcherParams::num_tiles`,
      `StereoMatcher::computeTiledRaw`, `stereo_matcher.cpp`): splits the
      image into `num_tiles` horizontal strips with a small overlap margin
      (absorbs SGBM's block/aggregation window at the seam), each strip
      computed by its own matcher instance via `cv::parallel_for_` (separate
      instances, not a shared one -- OpenCV's BM/SGBM keep internal scratch
      state that isn't safe to share across concurrent `compute()` calls),
      overlap cropped off before stitching. `num_tiles <= 1` (the default
      everywhere except the new `configs/kitti_tiled.yaml`) is byte-identical
      to the pre-Phase-6 code path -- nothing published before this phase
      changed.
    - First overlap heuristic (`max(32, 8*block)` = 40 rows) was too
      conservative for KITTI's short (~375-row) images: at `num_tiles=8` the
      overlap starts competing with tile height, and at 16 tiling was net
      *slower* than untiled (468 ms vs 163 ms) -- more tiles redoing a larger
      fraction of their neighbours' work, not free parallelism. Tightened to
      `max(16, 3*block)` (16 rows) after confirming accuracy held
      (`StereoMatcherTest.TiledMatchesUntiledClosely`, `benchmark_depth` on
      the exact-ground-truth synthetic scene: RMSE 1.192 vs 1.191, bad-2.0%
      0.854% vs 0.853%); re-measured the same tile-count sweep and the sweet
      spot moved to `num_tiles=4` at a real 1.95x stereo speedup (152.8 ->
      78.5 ms/frame, full 154-frame sequence), with `num_tiles=8` only barely
      ahead of untiled and 16 still a net loss. The lesson kept, not just the
      number: overlap-to-tile-height ratio, not core count, sets the
      practical tile-count ceiling on a given image size.
    - **MOT-metric impact: unmeasurable.** Re-ran the full Phase 4/5 KITTI
      table (`configs/kitti_tiled.yaml`, `num_tiles: 4`) against
      `configs/kitti.yaml`'s published numbers: IDF1, IDSW, Frag, FP, FN,
      MOTA, precision and recall are **bit-identical** for all three
      association methods; only MOTP (mean position error over matched
      pairs) moves in the 5th decimal place. No track identity or
      match/mismatch decision changed on this sequence. `kitti.yaml` (not
      `kitti_tiled.yaml`) stays the config the Phase 4/5 numbers above were
      measured with -- this doesn't retroactively change them, it confirms
      they'd have looked the same either way.
    - End-to-end: 342.2 -> 238.4 ms/frame (2.92 -> 4.19 FPS) including the
      imread artifact; 204.2 -> 128.2 ms/frame (4.90 -> 7.80 FPS) excluding
      it (stereo + detect + depth + promote only). Real, measured, roughly
      1.6x end-to-end -- well short of the 20 FPS target below, honestly
      reported as such, not rounded up.
- [x] **CUDA behind the same `Detector` interface -- built for real, and it
      doesn't help.** A real NVIDIA GPU (RTX 2080, 8 GB) is reachable from
      Docker (`--gpus all`), so this was pursued for real, not left as an
      unverified "future work" line. `Dockerfile.gpu`: separate image on
      `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` (CUDA 12.4 + cuDNN 9.1,
      matching ONNX Runtime 1.19.x's *documented* CUDA-EP requirement --
      checked the compatibility table, not guessed) with ORT's GPU tarball
      instead of the CPU one. `OnnxDetector::Options::use_cuda` /
      `--cuda` / `detector.use_cuda` registers the CUDA execution provider;
      no silent CPU fallback if unavailable (`Ort::Exception` propagates --
      verified by running `--cuda` against the CPU-only image and getting a
      clean, immediate failure, not a quiet no-op). Detection output is
      **bit-identical** to CPU on the full sequence (same MOTA/IDF1/IDSW/FP/
      FN for all three association methods) -- a correctness-verified
      comparison, not just "it didn't crash."
    - Result: **`detect` on CUDA EP measured 48.4 ms/frame vs CPU's ~49.5-
      54.3 ms/frame (16 threads) -- essentially no speedup.** First
      measurement attempt (511 ms stereo, 307 ms detect, badly regressed
      *everywhere* including CPU-only stages) coincided with unrelated heavy
      background CPU/GPU load from other work running on this machine at
      the time; re-confirmed the regression was purely that load (re-ran the
      plain CPU image in parallel, saw the identical ~3x slowdown across
      every stage with zero code changed) and re-measured cleanly once it
      cleared. The clean number stands: YOLOv8n is a genuinely tiny model
      (~3.2M params) run at batch size 1, where fixed per-call overhead
      (kernel launches, host<->device transfer, nothing to amortize it
      across) competes directly with the actual compute, and a 16-thread CPU
      running ORT's own optimised (MLAS) kernels is already competitive
      rather than an easy target. Not a dead end, a scoped one: FP16,
      `IOBinding`, CUDA graph capture, or batching could plausibly change
      this and are named rather than left vague, but weren't attempted here
      -- correctly out of scope for "does naively enabling CUDA EP help"
      rather than a gap quietly left unstated.
- [x] **Memory/allocation audit -- conclusion, not a TODO.** The
      profiling above already answers this: `depth`, `promote3d+appearance`
      and `track` are a combined <2 ms/frame, so there's no meaningful
      allocation/copy overhead hiding in this project's own glue code --
      the cost is concentrated in SGBM's own internal compute (`stereo`) and
      ONNX Runtime's own inference (`detect`), both library-internal, not
      inefficiencies in code this project wrote. Chasing micro-allocations
      in the 2 ms/frame that's left would not move the frame-time budget.
- [x] **Fixed frame-time budget, reported honestly against the >= 20 FPS
      target -- not met, and that's the finding, not a gap.** Final,
      GPU-detection-included picture: 204.2 -> 128.2 ms/frame (4.90 ->
      7.80 FPS) excluding the imread artifact, since GPU detection didn't
      move the number further than tiled-stereo + CPU-detect already had.
      ~1.6x over the Phase 6 starting point, honestly short of 20 FPS. The
      ceiling on this hardware without deeper GPU-specific work (FP16/
      IOBinding/graphs, or an OpenCV built with CUDA support for
      `cv::cuda::StereoSGBM` -- this project's apt-installed OpenCV isn't)
      is CPU-bound tiled SGBM plus CPU-or-GPU-equivalent detection, not a
      GPU win sitting unclaimed for lack of trying.

## Phase 7 — Visual odometry  ✅ done

- [x] **Feature extraction + matching (ORB), essential-matrix pose**
      (`VisualOdometry`, `include/s3m/vo/visual_odometry.hpp`): ORB keypoints
      + descriptors on consecutive LEFT frames, `cv::BFMatcher` (Hamming) +
      Lowe's ratio test, `cv::findEssentialMat` + `cv::recoverPose` (5-point,
      RANSAC) for relative rotation and a *unit-norm* translation direction
      -- the classic monocular scale ambiguity.
- [x] **Scale from the stereo baseline, not left unresolved.** This
      project already has a working stereo rig (Phases 1-6), so rather than
      leave translation scale-less or guess it, each frame's ORB keypoints
      are also stereo-triangulated (reusing `StereoMatcher`, including
      Phase 6's tiling for free), and for essential-matrix inlier matches
      with a valid 3D point on both sides, `X_curr = R_rel*X_prev + s*t_hat`
      is solved per-point for the scalar `s`, median over all such points.
      Full derivation and validation in the class-level doc comment and
      below.
- [x] **Validated the pose/scale math in Python before writing a line of
      C++** -- same discipline as every earlier phase, applied here to the
      one genuinely novel piece (the stereo-scale-recovery formula and the
      resulting pose-composition convention aren't textbook the way
      Kabsch/RPE are). Constructed a known synthetic camera motion (rotation
      + translation), projected points through it, ran
      `findEssentialMat`/`recoverPose`/the scale formula/the composition
      formula against the known ground truth: recovered rotation, scale and
      resulting camera position all matched to floating-point precision
      (~1e-14). This is the exact math the C++ implementation carries out.
- [x] **Trajectory vs. dataset ground truth (ATE / RPE)**
      (`include/s3m/vo/trajectory_eval.hpp`): standard Sturm et al. 2012
      definitions (the TUM RGB-D benchmark's own metrics). ATE aligns the
      estimated trajectory onto ground truth via Kabsch/Umeyama (rotation +
      translation, no scale -- this VO already recovers metric scale, so a
      free-scale alignment would hide a real scale error rather than
      measure it) before computing RMSE; RPE compares relative motion over a
      `--rpe-delta`-frame step. Textbook, not project-invented, so validated
      with hand-verified C++ unit tests rather than a separate Python pass:
      a known rotation+translation recovered exactly by `alignRigid`; both
      ATE and RPE proven invariant to a global "gauge" rigid transform
      applied to an entire trajectory (an earlier version of that test
      wrongly expected ATE to blow up under the shift -- it doesn't,
      correctly, since alignment's whole job is to undo exactly that kind of
      shift; the test was wrong, not the code, and was fixed rather than the
      implementation loosened to match a bad expectation).
- [x] **A second, different KITTI benchmark, not the tracking one already in
      use.** Only the *odometry* benchmark ships ego-motion ground truth
      (the tracking benchmark's `label_02` has object boxes, not vehicle
      poses) -- different sequence numbering (00-10 have ground truth),
      different file layout (`sequences/<seq>/image_0|image_1`,
      `poses/<seq>.txt`), grayscale not colour, P0/P1 not P2/P3.
      `scripts/fetch_kitti_odometry.py` reuses the same HTTP-range-request
      trick as the tracking fetch script (23 GB archive, ~140 MB needed for
      one short sequence). `readKittiCalib` gained optional `left_key`/
      `right_key` parameters (default P2/P3, unchanged for every existing
      caller) instead of a duplicate parser, since the two formats are
      otherwise identical; `KittiOdometrySource`/`readKittiPoses` are new,
      the odometry benchmark's layout being different enough (times.txt,
      poses.txt, no labels) to not fit the tracking loader's shape.
- [x] **Real run: KITTI odometry sequence 04** (271 frames, the shortest
      sequence with ground truth -- chosen deliberately for a fast, complete
      iteration loop rather than sequence 00's usual 4541-frame showcase
      length). `benchmark_vo --kitti-root data/kitti_odometry --sequence 04`:
      961.7 mean ORB matches/frame, 7/271 frames coasted (untrusted, held
      last pose) -- healthy tracking. Recovered scale: 1.32 m/step median,
      i.e. ~13.2 m/s at KITTI's ~10 Hz -- a physically plausible driving
      speed, a real sanity check the scale-recovery pipeline is measuring
      something real, not producing an arbitrary number.
      **ATE RMSE 15.7 m** over the sequence (mean 13.5 m) -- roughly 4-5% of
      the ~360 m path length. **RPE (1-frame step): 0.66 m translation,
      0.16° rotation; RPE (10-frame step): 2.49 m, 0.59°.** Honest context:
      this is frame-to-frame VO with no bundle adjustment, no keyframing, no
      loop closure -- pure per-frame drift accumulation with nothing
      correcting it, so this ATE is expected to be well behind full
      SLAM-grade systems (which report well under 1% on KITTI odometry) and
      is not compared against the official KITTI odometry leaderboard here,
      which scores length-normalised segments (100-800 m) under its own
      devkit, a different protocol from the frame-count-based RPE this
      project reports. What this result actually demonstrates: the
      geometry (essential matrix + stereo scale recovery) is correct and
      physically sane, not that this is a competitive SLAM system --
      bundle adjustment / loop closure would be the natural next layer, not
      attempted here (out of scope for "does the core geometry work").
- [x] Touches the VI-SLAM / camera-pose requirements named in the
      project's original ASELSAN-facing scope.

## Phase 8 — ROS2 integration (optional / bonus)  ✅ done

- [x] **`/stereo/left`, `/stereo/right` in; `/perception/detections`,
      `/perception/tracks`, `/perception/pointcloud` out** -- exactly the
      topic set this line originally specified, built after Phases 1-7 gave
      the perception stack something real worth wrapping, not before.
      `ros2/s3m_ros2/`: a deliberately thin ROS2 (Humble/ament_cmake)
      package -- `perception_node.cpp` is message<->s3m-type conversion and
      wiring, `message_filters::ApproximateTime` synchronises the stereo
      pair, then the SAME `s3m::s3m` library every other app in this repo
      links against runs stereo depth -> ONNX detection -> `promoteTo3D` ->
      `Tracker`, exactly as `benchmark_kitti`/`stereo_depth_demo` do. No
      perception logic was reimplemented for ROS2; this package has none of
      its own.
- [x] **A real ROS2 build, not a "should work" package left unverified.**
      Separate `Dockerfile.ros2` (`ros:humble-ros-base`, matching the
      project's existing Ubuntu 22.04 base) + `colcon build`. Hit one real
      issue on the first attempt: the package's CMakeLists.txt included the
      main project via a relative `"../.."` path from
      `CMAKE_CURRENT_SOURCE_DIR`, which works for a plain checkout but not
      through colcon's `src/s3m_ros2 -> <repo>/ros2/s3m_ros2` symlink --
      `CMAKE_CURRENT_SOURCE_DIR` reflects the symlink's own location in the
      colcon workspace, not the real path behind it, so `"../.."` resolved
      to the colcon workspace root instead of the repo. Fixed with an
      explicit `S3M_REPO_ROOT` CMake variable (passed as
      `-DS3M_REPO_ROOT=/src` for this project's own Docker-mounted build;
      the relative path stays as a fallback for a plain, symlink-free
      checkout). Clean build on the second attempt -- every `vision_msgs`/
      `cv_bridge` field name used in `perception_node.cpp` was correct on
      the first try; the only real bug was the path issue.
- [x] **Runtime-verified, not just compiled.** `ros2/s3m_ros2/test/
      publish_synthetic_pair.py`: launched the real node (real ONNX model
      loaded) inside the built image, published 5 synthetic stereo pairs
      (a striped test pattern -- textured enough for real SGBM matching,
      deliberately containing no actual COCO objects), and subscribed to
      all three output topics. Result: `/perception/pointcloud` received 2
      messages carrying 97,152 real 3D points each (~79% density of a
      640x192 frame -- real stereo matching ran, not a stub);
      `/perception/detections` and `/perception/tracks` each received 3
      messages, correctly empty (0 items) every time -- the detector
      running and correctly finding nothing, since the striped test pattern
      genuinely has no COCO-class objects in it, not a sign anything
      failed. This confirms the ROS2 *wiring* end to end (message
      conversion, synchronisation, publishing); the detector's own
      correctness on real content is already established elsewhere in this
      project (Phase 2's tests, every real-KITTI run in Phases 4-6) and
      wasn't worth re-proving here with a real-object test image -- the
      thing this test needed to catch (and did have a real bug to catch,
      the CMake path issue above) is ROS2-specific plumbing, not detection
      accuracy.
- [x] **Named, not hidden, simplifications** (stated in
      `perception_node.cpp`'s own header comment): camera intrinsics come
      from ROS parameters, not a subscribed `sensor_msgs/CameraInfo` topic
      -- every other app in this project also takes calibration from a
      file, not a live topic, so this matches the project's existing
      pattern rather than under-building relative to it; `Detection3D` has
      no measured 3D extent (position only, the same convention the
      tracker and metrics already use throughout this project), so
      published bounding boxes carry a fixed nominal size rather than an
      invented measured one.

## Non-goals (for now)

- Full bundle adjustment / global SLAM back-end.
- Learned stereo (RAFT-Stereo etc.) — the ONNX seam would allow it later.
- Multi-camera (>2) rigs.
