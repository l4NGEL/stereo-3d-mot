# Experiment inventory

What was used, where it came from, what was chosen by looking at which data, and what is missing.
Machine-readable companion: [`results/inventory.json`](results/inventory.json) (hashes, digests, versions).
Definitions and thresholds: [`PROTOCOL.md`](PROTOCOL.md). Results: [`REPORT.md`](REPORT.md).

## 1. Code under evaluation

| Item | Value |
|---|---|
| Repository | `stereo-3d-mot`, branch `main` |
| Protocol frozen at | `90544e0` (2026-09-26 15:14) |
| Harness + dev-only selection | `e8f6613` (2026-09-26 15:36) — committed before any held-out frontend ran |
| Build | CMake `Release`, `-Werror`, GCC 11.4.0, CMake 3.22.1, Ubuntu 22.04 container (`stereo-3d-mot:deps-onnx`) |
| Libraries | OpenCV 4.5.4, Eigen 3, ONNX Runtime 1.19.2 (CPU EP; CUDA EP in `stereo-3d-mot:deps-onnx-gpu`) |
| Note | the held-out VO runs used a `benchmark_vo` built with a one-line fix (the `--frames` limit is now checked before reading the next frame; the old order threw on a partially downloaded sequence). It does not change any computed value. |

## 2. Detector and weights

| Item | Value |
|---|---|
| File | `models/yolov8n.onnx`, 12,851,086 bytes, sha256 `542ab2cb76e6f9ca24c870e0a4480c50abc91154ca6be4e05f461c1edbcc9c69` |
| Origin (from the file's own ONNX metadata) | "Ultralytics YOLOv8n model trained on coco.yaml", Ultralytics 8.4.9, exported 2026-09-10 with `scripts/export_yolov8.py`, opset 12, input 1×3×640×640, output 1×84×8400, 80 COCO classes |
| Training / fine-tuning in this project | none. The weights are the unmodified pretrained ones. KITTI is not part of COCO. |
| Pre/post-processing | letterbox to 640, class scores, class-aware NMS 0.45, score floor 0.25 (0.01 for the evaluation dumps), max 300 detections, argmax class kept only if in `keep_classes` {0, 1, 2, 3, 5, 7} |
| Not verified | that the exported file is bit-for-bit the official checkpoint (only the metadata and the hash are recorded) |

## 3. Datasets, splits, sizes

| Dataset | Path | Used for | Role | Files / bytes | Manifest digest (sha256 over path + file hash) |
|---|---|---|---|---|---|
| KITTI tracking (images left+right, `label_02`, calib) | `data/kitti` | tracking, detection, object-level depth, speed | **dev**: 0000, 0003, 0017 (all frames, 154/144/145). **held-out**: 0001, 0002, 0004, 0007, 0008, 0010, 0012, 0016 (first 150 frames; 0012 has 78) | 3,184 / 2.60 GB | `a1708e77…76d9` |
| KITTI odometry (grayscale pair, poses, calib) | `data/kitti_odometry` | visual odometry | **dev**: 04 (all 271 frames). **held-out**: 00, 06, 07, 08, 10 (first 500 frames each) | 5,597 / 1.46 GB | `b1afb169…cabd85` |
| KITTI Stereo 2015 training | `data/kitti_stereo2015` | real-scene disparity | 200 pairs, frame `_10`, `disp_occ_0` / `disp_noc_0` ground truth. Never used for tuning. | 1,000 / 0.40 GB | `eb7c1134…0b15c76` |
| Synthetic stereo scene | generated (`SyntheticStereoSource`) | depth regression gate | seeds 42, 1, 2, 3, 4 × 30 frames | — | — |

Held-out selection: `numpy.random.RandomState(20260926).choice(<non-dev sequences>, k, replace=False)`, 8 of the 18 non-dev
tracking sequences and 5 of the 10 non-dev odometry sequences with ground truth, drawn before anything was downloaded.
KITTI's own test-split sequences have no public labels and are not used.

**Timing of the data, from file timestamps** (details and the correction this forced are in `PROTOCOL.md` §10): the held-out
tracking images were downloaded at 12:05–12:3x, before the protocol commit (15:14), but nothing was opened or processed until
15:38; the labels of all 21 tracking sequences had been on disk since Phase 4 and their occlusion field had been surveyed in
aggregate in Phase 5 to choose dev sequences.

## 4. Where each tuned or fixed value came from

| Parameter | Value | Provenance (git) | Dev-influenced? |
|---|---|---|---|
| SGBM: 128 disparities, block 5, P1/P2 multipliers 8/32, uniqueness 10, speckle 100/2, disp12 1, pre-filter cap 63 | `configs/default.yaml`, Phase 1 (2026-09-09) | OpenCV conventions; no trace of tuning on KITTI. Whether they were tuned on the synthetic scene is not recorded. | no evidence |
| `num_tiles` = 4 | Phase 6 (2026-09-15) | chosen for speed on KITTI 0000; accuracy neutrality checked on the synthetic scene and 0000 | dev (speed only) |
| detector score 0.25, NMS 0.45, input 640 | Phase 2 | Ultralytics defaults | no |
| `keep_classes` {0,1,2,3,5,7} | `configs/kitti.yaml`, commit `079e762` (2026-09-11) | added after the first KITTI 0000 run and its frame-0 spot check | **yes (0000)** |
| `measurement_noise` 1.0 m | same commit | raised from 0.05 after the first 0000 run | **yes (0000)** |
| `process_noise` 1.0, `max_age` 30, `min_hits` 3, `dt` 0.1 | Phase 3 | defaults from the synthetic scene | no |
| `gating_chi2` 7.815, `iou_gate` 0.3 | Phase 3, `61fc5c6` (2026-09-11) | χ²(3 dof, 95 %) and the SORT convention | no |
| fused weights 0.5 / 0.2 / 0.3 | Phase 5, `69021ba` (2026-09-14) | the roadmap says they were fixed by reasoning before looking at KITTI numbers; git cannot confirm the order (same commit as the first fused results) | unverifiable |
| VO: ORB 2000 features, ratio 0.75, RANSAC 0.999 / 1.0 px, min inliers 15 | Phase 7, `0151d8d` (2026-09-23) | run first on sequence 04 | dev (04) |
| association ∈ {…}, emission E, score threshold, `min_hits` | this evaluation | selected on dev only by the pre-registered rule (`results/dev_selection.json`): IoU-2D, E = 0, 0.25, 3 | dev only |

## 5. Hardware and software of the measurements

| Item | Value |
|---|---|
| CPU / RAM | Intel Core i9-9880H (8 cores / 16 threads, 2.3 GHz nominal, laptop), 31.9 GB |
| GPU | NVIDIA GeForce RTX 2080 (8 GB), driver 592.82. Not used by the default path; used only for the CUDA rows. |
| Host | Windows 10 (19045), Docker Desktop 29.1.3 (WSL2 backend) |
| Host Python for scoring | 3.10.8, numpy 2.2.6, scipy 1.15.3, pandas 2.3.3, py-motmetrics 1.4.0, opencv-python 5.0.0, onnx 1.22.0 |
| State during the speed runs | no other job of ours; unrelated desktop apps and two unrelated database/backend containers idle. GPU load from another workload was seen at times (see the CUDA caveat in the report). |

## 6. Commands

Everything is produced by [`scripts/eval/run_all.sh`](../../scripts/eval/run_all.sh) (all stages, or `--score-only` for the Python
scoring stage from the dumps). Main entry points:

| Stage | Command |
|---|---|
| dumps: detections, timings, every tracker variant | `apps/eval_kitti --kitti-root … --sequence … --config configs/kitti_tiled.yaml --variants configs/eval/variants_{dev,test}.txt` |
| dumps: disparity maps | `apps/dump_disparity --source synthetic --seed S --frames 30` / `--stereo2015 <root>` |
| dumps: visual odometry | `apps/benchmark_vo --kitti-root … --sequence … --frames 500 --out …` |
| latency | `scripts/eval/run_speed.sh` (→ `apps/benchmark_e2e`) |
| scoring | `scripts/eval/{regress_legacy,dev_selection,run_tracking_eval,run_detection,run_object_depth,run_diagnosis,run_stereo_synthetic,run_stereo_kitti2015,run_vo,run_speed_eval,check_counters,make_verdicts}.py` |
| tests of the scorers | `cd scripts/eval && python -m pytest tests` (19 tests) |

## 7. Leakage check

| Question | Answer |
|---|---|
| Was the detector trained or fine-tuned on KITTI by this project? | No. Unmodified COCO-pretrained weights; hash recorded. |
| Was anything selected on the held-out set? | No. Selection used dev only and was committed (`e8f6613`) before the first held-out output existed. |
| Are dev and held-out disjoint? | Yes (different sequences). Sequences of the same KITTI drives can be adjacent in time; no such adjacency check was made. |
| Were held-out labels ever seen? | Aggregate occlusion statistics of all sequences' labels, in Phase 5 (see §3). No per-frame inspection of held-out data before the freeze. |
| Choices that were made looking at dev data | `keep_classes`, `measurement_noise` (0000), `num_tiles` (0000 speed), VO parameters (04). They are used unchanged on the held-out set. |

## 8. Missing data / not measured, and how to get it

| Missing | Why | How to obtain / do |
|---|---|---|
| COCO mAP of the detector | the project never runs COCO validation | `yolo val model=yolov8n.pt data=coco.yaml` (needs COCO val2017, ~1 GB); would still not be a KITTI number |
| Official KITTI tracking leaderboard protocol (HOTA, devkit ignore rules) | not implemented; the protocol here is KITTI-*like* | run the official `devkit_tracking` on a results file (needs the test split via the KITTI server) |
| Per-pixel depth accuracy on the tracking frames | no LiDAR downloaded | `data_tracking_velodyne.zip` (35 GB; per-sequence range extraction as in `fetch_kitti_sequence.py`) + calibration `Tr_velo_cam` |
| Full-length odometry (100–800 m segments, all of 00–10) | only prefixes downloaded | `scripts/fetch_kitti_odometry.py NN` (full sequence, 0.5–2.4 GB each) |
| Speed on target hardware, pipelined throughput, power | one laptop, sequential loop | rerun `run_speed.sh` on the target; pipelining is not implemented |
| Adverse conditions (night, rain), other cameras, other detectors | not in KITTI training data used / out of scope | new data |
| Learned ReID | the appearance cue is an HSV histogram | out of scope |
