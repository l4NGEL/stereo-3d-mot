| ID | Criterion (from PROTOCOL.md §7) | Measured | Verdict |
|---|---|---|---|
| SP-1 | end-to-end >= 20 FPS incl. decode (each of 3 repetitions) | 5.84 FPS (reps [5.63, 6.0, 5.92]) | **KALDI** |
| SP-2 | p95 latency <= 50 ms (each repetition) | p95 205.0 ms (median 165.8) | **KALDI** |
| SP-3 | compute-only (no decode) >= 20 FPS | 6.87 FPS | **KALDI** |
| DE-1 | Vehicle precision >= 0.80 and recall >= 0.70 (IoU 0.5, moderate, score >= 0.25) | P 0.788 [0.772,0.802], R 0.761 — recall passes; precision misses | **KALDI** |
| DE-2 | Vehicle AP50 (R40) >= 0.56 | 0.798 (all-point 0.812) | **GEÇTİ** |
| DE-3 | Person precision >= 0.75 and recall >= 0.50; AP50 (R40) >= 0.375 | P 0.709, R 0.679, AP50 0.652 | **KALDI** |
| DE-4 | COCO mAP | not measured in this project — only KITTI-protocol numbers are claimed; Ultralytics' published COCO mAP is not a KITTI number | **ÖLÇÜLEMEDİ** |
| TR-1 | minimum: pooled P-K2D MOTA > 0 (dev-selected system, held-out) | MOTA +0.492 | **GEÇTİ** |
| TR-2 | target: Vehicle MOTA >= 0.50 and IDF1 >= 0.60 | MOTA 0.519, IDF1 0.591 | **KALDI** |
| TR-3 | target: Person MOTA >= 0.30 and IDF1 >= 0.50 | MOTA 0.446, IDF1 0.487 | **KALDI** |
| TR-4 | IDSW <= 2 % of GT objects (pooled) | 246/4550 = 5.41 % | **KALDI** |
| TR-5 | beats SORT-2D on pooled MOTA and IDF1, bootstrap interval excluding 0 | MOTA -0.059 [-0.129,-0.026], IDF1 -0.070 [-0.147,-0.032] — S* is significantly WORSE than the depth-free baseline | **KALDI** |
| TR-6 | harness reproduces published legacy counts; independent counter == py-motmetrics | legacy regression PASS; counters 347 inputs, 0 disagreements | **GEÇTİ** |
| TR-7a | claim: 3D Mahalanobis association beats plain 2D IoU (A4 vs S*) | MOTA -0.057 [-0.160,-0.023]; IDF1 +0.003 [-0.152,+0.099] — contradicted on MOTA | **KALDI** |
| TR-7b | claim: fusion (3D+IoU+appearance) beats the selected plain IoU (F vs S*) | MOTA +0.005 [-0.040,+0.033]; IDF1 +0.067 [-0.026,+0.106] — inside the noise | **KANITLANMADI** |
| TR-7c | claim: the appearance cue helps (F vs A1) | MOTA +0.003 [+0.000,+0.007]; IDF1 +0.005 [-0.002,+0.013] — inside the noise | **KANITLANMADI** |
| TR-7d | claim: the IoU term of the fused cost helps (F vs A2) | MOTA +0.010 [+0.001,+0.028]; IDF1 +0.004 [-0.031,+0.068] — demonstrated on MOTA only | **GEÇTİ** |
| TR-7e | claim: the 3D term of the fused cost helps (F vs A3) | MOTA -0.002 [-0.016,+0.008]; IDF1 -0.003 [-0.079,+0.037] — inside the noise | **KANITLANMADI** |
| TR-7f | claim: Hungarian beats greedy assignment (F vs A7) | MOTA +0.003 [-0.002,+0.010]; IDF1 +0.009 [-0.008,+0.040] — inside the noise | **KANITLANMADI** |
| TR-7g | claim: emitting only updated tracks (E=0) beats the shipped E=30 (S*) | MOTA +1.387 [+0.645,+2.982]; IDF1 +0.206 [+0.090,+0.364] — demonstrated on MOTA AND IDF1 | **GEÇTİ** |
| TR-7h | claim: tiled stereo does not change tracking accuracy (A9) | max abs change in MOTA 0.0070 and in IDF1 0.0035 over 11 systems; bit-identical for 4 — practically identical, not bit-identical, on held-out data | **GEÇTİ** |
| VO-1 | t_rel <= 5 % (minimum) / <= 2.5 % (target), 100-400 m segments, held-out pooled | 13.78 % (421 segments) — target tier not met | **KALDI** |
| VO-2 | r_rel <= 0.03 deg/m (minimum) / <= 0.015 (target) | 0.0796 deg/m — target tier not met | **KALDI** |
| VO-3 | >= 95 % of frames tracked on every held-out sequence | min 98.2 % — {"test_tiled_00": 99.6, "test_tiled_06": 99.2, "test_tiled_07": 99.4, "test_tiled_08": 98.2, "test_tiled_10": 99.0} | **GEÇTİ** |
| VO-4 | no held-out sequence with t_rel > 10 % | worst 19.69 % {"test_tiled_00": 19.36, "test_tiled_06": 18.07, "test_tiled_07": 3.36, "test_tiled_08": 8.26, "test_tiled_10": 19.69} | **KALDI** |
| ST-1 | synthetic (5 seeds x 30 frames, project ROI): bad-2.0 <= 2 %, RMSE <= 1.5 px, density >= 95 % | bad-2.0 0.85 %, RMSE 1.281 px (worst seed 1.400), density 98.90 % — regression gate on a benign scene; low evidential value | **GEÇTİ** |
| ST-2 | KITTI Stereo 2015 D1-all (holes = errors) <= 25 % and better than the BM baseline | SGBM(4 tiles) 20.37 % (valid-only 5.24 %, density 84.0 %); BM 62.90 % — BM baseline is weak by construction (density only 39 %) | **GEÇTİ** |
| ST-3 | Vehicle 5-30 m: RMSE(detection depth - label front-surface depth) <= 1.0 m (the tracker's configured sigma) | RMSE 1.50 m (median signed +1.11 m, n=1515); vs cuboid centre RMSE 1.16 m | **KALDI** |
| ST-4 | \|D1-all(4 tiles) - D1-all(untiled)\| <= 0.5 pp | 0.108 pp | **GEÇTİ** |
| IN-1 | every headline number regenerates from one documented command | scripts/eval/run_all.sh --score-only re-run from the dumps: 24 result files, 0 differ — scoring stage re-executed; dump stage regenerated for dev (legacy regression) and checked for determinism on one sequence, not re-run end-to-end for every sequence | **GEÇTİ** |
| IN-2 | two identical runs give bit-identical dumps | dets + 8 tracker dumps of sequence 0012 identical (byte compare) — checked on one held-out sequence (78 frames) under load | **GEÇTİ** |
| IN-3 | no leakage: unmodified pretrained detector; selection on dev only | model metadata: 'Ultralytics YOLOv8n model trained on coco.yaml'; selection committed before held-out frontend — dev-influenced choices are declared in INVENTORY.md | **GEÇTİ** |
