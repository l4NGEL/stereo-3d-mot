#!/usr/bin/env bash
# Regenerates every number in docs/evaluation/REPORT.md.
#
#   scripts/eval/run_all.sh              all stages (needs Docker, the datasets and models/yolov8n.onnx)
#   scripts/eval/run_all.sh --score-only only the Python scoring stage, from the dumps under out/eval/raw
#
# Stages (see docs/evaluation/PROTOCOL.md and INVENTORY.md for what each measures):
#   0 data       scripts/fetch_kitti_sequence.py / fetch_kitti_odometry.py / fetch_kitti_stereo2015.py
#   1 build      Release, -Werror, in the stereo-3d-mot:deps-onnx image (build dir in a named volume)
#   2 dumps      eval_kitti (dev + held-out, tiled and untiled), dump_disparity, benchmark_vo, benchmark_e2e
#   3 scoring    scripts/eval/*.py -- independent of the C++ code that produced the dumps
#
# Run from the repository root. The committed numbers were produced by exactly these commands.
set -euo pipefail

DEV="0000 0003 0017"
TEST="0001 0002 0004 0007 0008 0010 0012 0016"
RAW=out/eval/raw
KITTI=data/kitti
R=docs/evaluation/results

if [ "${1:-}" != "--score-only" ]; then
    # ---- 0 data (skips files already on disk) ----
    bash scripts/download_kitti.sh 0000 || true            # calib + labels of all training sequences (small)
    python scripts/fetch_kitti_sequence.py $DEV --out "$KITTI"
    python scripts/fetch_kitti_sequence.py $TEST --max-frames 150 --out "$KITTI"
    python scripts/fetch_kitti_odometry.py 04 --out data/kitti_odometry
    python scripts/fetch_kitti_odometry.py 00 06 07 08 10 --max-frames 500 --out data/kitti_odometry
    python scripts/fetch_kitti_stereo2015.py --out data/kitti_stereo2015

    # ---- 1 build ----
    export MSYS_NO_PATHCONV=1
    SRC="$(pwd -W 2>/dev/null || pwd)"
    IMG=stereo-3d-mot:deps-onnx
    RUN="docker run --rm -v $SRC:/src:ro -v $SRC/data:/data:ro -v $SRC/$RAW:/out -v s3m-eval-build:/b -w /src $IMG"
    mkdir -p "$RAW"
    $RUN bash -c "cmake -S /src -B /b -G Ninja -DCMAKE_BUILD_TYPE=Release -DS3M_WARNINGS_AS_ERRORS=ON -DONNXRUNTIME_ROOT=/opt/onnxruntime && cmake --build /b --parallel 12"

    # ---- 2 dumps ----
    for CFG in kitti kitti_tiled; do
        for S in $DEV; do
            $RUN /b/apps/eval_kitti --kitti-root /data/kitti --sequence "$S" --config "configs/$CFG.yaml" \
                --model /src/models/yolov8n.onnx --variants configs/eval/variants_dev.txt --out "/out/dev_$CFG"
        done
        for S in $TEST; do
            $RUN /b/apps/eval_kitti --kitti-root /data/kitti --sequence "$S" --config "configs/$CFG.yaml" \
                --model /src/models/yolov8n.onnx --variants configs/eval/variants_test.txt --frames 150 --out "/out/test_$CFG"
        done
    done
    for S in $DEV; do
        $RUN bash -c "/b/apps/benchmark_kitti --kitti-root /data/kitti --sequence $S --config configs/kitti.yaml --model /src/models/yolov8n.onnx > /out/cpp_benchmark_kitti/$S.txt" || true
    done
    $RUN bash -c 'for S in 42 1 2 3 4; do for T in 1 4; do /b/apps/dump_disparity --source synthetic --seed $S --frames 30 --config configs/default.yaml --tiles $T --out /out/stereo/synth_seed${S}_tiles${T}; done; done'
    $RUN bash -c '/b/apps/dump_disparity --stereo2015 /data/kitti_stereo2015 --config configs/kitti_tiled.yaml --out /out/stereo/kitti2015_sgbm_tiles4 && /b/apps/dump_disparity --stereo2015 /data/kitti_stereo2015 --config configs/kitti.yaml --out /out/stereo/kitti2015_sgbm_tiles1 && /b/apps/dump_disparity --stereo2015 /data/kitti_stereo2015 --config configs/kitti.yaml --matcher BM --block 15 --out /out/stereo/kitti2015_bm15'
    $RUN bash -c '/b/apps/benchmark_vo --kitti-root /data/kitti_odometry --sequence 04 --config configs/kitti_tiled.yaml --out /out/vo/dev_tiled_04 > /out/vo/dev_tiled_04.log; /b/apps/benchmark_vo --kitti-root /data/kitti_odometry --sequence 04 --config configs/default.yaml --out /out/vo/dev_untiled_04 > /out/vo/dev_untiled_04.log; for S in 00 06 07 08 10; do /b/apps/benchmark_vo --kitti-root /data/kitti_odometry --sequence $S --frames 500 --config configs/kitti_tiled.yaml --out /out/vo/test_tiled_$S > /out/vo/test_tiled_$S.log; done'
    # speed: the machine must be otherwise idle (see PROTOCOL.md 3.7); CPU image, then the GPU image
    for T in "tiled_local 3 configs/kitti_tiled.yaml" "untiled_local 1 configs/kitti.yaml"; do
        set -- $T
        docker run --rm -e TAG="$1" -e REPS="$2" -e CONFIG="$3" -e MODE=local -v "$SRC:/src:ro" -v "$SRC/data:/data:ro" \
            -v "$SRC/$RAW/speed:/out" -v s3m-eval-build:/b -w /src $IMG bash /src/scripts/eval/run_speed.sh
    done
    docker run --rm -e TAG=tiled_bind -e REPS=1 -e MODE=bind -e SEQS=0000 -v "$SRC:/src:ro" -v "$SRC/data:/data:ro" \
        -v "$SRC/$RAW/speed:/out" -v s3m-eval-build:/b -w /src $IMG bash /src/scripts/eval/run_speed.sh
    # GPU image (build there first: cmake -S /src -B /b -DONNXRUNTIME_ROOT=/opt/onnxruntime, volume s3m-eval-build-gpu)
    docker run --rm --gpus all -e TAG=cuda_local -e REPS=1 -e MODE=local -e EXTRA=--cuda -v "$SRC:/src:ro" -v "$SRC/data:/data:ro" \
        -v "$SRC/$RAW/speed:/out" -v s3m-eval-build-gpu:/b -w /src stereo-3d-mot:deps-onnx-gpu bash /src/scripts/eval/run_speed.sh
fi

# ---- 3 scoring (pure Python: numpy, scipy, pandas, py-motmetrics, opencv-python) ----
python scripts/eval/regress_legacy.py --raw "$RAW/dev_kitti" --cpp-dir "$RAW/cpp_benchmark_kitti" --out "$R/tr6_legacy_regression.json"
python scripts/eval/dev_selection.py --raw "$RAW/dev_kitti_tiled" --out-csv "$R/dev_selection_grid.csv" --out-json "$R/dev_selection.json"
python scripts/eval/run_tracking_eval.py --raw "$RAW/dev_kitti_tiled" --seqs $DEV --kind dev --n-boot 2000 --out "$R/tracking_dev.json"
python scripts/eval/run_tracking_eval.py --raw "$RAW/test_kitti_tiled" --seqs $TEST --kind test --n-boot 10000 --out "$R/tracking_test.json"
python scripts/eval/run_tracking_eval.py --raw "$RAW/test_kitti" --seqs $TEST --kind test --n-boot 200 --out "$R/tracking_test_untiled.json"
python scripts/eval/run_detection.py --raw "$RAW/dev_kitti_tiled" --seqs $DEV --out "$R/detection_dev.json"
python scripts/eval/run_detection.py --raw "$RAW/test_kitti_tiled" --seqs $TEST --out "$R/detection_test.json"
python scripts/eval/run_object_depth.py --raw "$RAW/dev_kitti_tiled" --seqs $DEV --out "$R/object_depth_dev.json"
python scripts/eval/run_object_depth.py --raw "$RAW/test_kitti_tiled" --seqs $TEST --out "$R/object_depth_test.json"
python scripts/eval/run_diagnosis.py --raw "$RAW/dev_kitti_tiled" --seqs $DEV --kind dev --out "$R/diagnosis_dev.json"
python scripts/eval/run_diagnosis.py --raw "$RAW/test_kitti_tiled" --seqs $TEST --kind test --out "$R/diagnosis_test.json"
python scripts/eval/run_stereo_synthetic.py
python scripts/eval/run_stereo_kitti2015.py
python scripts/eval/run_vo.py
python scripts/eval/run_speed_eval.py
python scripts/eval/run_variant_cost.py
python scripts/eval/check_counters.py
python scripts/eval/apply_review_labels.py
python scripts/eval/make_verdicts.py
python scripts/eval/embed_verdicts.py
python scripts/eval/embed_appendix.py
