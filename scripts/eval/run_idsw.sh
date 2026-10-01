#!/usr/bin/env bash
# Regenerates every number in docs/evaluation/IDSW_REPORT.md (plan: IDSW_PLAN.md).
# Run from the repository root; needs Docker, data/kitti, models/yolov8n.onnx (as run_all.sh).
set -euo pipefail

DEV="0000 0003 0017"
TEST="0001 0002 0004 0007 0008 0010 0012 0016"
RAW=out/eval/raw/idsw
R=docs/evaluation/results/idsw
mkdir -p "$RAW" "$R"

if [ "${1:-}" != "--score-only" ]; then
    export MSYS_NO_PATHCONV=1
    SRC="$(pwd -W 2>/dev/null || pwd)"
    IMG=stereo-3d-mot:deps-onnx
    RUN="docker run --rm -v $SRC:/src:ro -v $SRC/data:/data:ro -v $SRC/$RAW:/out -v s3m-eval-build:/b -w /src $IMG"
    $RUN bash -c "cmake -S /src -B /b -G Ninja -DCMAKE_BUILD_TYPE=Release -DS3M_WARNINGS_AS_ERRORS=ON -DONNXRUNTIME_ROOT=/opt/onnxruntime && cmake --build /b --parallel 12 && ctest --test-dir /b"
    for S in $DEV; do
        $RUN /b/apps/eval_kitti --kitti-root /data/kitti --sequence "$S" --config configs/kitti_tiled.yaml \
            --model /src/models/yolov8n.onnx --variants configs/eval/variants_idsw_dev.txt --diag --out /out/dev_kitti_tiled
    done
    for S in $TEST; do
        $RUN /b/apps/eval_kitti --kitti-root /data/kitti --sequence "$S" --config configs/kitti_tiled.yaml \
            --model /src/models/yolov8n.onnx --variants configs/eval/variants_idsw_test.txt --frames 150 --diag --out /out/test_kitti_tiled
    done
    # baseline integrity: same detections and bit-identical S* tracks as the published dumps
    for S in $DEV; do
        cmp "$RAW/dev_kitti_tiled/$S/dets.txt" "out/eval/raw/dev_kitti_tiled/$S/dets.txt"
        cmp "$RAW/dev_kitti_tiled/$S/tracks_A_iou_last.txt" "out/eval/raw/dev_kitti_tiled/$S/tracks_iou2d_s0.25_h3.txt"
    done
    for S in $TEST; do
        cmp "$RAW/test_kitti_tiled/$S/dets.txt" "out/eval/raw/test_kitti_tiled/$S/dets.txt"
        cmp "$RAW/test_kitti_tiled/$S/tracks_A_iou_last.txt" "out/eval/raw/test_kitti_tiled/$S/tracks_A5_plain_iou.txt"
    done
fi

DEV_V=$(grep -v '^#' configs/eval/variants_idsw_dev.txt | awk '{print $1}')
TEST_V=$(grep -v '^#' configs/eval/variants_idsw_test.txt | awk '{print $1}')
python scripts/eval/run_idsw_eval.py --raw "$RAW/dev_kitti_tiled" --seqs $DEV --variants $DEV_V --mode select \
    --out "$R/dev_selection.json" --events-csv "$R/dev_events.csv"
python scripts/eval/run_idsw_eval.py --raw "$RAW/test_kitti_tiled" --seqs $TEST --variants $TEST_V --mode report \
    --selection "$R/dev_selection.json" --n-boot 10000 --out "$R/test_report.json" --events-csv "$R/test_events.csv"

# ---- round 2 (IDSW_PLAN.md): reconfirm_after_miss x max_age ----
if [ "${1:-}" != "--score-only" ]; then
    for S in $DEV; do
        $RUN /b/apps/eval_kitti --kitti-root /data/kitti --sequence "$S" --config configs/kitti_tiled.yaml \
            --model /src/models/yolov8n.onnx --variants configs/eval/variants_idsw2_dev.txt --out /out/dev2_kitti_tiled
    done
    for S in $TEST; do
        $RUN /b/apps/eval_kitti --kitti-root /data/kitti --sequence "$S" --config configs/kitti_tiled.yaml \
            --model /src/models/yolov8n.onnx --variants configs/eval/variants_idsw2_test.txt --frames 150 --out /out/test2_kitti_tiled
    done
fi
DEV2_V=$(grep -v '^#' configs/eval/variants_idsw2_dev.txt | awk '{print $1}')
TEST2_V=$(grep -v '^#' configs/eval/variants_idsw2_test.txt | awk '{print $1}')
python scripts/eval/run_idsw_eval.py --round 2 --raw "$RAW/dev2_kitti_tiled" --seqs $DEV --variants $DEV2_V --mode select \
    --out "$R/dev2_selection.json" --events-csv "$R/dev2_events.csv"
python scripts/eval/run_idsw_eval.py --round 2 --raw "$RAW/test2_kitti_tiled" --seqs $TEST --variants $TEST2_V --mode report \
    --selection "$R/dev2_selection.json" --out "$R/test2_report.json" --events-csv "$R/test2_events.csv"

# ---- round 3 (IDSW_PLAN.md): reconfirm x sort_box x max_total ----
if [ "${1:-}" != "--score-only" ]; then
    for S in $DEV; do
        $RUN /b/apps/eval_kitti --kitti-root /data/kitti --sequence "$S" --config configs/kitti_tiled.yaml \
            --model /src/models/yolov8n.onnx --variants configs/eval/variants_idsw3_dev.txt --out /out/dev3_kitti_tiled
    done
    for S in $TEST; do
        $RUN /b/apps/eval_kitti --kitti-root /data/kitti --sequence "$S" --config configs/kitti_tiled.yaml \
            --model /src/models/yolov8n.onnx --variants configs/eval/variants_idsw3_test.txt --frames 150 --out /out/test3_kitti_tiled
    done
fi
DEV3_V=$(grep -v '^#' configs/eval/variants_idsw3_dev.txt | awk '{print $1}')
TEST3_V=$(grep -v '^#' configs/eval/variants_idsw3_test.txt | awk '{print $1}')
python scripts/eval/run_idsw_eval.py --round 3 --raw "$RAW/dev3_kitti_tiled" --seqs $DEV --variants $DEV3_V --mode select \
    --out "$R/dev3_selection.json" --events-csv "$R/dev3_events.csv"
python scripts/eval/run_idsw_eval.py --round 3 --raw "$RAW/test3_kitti_tiled" --seqs $TEST --variants $TEST3_V --mode report \
    --selection "$R/dev3_selection.json" --out "$R/test3_report.json" --events-csv "$R/test3_events.csv"

# ---- round 4 (IDSW_PLAN.md): wider dev (11), fresh test (10 never-used sequences) ----
DEV4="0000 0003 0017 0001 0002 0004 0007 0008 0010 0012 0016"
TEST4="0005 0006 0009 0011 0013 0014 0015 0018 0019 0020"
if [ "${1:-}" != "--score-only" ]; then
    python scripts/fetch_kitti_sequence.py $TEST4 --max-frames 150 --out data/kitti
    for S in 0000 0003 0017; do
        $RUN /b/apps/eval_kitti --kitti-root /data/kitti --sequence "$S" --config configs/kitti_tiled.yaml \
            --model /src/models/yolov8n.onnx --variants configs/eval/variants_idsw4_dev.txt --out /out/dev4_kitti_tiled
    done
    for S in $TEST; do
        $RUN /b/apps/eval_kitti --kitti-root /data/kitti --sequence "$S" --config configs/kitti_tiled.yaml \
            --model /src/models/yolov8n.onnx --variants configs/eval/variants_idsw4_dev.txt --frames 150 --out /out/dev4_kitti_tiled
    done
    for S in $TEST4; do
        $RUN /b/apps/eval_kitti --kitti-root /data/kitti --sequence "$S" --config configs/kitti_tiled.yaml \
            --model /src/models/yolov8n.onnx --variants configs/eval/variants_idsw4_test.txt --frames 150 --out /out/test4_kitti_tiled
    done
fi
DEV4_V=$(grep -v '^#' configs/eval/variants_idsw4_dev.txt | awk '{print $1}')
TEST4_V=$(grep -v '^#' configs/eval/variants_idsw4_test.txt | awk '{print $1}')
python scripts/eval/run_idsw_eval.py --round 4 --raw "$RAW/dev4_kitti_tiled" --seqs $DEV4 --variants $DEV4_V --mode select \
    --out "$R/dev4_selection.json" --events-csv "$R/dev4_events.csv"
python scripts/eval/run_idsw_eval.py --round 4 --raw "$RAW/test4_kitti_tiled" --seqs $TEST4 --variants $TEST4_V --mode report \
    --selection "$R/dev4_selection.json" --out "$R/test4_report.json" --events-csv "$R/test4_events.csv"
