#!/usr/bin/env bash
# Speed benchmark of the shipped pipeline (docs/evaluation/PROTOCOL.md §3.7). Runs INSIDE a container:
#   /src  the repository (read-only, cwd)      /data  the datasets (read-only)
#   /b    the build directory with apps/benchmark_e2e      /out  where per-frame CSVs are written
#
#   MODE=local (default) copies the sequences into the container's own filesystem first, so the
#   Docker-Desktop bind-mount penalty is not part of the decode time; MODE=bind reads /data directly.
#   Files are read from the page cache after the copy, so "decode" = PNG decode + a read from RAM.
#
# Env: TAG CONFIG REPS SEQS MODE BUILD OUT EXTRA (extra benchmark_e2e flags, e.g. "--cuda")
set -euo pipefail
BUILD=${BUILD:-/b}
OUT=${OUT:-/out}
CONFIG=${CONFIG:-configs/kitti_tiled.yaml}
REPS=${REPS:-3}
SEQS=${SEQS:-"0000 0003 0017"}
TAG=${TAG:-run}
ROOT=/data/kitti

if [ "${MODE:-local}" = "local" ]; then
    ROOT=/tmp/kitti
    mkdir -p "$ROOT/calib" "$ROOT/label_02" "$ROOT/image_02" "$ROOT/image_03"
    cp /data/kitti/calib/*.txt "$ROOT/calib/"
    cp /data/kitti/label_02/*.txt "$ROOT/label_02/"
    for s in $SEQS; do
        cp -r "/data/kitti/image_02/$s" "$ROOT/image_02/"
        cp -r "/data/kitti/image_03/$s" "$ROOT/image_03/"
    done
    sync
fi

mkdir -p "$OUT"
echo "[$TAG] mode=${MODE:-local} config=$CONFIG reps=$REPS seqs=$SEQS extra='${EXTRA:-}'"
echo "[$TAG] loadavg before: $(cat /proc/loadavg)"
for rep in $(seq 1 "$REPS"); do
    for s in $SEQS; do
        # shellcheck disable=SC2086
        "$BUILD/apps/benchmark_e2e" --kitti-root "$ROOT" --sequence "$s" --config "$CONFIG" \
            --model /src/models/yolov8n.onnx --assoc fused --warmup 10 ${EXTRA:-} \
            --out "$OUT/${TAG}_rep${rep}_$s.csv" > "$OUT/${TAG}_rep${rep}_$s.txt"
    done
done
echo "[$TAG] loadavg after: $(cat /proc/loadavg)"
