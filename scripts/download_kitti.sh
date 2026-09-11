#!/usr/bin/env bash
# Download KITTI tracking-benchmark calibration + labels (small, always) and,
# optionally, one sequence's stereo images into data/kitti/.
#
# Usage: scripts/download_kitti.sh [sequence] [--with-images]
#   sequence      zero-padded training-sequence id, e.g. 0000 (default: 0000)
#   --with-images also fetch this sequence's stereo images. The official
#                  archives bundle all 21 training + 29 test sequences at
#                  ~15 GB each with no server-side per-sequence download --
#                  but they're stored uncompressed on an S3 bucket that
#                  supports HTTP range requests, so scripts/fetch_kitti_
#                  sequence.py reads the zip's central directory and pulls
#                  just this sequence's files (typically 100-400 MB for one
#                  sequence, both cameras, not ~30 GB). Needs `pip install
#                  requests`; falls back to a plain full-archive download if
#                  python3/requests aren't available.
#
# Result layout (matches KittiTrackingSource's expected root):
#   data/kitti/calib/<seq>.txt
#   data/kitti/label_02/<seq>.txt
#   data/kitti/image_02/<seq>/*.png   (left,  only with --with-images)
#   data/kitti/image_03/<seq>/*.png   (right, only with --with-images)
set -euo pipefail

SEQ="0000"
WITH_IMAGES=0
for arg in "$@"; do
    case "${arg}" in
        --with-images) WITH_IMAGES=1 ;;
        *) SEQ="${arg}" ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/data/kitti"
BASE="https://s3.eu-central-1.amazonaws.com/avg-kitti"
mkdir -p "${DEST}/calib" "${DEST}/label_02"

fetch() {
    local url="$1" out="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fSL --retry 3 -o "${out}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "${out}" "${url}"
    else
        echo "need curl or wget" >&2
        return 1
    fi
}

echo "KITTI tracking :: calibration + labels -> ${DEST}"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

fetch "${BASE}/data_tracking_calib.zip" "${tmp}/calib.zip"
unzip -o "${tmp}/calib.zip" "training/calib/*" -d "${tmp}" >/dev/null
cp "${tmp}"/training/calib/*.txt "${DEST}/calib/"

fetch "${BASE}/data_tracking_label_2.zip" "${tmp}/label.zip"
unzip -o "${tmp}/label.zip" "training/label_02/*" -d "${tmp}" >/dev/null
cp "${tmp}"/training/label_02/*.txt "${DEST}/label_02/"

echo "calib + labels for all training sequences are in place."

if [ "${WITH_IMAGES}" -eq 1 ]; then
    echo
    PY="$(command -v python3 || command -v python || true)"
    if [ -n "${PY}" ] && "${PY}" -c "import requests" >/dev/null 2>&1; then
        echo "Fetching stereo images for sequence ${SEQ} via HTTP range requests"
        echo "(one sequence's files only, not the full ~15 GB/camera archive)..."
        "${PY}" "${ROOT}/scripts/fetch_kitti_sequence.py" "${SEQ}" --out "${DEST}"
    else
        echo "python3 + 'requests' not found -- falling back to the full archives"
        echo "(~15 GB left + ~15 GB right; this will take a while). Install"
        echo "requests (pip install requests) and re-run for the fast path instead."
        mkdir -p "${DEST}/image_02" "${DEST}/image_03"

        fetch "${BASE}/data_tracking_image_2.zip" "${tmp}/image_2.zip"
        unzip -o "${tmp}/image_2.zip" "training/image_02/${SEQ}/*" -d "${tmp}" >/dev/null
        cp -r "${tmp}/training/image_02/${SEQ}" "${DEST}/image_02/"
        rm -f "${tmp}/image_2.zip"

        fetch "${BASE}/data_tracking_image_3.zip" "${tmp}/image_3.zip"
        unzip -o "${tmp}/image_3.zip" "training/image_03/${SEQ}/*" -d "${tmp}" >/dev/null
        cp -r "${tmp}/training/image_03/${SEQ}" "${DEST}/image_03/"
        rm -f "${tmp}/image_3.zip"
    fi

    echo "done: $(ls "${DEST}/image_02/${SEQ}" | wc -l) frames in ${DEST}/image_02/${SEQ}"
else
    cat <<EOF

Images were NOT downloaded (pass --with-images to fetch them). Once you have
them:

  ./build/apps/benchmark_kitti --kitti-root data/kitti --sequence ${SEQ} \\
      --detector onnx --model models/yolov8n.onnx
EOF
fi
