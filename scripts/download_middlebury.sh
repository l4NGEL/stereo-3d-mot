#!/usr/bin/env bash
# Download one Middlebury 2014 stereo scene into data/middlebury/<Scene>/.
# Usage: scripts/download_middlebury.sh [Scene]   (default: Motorcycle)
set -euo pipefail

SCENE="${1:-Motorcycle}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/data/middlebury/${SCENE}"
BASE="https://vision.middlebury.edu/stereo/data/scenes2014"
FILES=(im0.png im1.png calib.txt disp0.pfm)

mkdir -p "${DEST}"
echo "Middlebury 2014 :: ${SCENE}  ->  ${DEST}"

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

ok=1
for f in "${FILES[@]}"; do
    fetch "${BASE}/datasets/${SCENE}-perfect/${f}" "${DEST}/${f}" || { ok=0; break; }
done

if [ "${ok}" -ne 1 ]; then
    echo "per-file download failed, trying the zip archive..."
    tmp="$(mktemp -d)"
    trap 'rm -rf "${tmp}"' EXIT
    if fetch "${BASE}/zip/${SCENE}-perfect.zip" "${tmp}/scene.zip" \
        && command -v unzip >/dev/null 2>&1; then
        unzip -o "${tmp}/scene.zip" -d "${tmp}" >/dev/null
        for f in "${FILES[@]}"; do
            [ -f "${tmp}/${SCENE}-perfect/${f}" ] && cp "${tmp}/${SCENE}-perfect/${f}" "${DEST}/${f}"
        done
    else
        cat >&2 <<EOF

Automatic download failed. Get '${SCENE}' manually from
  https://vision.middlebury.edu/stereo/data/scenes2014/
and put im0.png, im1.png, calib.txt, disp0.pfm in:
  ${DEST}
EOF
        exit 1
    fi
fi

echo "done:"
ls -la "${DEST}"
