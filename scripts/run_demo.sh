#!/usr/bin/env bash
# Convenience wrapper for the depth demo.
# Usage: scripts/run_demo.sh [synthetic|<middlebury-scene-dir>] [extra --flags...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build/apps/stereo_depth_demo"
SOURCE="${1:-synthetic}"
shift || true

if [ ! -x "${BIN}" ]; then
    echo "build first, e.g.:  cmake -S . -B build && cmake --build build --parallel" >&2
    exit 1
fi

exec "${BIN}" --source "${SOURCE}" --out "${ROOT}/out" --cloud "$@"
