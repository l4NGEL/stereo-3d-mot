#!/usr/bin/env python3
"""Fetch one KITTI *odometry* benchmark sequence -- ground-truth poses and
one stereo pair's grayscale images -- for Phase 7 (visual odometry).

This is a different benchmark from the tracking one this project already
uses (see fetch_kitti_sequence.py): different sequence numbering (00-10 have
ground truth; the tracking benchmark's 0000-0020 are a different split of
the same underlying raw data), and crucially it's the only one of the two
that ships ego-motion ground truth (poses) at all -- the tracking benchmark
has object labels, not vehicle poses.

Same trick as fetch_kitti_sequence.py: the official download bundles all 22
sequences' images into one ~23 GB archive with no per-sequence option, but
it's stored uncompressed (ZIP_STORED) on an S3 bucket that supports HTTP
range requests, so this reads the zip's central directory (a few MB) and
pulls just one sequence's files directly.

Usage:
    python scripts/fetch_kitti_odometry.py 04
    python scripts/fetch_kitti_odometry.py 04 --out data/kitti_odometry

Requires: pip install requests
"""
from __future__ import annotations

import argparse
import os
import sys

from fetch_kitti_sequence import HttpRangeFile

import requests
import zipfile

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
CAMERAS = [("image_0", "left"), ("image_1", "right")]


def fetch_small(url: str, out_path: str) -> None:
    if os.path.exists(out_path):
        print(f"  {out_path} already present, skipping")
        return
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(r.content)
    print(f"  wrote {out_path} ({len(r.content) / 1e6:.2f} MB)")


def fetch_small_zip_members(url: str, out_dir: str, prefix: str | None = None) -> None:
    """Download a small zip fully, then extract every member (optionally
    filtered by prefix) into out_dir, flattened to its basename under the
    member's own subpath structure preserved relative to `prefix`."""
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    import io

    zf = zipfile.ZipFile(io.BytesIO(r.content))
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        if prefix is not None and not name.startswith(prefix):
            continue
        rel = name[len(prefix):] if prefix else os.path.basename(name)
        dest = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(zf.read(name))
        n += 1
    print(f"  extracted {n} file(s) -> {out_dir}/")


def fetch_camera(zf: zipfile.ZipFile, sequence: str, camera: str, out_dir: str,
                 max_frames: int | None = None) -> int:
    """Fetch one camera's frames of one sequence from an already-opened remote
    zip. `max_frames` keeps only the first N frames; files already on disk with
    the right byte size are skipped (an interrupted run resumes)."""
    prefix = f"dataset/sequences/{sequence}/{camera}/"
    names = sorted(n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".png"))
    if not names:
        print(f"  no files found for sequence {sequence} camera {camera}", file=sys.stderr)
        return 0
    if max_frames is not None:
        names = names[:max_frames]

    os.makedirs(out_dir, exist_ok=True)
    total = sum(zf.getinfo(n).file_size for n in names)
    got = 0
    for i, name in enumerate(names):
        dest = os.path.join(out_dir, os.path.basename(name))
        size = zf.getinfo(name).file_size
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            got += size
        else:
            data = zf.read(name)
            with open(dest, "wb") as out:
                out.write(data)
            got += len(data)
        if (i + 1) % 25 == 0 or i + 1 == len(names):
            print(f"  {camera}/{sequence}: {i + 1}/{len(names)} frames  {got / 1e6:.1f}/{total / 1e6:.1f} MB")
    return len(names)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", nargs="+",
                    help="zero-padded odometry sequence id(s) with ground truth, e.g. 04 (00-10)")
    ap.add_argument("--out", default="data/kitti_odometry", help="output root (default: data/kitti_odometry)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="only fetch the first N frames of each sequence (default: all)")
    args = ap.parse_args()

    print("fetching calibration (all sequences, ~0.6 MB) ...")
    fetch_small_zip_members(f"{BASE}/data_odometry_calib.zip", os.path.join(args.out, "sequences"),
                            prefix="dataset/sequences/")

    print("fetching ground-truth poses (all sequences, ~1.3 MB) ...")
    fetch_small_zip_members(f"{BASE}/data_odometry_poses.zip", os.path.join(args.out, "poses"),
                            prefix="dataset/poses/")

    zf = zipfile.ZipFile(HttpRangeFile(f"{BASE}/data_odometry_gray.zip", requests.Session()))
    for sequence in args.sequence:
        for camera, label in CAMERAS:
            print(f"fetching {label} ({camera})/{sequence} from data_odometry_gray.zip (range requests) ...")
            n = fetch_camera(zf, sequence, camera,
                             os.path.join(args.out, "sequences", sequence, camera), args.max_frames)
            print(f"  wrote {n} frames -> {args.out}/sequences/{sequence}/{camera}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
