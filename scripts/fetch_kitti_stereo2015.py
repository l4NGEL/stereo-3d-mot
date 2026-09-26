#!/usr/bin/env python3
"""Fetch the KITTI *Stereo 2015* training set (200 stereo pairs with
ground-truth disparity) without downloading the full ~1.7 GB scene-flow
archive -- real-scene depth ground truth for benchmark_stereo_kitti.

Only frame `_10` of each training scene is used (frame `_11` is the temporal
neighbour that the flow / scene-flow benchmarks need, not stereo): the left
and right image, `disp_occ_0` (ground-truth disparity for ALL pixels that have
a LiDAR/CAD-model measurement, occluded ones included) and `disp_noc_0` (the
non-occluded subset). Ground truth is a 16-bit PNG: disparity = value / 256,
value 0 = no ground truth at that pixel.

Same trick as fetch_kitti_sequence.py (the archive is stored uncompressed on an
S3 bucket that supports HTTP range requests).

Usage:
    python scripts/fetch_kitti_stereo2015.py
    python scripts/fetch_kitti_stereo2015.py --max-pairs 20 --out data/kitti_stereo2015

Requires: pip install requests
Layout written:
    <out>/image_2/000000_10.png  <out>/image_3/000000_10.png
    <out>/disp_occ_0/000000_10.png  <out>/disp_noc_0/000000_10.png
    <out>/calib_cam_to_cam/000000.txt
"""
from __future__ import annotations

import argparse
import io
import os
import time
import zipfile

import requests

from fetch_kitti_sequence import HttpRangeFile

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
SUBDIRS = ("image_2", "image_3", "disp_occ_0", "disp_noc_0")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/kitti_stereo2015", help="output root (default: data/kitti_stereo2015)")
    ap.add_argument("--max-pairs", type=int, default=None, help="only the first N scenes (default: all 200)")
    args = ap.parse_args()

    print("fetching calibration (data_scene_flow_calib.zip, ~1.6 MB) ...")
    r = requests.get(f"{BASE}/data_scene_flow_calib.zip", timeout=120)
    r.raise_for_status()
    calib_zip = zipfile.ZipFile(io.BytesIO(r.content))
    calib_dir = os.path.join(args.out, "calib_cam_to_cam")
    os.makedirs(calib_dir, exist_ok=True)
    n_calib = 0
    for name in calib_zip.namelist():
        if "calib_cam_to_cam/" in name and name.endswith(".txt"):
            with open(os.path.join(calib_dir, os.path.basename(name)), "wb") as f:
                f.write(calib_zip.read(name))
            n_calib += 1
    print(f"  wrote {n_calib} calib file(s) -> {calib_dir}/")

    zf = zipfile.ZipFile(HttpRangeFile(f"{BASE}/data_scene_flow.zip", requests.Session()))
    names = set(zf.namelist())
    ids = sorted(n.split("/")[-1][:6] for n in names if n.startswith("training/image_2/") and n.endswith("_10.png"))
    if args.max_pairs is not None:
        ids = ids[: args.max_pairs]
    print(f"{len(ids)} training scenes selected (frame _10 only)")

    t0 = time.time()
    got = 0
    for k, sid in enumerate(ids):
        for sub in SUBDIRS:
            member = f"training/{sub}/{sid}_10.png"
            if member not in names:
                raise SystemExit(f"missing archive member {member}")
            dest_dir = os.path.join(args.out, sub)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, f"{sid}_10.png")
            size = zf.getinfo(member).file_size
            if os.path.exists(dest) and os.path.getsize(dest) == size:
                got += size
                continue
            data = zf.read(member)
            with open(dest, "wb") as f:
                f.write(data)
            got += len(data)
        if (k + 1) % 20 == 0 or k + 1 == len(ids):
            dt = max(time.time() - t0, 1e-6)
            print(f"  {k + 1}/{len(ids)} scenes  {got / 1e6:.1f} MB  ({got / 1e6 / dt:.2f} MB/s)")
    print(f"done -> {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
