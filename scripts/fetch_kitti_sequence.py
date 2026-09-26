#!/usr/bin/env python3
"""Fetch one KITTI tracking sequence's stereo images without downloading the
full image archives.

The official KITTI tracking download bundles all 21 training + 29 test
sequences into two ~15 GB zips (one per camera) with no per-sequence
download option -- but the files inside are stored uncompressed
(ZIP_STORED) on an S3 bucket that supports HTTP range requests. Python's
zipfile only needs to read the central directory (a few MB, at the end of
the archive) to know every entry's exact byte offset, so it can fetch just
one sequence's ~100-400 files directly -- typically 100-400 MB total for one
sequence instead of ~30 GB for both cameras' full archives.

Usage:
    python scripts/fetch_kitti_sequence.py 0000
    python scripts/fetch_kitti_sequence.py 0000 --out data/kitti

Requires: pip install requests
Calibration and labels are tiny (~1 MB / ~10 MB total, all sequences) and
have no such shortcut worth taking -- scripts/download_kitti.sh fetches
those directly.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile

import requests

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
CAMERAS = [("image_02", "data_tracking_image_2.zip"), ("image_03", "data_tracking_image_3.zip")]


class HttpRangeFile:
    """A read+seek-able file-like object backed by HTTP range requests, just
    enough of the interface for zipfile.ZipFile to treat a remote URL like a
    local file without ever reading it start-to-end."""

    def __init__(self, url: str, session: requests.Session):
        self.url = url
        self.session = session
        self.pos = 0
        r = self.session.head(url, timeout=30)
        r.raise_for_status()
        self.length = int(r.headers["Content-Length"])

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.length + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        end = (self.length - 1) if (size is None or size < 0) else min(self.pos + size, self.length) - 1
        if end < self.pos:
            return b""
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=30)
                r.raise_for_status()
                data = r.content
                self.pos += len(data)
                return data
            except requests.RequestException as e:  # pragma: no cover - network flakiness
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise last_err  # type: ignore[misc]


def fetch_camera(zf: zipfile.ZipFile, sequence: str, camera: str, out_dir: str,
                 max_frames: int | None = None) -> int:
    """Fetch one camera's frames of one sequence from an already-opened remote
    zip. `max_frames` keeps only the first N frames (by file name order); files
    already on disk with the right byte size are skipped, so an interrupted
    run resumes instead of starting over."""
    prefix = f"training/{camera}/{sequence}/"
    names = sorted(n for n in zf.namelist() if n.startswith(prefix) and not n.endswith("/"))
    if not names:
        print(f"  no files found for sequence {sequence} under {prefix}", file=sys.stderr)
        return 0
    if max_frames is not None:
        names = names[:max_frames]

    os.makedirs(out_dir, exist_ok=True)
    total = sum(zf.getinfo(n).file_size for n in names)
    got = 0
    t0 = time.time()
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
            dt = max(time.time() - t0, 1e-6)
            print(f"  {camera}/{sequence}: {i + 1}/{len(names)} frames  {got / 1e6:.1f}/{total / 1e6:.1f} MB"
                  f"  ({got / 1e6 / dt:.2f} MB/s)")
    return len(names)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", nargs="+", help="zero-padded training sequence id(s), e.g. 0000")
    ap.add_argument("--out", default="data/kitti", help="KittiTrackingSource root (default: data/kitti)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="only fetch the first N frames of each sequence (default: all)")
    args = ap.parse_args()

    for camera, archive in CAMERAS:
        zf = zipfile.ZipFile(HttpRangeFile(f"{BASE}/{archive}", requests.Session()))
        for sequence in args.sequence:
            print(f"fetching {camera}/{sequence} from {archive} (range requests, not the full archive) ...")
            n = fetch_camera(zf, sequence, camera, os.path.join(args.out, camera, sequence),
                             args.max_frames)
            print(f"  wrote {n} frames -> {args.out}/{camera}/{sequence}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
