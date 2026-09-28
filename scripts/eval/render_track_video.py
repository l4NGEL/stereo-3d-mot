#!/usr/bin/env python3
"""Renders an annotated video of the REAL fixed tracker (post Phase-1 isEmittable() fix)
running on a REAL, already-downloaded KITTI sequence, from the already-dumped per-frame
track states -- no new C++ run, no cherry-picked frames, the whole sequence in order.

Draws: evaluable GT boxes (thin, by type), LIVE tracks (confirmed AND matched this frame --
what the fixed product actually reports, solid box + ID), and COASTING tracks (confirmed but
not matched this frame -- what the OLD code used to report as an observation and the fixed
code no longer does, dashed box, labelled "coasting"), so the emission fix is visible, not
just claimed. This is what `docs/evaluation/review/`'s static images can't show: an object
briefly lost and re-matched keeping its ID across real frames.

Usage:
    python scripts/eval/render_track_video.py --seq 0016 --variant A5_plain_iou \
        --raw out/eval/raw/test_kitti_tiled --kitti-root data/kitti --out out/eval/videos/0016.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import kitti_io as K            # noqa: E402
import protocol as P            # noqa: E402

TYPE_COLOR = {"Car": (0, 200, 0), "Van": (0, 200, 0), "Truck": (0, 200, 0), "Pedestrian": (255, 200, 0),
              "Cyclist": (0, 140, 255), "Person_sitting": (0, 140, 255), "Tram": (0, 220, 220),
              "Misc": (0, 220, 220)}
LIVE_COLOR = (0, 0, 255)      # red: this is what the fixed pipeline reports
COAST_COLOR = (140, 140, 140)  # grey dashed: what the OLD code reported too, fixed code does not


def dashed_rect(img, x1, y1, x2, y2, color, thickness=2, dash=8):
    for (a, b, c, d) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        length = int(np.hypot(c - a, d - b))
        for s in range(0, length, dash * 2):
            e = min(s + dash, length)
            pa = (int(a + (c - a) * s / max(length, 1)), int(b + (d - b) * s / max(length, 1)))
            pb = (int(a + (c - a) * e / max(length, 1)), int(b + (d - b) * e / max(length, 1)))
            cv2.line(img, pa, pb, color, thickness)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--variant", default="A5_plain_iou", help="tracks_<variant>.txt to render")
    ap.add_argument("--raw", default="out/eval/raw/test_kitti_tiled")
    ap.add_argument("--kitti-root", default="data/kitti")
    ap.add_argument("--fps", type=int, default=10, help="KITTI tracking is ~10 Hz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    img_dir = os.path.join(a.kitti_root, "image_02", a.seq)
    n_frames = len([n for n in os.listdir(img_dir) if n.endswith(".png")])
    tracks = K.read_tracks(os.path.join(a.raw, a.seq, f"tracks_{a.variant}.txt"))
    labels = K.read_labels(os.path.join(a.kitti_root, "label_02", f"{a.seq}.txt"))
    real = labels[labels["type"] != "DontCare"]

    first = cv2.imread(os.path.join(img_dir, "000000.png"))
    h, w = first.shape[:2]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))

    for f in range(n_frames):
        img = cv2.imread(os.path.join(img_dir, f"{f:06d}.png"))
        if img is None:
            break

        lab_f = real[real["frame"] == f]
        for _, g in lab_f.iterrows():
            col = TYPE_COLOR.get(g["type"], (200, 200, 200))
            cv2.rectangle(img, (int(g.x1), int(g.y1)), (int(g.x2), int(g.y2)), col, 1)
            cv2.putText(img, g["type"][:3], (int(g.x1), max(int(g.y1) - 3, 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)

        tr_f = tracks[(tracks["frame"] == f) & (tracks["confirmed"] == 1)]
        n_live = n_coast = 0
        for _, t in tr_f.iterrows():
            x1, y1, x2, y2 = int(t.x), int(t.y), int(t.x + t.w), int(t.y + t.h)
            if t.tsu == 0:
                n_live += 1
                cv2.rectangle(img, (x1, y1), (x2, y2), LIVE_COLOR, 2)
                cv2.putText(img, f"ID{int(t.id)}", (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                           LIVE_COLOR, 2, cv2.LINE_AA)
            else:
                n_coast += 1
                dashed_rect(img, x1, y1, x2, y2, COAST_COLOR, 1)
                cv2.putText(img, f"ID{int(t.id)} coasting({int(t.tsu)})", (x1, y2 + 14),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, COAST_COLOR, 1, cv2.LINE_AA)

        cv2.rectangle(img, (0, 0), (w, 22), (0, 0, 0), -1)
        cv2.putText(img, f"seq {a.seq}  frame {f}/{n_frames-1}  variant={a.variant}  "
                         f"RED solid = reported (fixed pipeline)  GREY dashed = coasting, NOT reported  "
                         f"live={n_live} coasting={n_coast}",
                   (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(img)

    writer.release()
    print(f"wrote {a.out} ({n_frames} frames @ {a.fps} fps = {n_frames/a.fps:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
