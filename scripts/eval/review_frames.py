#!/usr/bin/env python3
"""Build the visual-review material of PROTOCOL.md §6.

Frames are selected by a rule, not by eye: under P-LEGACY (pooled over the given sequences,
one tracker variant) the K frames with the most false positives, the K with the most misses,
and the K frames of one named sequence with the most misses. For each frame this writes an
annotated image (ground-truth boxes by type, ignored/neutral labels dashed, DontCare regions
shaded, emitted tracks in blue, UNMATCHED detections numbered in red, ghost tracks counted but
not drawn) plus a contact sheet of the numbered detections, and one row per item to label in
review_items.csv. Labels (A/B/C/D) are filled in afterwards by looking at the images.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import mot_eval as M            # noqa: E402
import protocol as P            # noqa: E402
from bundle import load_bundle  # noqa: E402

TYPE_COLOR = {"Car": (0, 200, 0), "Van": (0, 200, 0), "Truck": (0, 200, 0), "Pedestrian": (255, 200, 0),
              "Cyclist": (0, 140, 255), "Person_sitting": (0, 140, 255), "Tram": (0, 220, 220),
              "Misc": (0, 220, 220)}
MAX_ITEMS_PER_FRAME = 12


def dashed_rect(img, p1, p2, color, thickness=1, dash=6):
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    for (a, b, c, d) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        length = int(np.hypot(c - a, d - b))
        for s in range(0, length, dash * 2):
            e = min(s + dash, length)
            pa = (int(a + (c - a) * s / max(length, 1)), int(b + (d - b) * s / max(length, 1)))
            pb = (int(a + (c - a) * e / max(length, 1)), int(b + (d - b) * e / max(length, 1)))
            cv2.line(img, pa, pb, color, thickness)


def per_frame_counts(bundles, variant):
    rows = []
    for b in bundles:
        _, ev = M.evaluate_legacy(b.gt, b.tracks(variant), return_events=True)
        fp = ev[ev["Type"] == "FP"].groupby("FrameId").size()
        fn = ev[ev["Type"] == "MISS"].groupby("FrameId").size()
        for f in range(b.n_frames):
            rows.append({"seq": b.seq, "frame": f, "fp": int(fp.get(f, 0)), "fn": int(fn.get(f, 0))})
    return pd.DataFrame(rows)


def select(counts: pd.DataFrame, k: int, special_seq: str, k_special: int) -> list[tuple[str, int, str]]:
    def top(df, col, n):
        return df.sort_values([col, "seq", "frame"], ascending=[False, True, True]).head(n)
    chosen, seen = [], set()
    for name, df, col, n in (("top_fp", counts, "fp", k), ("top_fn", counts, "fn", k),
                             ("top_fn_" + special_seq, counts[counts["seq"] == special_seq], "fn", k_special)):
        for _, r in top(df, col, n).iterrows():
            key = (r["seq"], int(r["frame"]))
            if key not in seen:
                seen.add(key)
                chosen.append((r["seq"], int(r["frame"]), name))
    return chosen


def draw_frame(b, variant, frame, out_path, kitti_root):
    img = cv2.imread(os.path.join(kitti_root, "image_02", b.seq, f"{frame:06d}.png"))
    tracks = b.tracks(variant)
    conf = tracks[(tracks["confirmed"] == 1) & (tracks["frame"] == frame)]
    ghosts = int((conf["tsu"] > 0).sum())
    live = conf[conf["tsu"] == 0]
    _, ev = M.evaluate_legacy(b.gt, tracks, return_events=True)
    ev_f = ev[ev.index.get_level_values(0) == frame] if isinstance(ev.index, pd.MultiIndex) else ev[ev["FrameId"] == frame]
    fp_ids = set(ev_f[ev_f["Type"] == "FP"]["HId"].dropna().astype(int))
    miss_ids = set(ev_f[ev_f["Type"] == "MISS"]["OId"].dropna().astype(int))
    lab = b.labels[b.labels["frame"] == frame]

    for _, g in lab.iterrows():
        p1, p2 = (g["x1"], g["y1"]), (g["x2"], g["y2"])
        if g["type"] == "DontCare":
            ov = img.copy()
            cv2.rectangle(ov, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (128, 128, 128), -1)
            img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
            continue
        col = TYPE_COLOR.get(g["type"], (200, 200, 200))
        evaluable = any(P.gt_categories(pd.DataFrame([g]), grp).iloc[0] == "valid" for grp in P.GROUPS)
        if evaluable:
            cv2.rectangle(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), col, 2 if int(g["track_id"]) in miss_ids else 1)
        else:
            dashed_rect(img, p1, p2, col)
        if int(g["track_id"]) in miss_ids:
            cv2.putText(img, f"MISS {g['type'][:3]}{int(g['track_id'])}", (int(p1[0]), max(int(p1[1]) - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    items = []
    for _, t in live.iterrows():
        p1, p2 = (int(t["x"]), int(t["y"])), (int(t["x"] + t["w"]), int(t["y"] + t["h"]))
        if int(t["id"]) in fp_ids and len(items) < MAX_ITEMS_PER_FRAME:
            items.append(t)
    items.sort(key=lambda t: -t["h"])
    rows = []
    crops = []
    for n, t in enumerate(items, start=1):
        p1, p2 = (int(t["x"]), int(t["y"])), (int(t["x"] + t["w"]), int(t["y"] + t["h"]))
        cv2.rectangle(img, p1, p2, (0, 0, 255), 2)
        cv2.putText(img, str(n), (p1[0] + 2, p1[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        rows.append({"seq": b.seq, "frame": frame, "item": n, "kind": "FP_detection", "cls": int(t["cls"]),
                     "x": round(t["x"], 1), "y": round(t["y"], 1), "w": round(t["w"], 1), "h": round(t["h"], 1),
                     "Z": round(t["Z"], 1), "label": "", "note": ""})
        crops.append((n, t))
    matched_live = live[~live["id"].isin(fp_ids)]
    for _, t in matched_live.iterrows():
        cv2.rectangle(img, (int(t["x"]), int(t["y"])), (int(t["x"] + t["w"]), int(t["y"] + t["h"])), (255, 120, 0), 1)
    cv2.putText(img, f"{b.seq} f{frame}  legacy FP={len(fp_ids) + ghosts - ghosts}  ghosts(not drawn)={ghosts}  "
                     f"misses={len(miss_ids)}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # contact sheet of the numbered detections, on the raw image (no overlays)
    raw = cv2.imread(os.path.join(kitti_root, "image_02", b.seq, f"{frame:06d}.png"))
    cells = []
    for n, t in crops:
        w, h = t["w"], t["h"]
        pad = 0.4
        x1, y1 = max(int(t["x"] - pad * w), 0), max(int(t["y"] - pad * h), 0)
        x2, y2 = min(int(t["x"] + (1 + pad) * w), raw.shape[1]), min(int(t["y"] + (1 + pad) * h), raw.shape[0])
        c = raw[y1:y2, x1:x2].copy()
        cv2.rectangle(c, (int(t["x"]) - x1, int(t["y"]) - y1), (int(t["x"] + w) - x1, int(t["y"] + h) - y1), (0, 0, 255), 1)
        scale = 190.0 / max(c.shape[0], 1)
        scale = min(scale, 260.0 / max(c.shape[1], 1))
        c = cv2.resize(c, (max(int(c.shape[1] * scale), 1), max(int(c.shape[0] * scale), 1)), interpolation=cv2.INTER_CUBIC)
        cell = np.zeros((215, 270, 3), np.uint8)
        cell[22:22 + c.shape[0], 0:c.shape[1]] = c
        cv2.putText(cell, f"#{n} cls{int(t['cls'])} {int(t['w'])}x{int(t['h'])}px Z={t['Z']:.0f}m", (2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(cell)
    if cells:
        per_row = 4
        while len(cells) % per_row:
            cells.append(np.zeros_like(cells[0]))
        sheet = np.vstack([np.hstack(cells[i:i + per_row]) for i in range(0, len(cells), per_row)])
        canvas = np.zeros((img.shape[0] + sheet.shape[0], max(img.shape[1], sheet.shape[1]), 3), np.uint8)
        canvas[:img.shape[0], :img.shape[1]] = img
        canvas[img.shape[0]:, :sheet.shape[1]] = sheet
    else:
        canvas = img
    cv2.imwrite(out_path, canvas)
    return rows, ghosts, sorted(miss_ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--kitti-root", default="data/kitti")
    ap.add_argument("--seqs", nargs="+", default=["0000", "0003", "0017"])
    ap.add_argument("--variant", default="mahalanobis3d_s0.25_h3")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--special-seq", default="0003")
    ap.add_argument("--k-special", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    bundles = {s: load_bundle(a.raw, a.kitti_root, s) for s in a.seqs}
    counts = per_frame_counts(list(bundles.values()), a.variant)
    chosen = select(counts, a.k, a.special_seq, a.k_special)
    frames_rows, item_rows = [], []
    for seq, frame, why in chosen:
        name = f"{seq}_{frame:06d}.png"
        rows, ghosts, misses = draw_frame(bundles[seq], a.variant, frame, os.path.join(a.out, name), a.kitti_root)
        c = counts[(counts["seq"] == seq) & (counts["frame"] == frame)].iloc[0]
        frames_rows.append({"image": name, "seq": seq, "frame": frame, "selected_as": why, "legacy_fp": int(c["fp"]),
                            "legacy_fn": int(c["fn"]), "ghost_tracks": ghosts, "reviewable_fp_detections": len(rows)})
        for r in rows:
            item_rows.append({"image": name, **r})
    pd.DataFrame(frames_rows).to_csv(os.path.join(a.out, "review_frames.csv"), index=False)
    pd.DataFrame(item_rows).to_csv(os.path.join(a.out, "review_items.csv"), index=False)
    print(f"{len(chosen)} frames, {len(item_rows)} unmatched detections to label -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
