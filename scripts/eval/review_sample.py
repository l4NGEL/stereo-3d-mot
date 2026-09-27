#!/usr/bin/env python3
"""Seeded review samples (PROTOCOL.md §10, second entry): a uniform random sample of residual
P-LEGACY false positives, and an audit sample of the automatically categorised ones. Writes
contact sheets (12 numbered crops each, automatic category NOT shown) and the items table with
the automatic category kept in the CSV only."""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import diagnose as X            # noqa: E402
import mot_eval as M            # noqa: E402
import protocol as P            # noqa: E402
from bundle import load_bundle  # noqa: E402

AUDIT_CATEGORIES = ["in_dontcare", "on_ignored_or_neutral_label",
                    "overlaps_valid_label_3d_gate_rejected", "two_wheeler_class"]
PER_SHEET = 12


def all_fp_items(bundles, variant) -> pd.DataFrame:
    rows = []
    for b in bundles:
        tracks = b.tracks(variant)
        _, ev = M.evaluate_legacy(b.gt, tracks, return_events=True)
        fp = ev[ev["Type"] == "FP"].reset_index()
        trk = tracks.set_index(["frame", "id"])
        real = b.labels[b.labels["type"] != "DontCare"]
        cat_by_group = {g: P.gt_categories(real, g) for g in P.GROUPS}
        for _, e in fp.iterrows():
            f, h = int(e["FrameId"]), int(e["HId"])
            t = trk.loc[(f, h)]
            if isinstance(t, pd.DataFrame):
                t = t.iloc[0]
            cat = X.classify_legacy_fp(t, f, b.labels, real, cat_by_group)
            rows.append({"seq": b.seq, "frame": f, "hid": h, "cls": int(t["cls"]), "x": float(t["x"]),
                         "y": float(t["y"]), "w": float(t["w"]), "h": float(t["h"]), "Z": float(t["Z"]),
                         "auto_category": cat})
    return pd.DataFrame(rows)


def cell_for(item, kitti_root, caption):
    raw = cv2.imread(os.path.join(kitti_root, "image_02", item["seq"], f"{int(item['frame']):06d}.png"))
    w, h = item["w"], item["h"]
    pad = 0.4
    x1, y1 = max(int(item["x"] - pad * w), 0), max(int(item["y"] - pad * h), 0)
    x2, y2 = min(int(item["x"] + (1 + pad) * w), raw.shape[1]), min(int(item["y"] + (1 + pad) * h), raw.shape[0])
    c = raw[y1:y2, x1:x2].copy()
    cv2.rectangle(c, (int(item["x"]) - x1, int(item["y"]) - y1), (int(item["x"] + w) - x1, int(item["y"] + h) - y1),
                  (0, 0, 255), 1)
    scale = min(190.0 / max(c.shape[0], 1), 260.0 / max(c.shape[1], 1))
    c = cv2.resize(c, (max(int(c.shape[1] * scale), 1), max(int(c.shape[0] * scale), 1)), interpolation=cv2.INTER_CUBIC)
    cell = np.zeros((215, 270, 3), np.uint8)
    cell[22:22 + c.shape[0], 0:c.shape[1]] = c
    cv2.putText(cell, caption, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def write_sheets(items: pd.DataFrame, kitti_root: str, out_dir: str, prefix: str) -> pd.DataFrame:
    items = items.reset_index(drop=True).copy()
    items["sheet"] = items.index // PER_SHEET + 1
    items["item"] = items.index % PER_SHEET + 1
    for sheet, g in items.groupby("sheet"):
        cells = [cell_for(r, kitti_root, f"#{int(r['item'])} cls{int(r['cls'])} {int(r['w'])}x{int(r['h'])}px Z={r['Z']:.0f}m")
                 for _, r in g.iterrows()]
        while len(cells) % 4:
            cells.append(np.zeros_like(cells[0]))
        img = np.vstack([np.hstack(cells[i:i + 4]) for i in range(0, len(cells), 4)])
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_sheet{int(sheet)}.png"), img)
    items["label"] = ""
    items["note"] = ""
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--kitti-root", default="data/kitti")
    ap.add_argument("--seqs", nargs="+", default=["0000", "0003", "0017"])
    ap.add_argument("--variant", default="mahalanobis3d_s0.25_h3")
    ap.add_argument("--n-residual", type=int, default=60)
    ap.add_argument("--n-audit-per-category", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bundles = [load_bundle(a.raw, a.kitti_root, s) for s in a.seqs]
    items = all_fp_items(bundles, a.variant)
    rng = np.random.default_rng(a.seed)

    resid = items[items["auto_category"] == "residual"]
    pick = resid.iloc[np.sort(rng.choice(len(resid), size=min(a.n_residual, len(resid)), replace=False))]
    res_items = write_sheets(pick, a.kitti_root, a.out, "residual")
    res_items.to_csv(os.path.join(a.out, "residual_items.csv"), index=False)

    parts = []
    for cat in AUDIT_CATEGORIES:
        pool = items[items["auto_category"] == cat]
        parts.append(pool.iloc[np.sort(rng.choice(len(pool), size=min(a.n_audit_per_category, len(pool)), replace=False))])
    audit = pd.concat(parts).sample(frac=1.0, random_state=a.seed)      # shuffled: no category runs
    aud_items = write_sheets(audit, a.kitti_root, a.out, "audit")
    aud_items.to_csv(os.path.join(a.out, "audit_items.csv"), index=False)
    print(f"universe: {items['auto_category'].value_counts().to_dict()}")
    print(f"residual sample {len(res_items)} items in {res_items['sheet'].max()} sheets; audit sample {len(aud_items)} items in {aud_items['sheet'].max()} sheets -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
