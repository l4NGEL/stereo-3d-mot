#!/usr/bin/env python3
"""ST-3: object-level depth error on the real KITTI tracking sequences (PROTOCOL.md §3.5).

For every evaluable Vehicle / Person GT matched (IoU >= 0.5, score >= 0.25) to a detection, the
detection's promoted depth (median in the shrunken box, what the tracker consumes) is compared to
the label's front-surface depth (ray through the detection-box centre into the 3D cuboid) and to
the cuboid-centre depth. Reference is KITTI's own LiDAR-annotated 3D boxes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import stereo_eval as SE        # noqa: E402
from bundle import load_bundle  # noqa: E402

BINS = ((5, 15), (15, 30), (5, 30), (30, 50))


def run(raw: str, kitti_root: str, seqs: list[str]) -> dict:
    frames = []
    for s in seqs:
        b = load_bundle(raw, kitti_root, s)
        df = SE.object_depth(b.dets, b.labels, b.gt, b.meta)
        frames.append(df.assign(seq=s))
    allf = pd.concat(frames, ignore_index=True)
    out = {"sequences": seqs, "summary": SE.summarize_object_depth(allf, BINS), "per_sequence": {}}
    for s in seqs:
        sub = allf[allf.seq == s]
        out["per_sequence"][s] = SE.summarize_object_depth(sub, BINS) if len(sub) else {"n_matched": 0}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--kitti-root", default="data/kitti")
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = run(a.raw, a.kitti_root, a.seqs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    s = res["summary"]
    print(f"matched pairs {s['n_matched']}, missing depth {s['n_missing_depth']}, ray missed {s['n_ray_missed']}")
    for grp in ("Vehicle", "Person", "All"):
        for k, v in s[grp].items():
            if v.get("n"):
                print(f"  {grp:8s} GT depth {k:7s} n={v['n']:5d}  median signed err {v['median_signed_err_m']:+.2f} m  RMSE {v['rmse_m']:.2f} m  "
                      f"MAE {v['mae_m']:.2f} m  median |rel| {100*v['median_rel_err']:.1f}%  within 5% {100*v['within_5pct']:.0f}%  "
                      f"(vs centre: bias {v['vs_center_median_signed_err_m']:+.2f} m, RMSE {v['vs_center_rmse_m']:.2f} m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
