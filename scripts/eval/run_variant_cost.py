#!/usr/bin/env python3
"""Speed cost of each tracking variant next to its accuracy (PROTOCOL.md §5): per-frame milliseconds of
the stage each variant changes, from the per-frame timings eval_kitti dumped for the held-out sequences
(tiled config). The association cost is the tracker.update time of the variant's replay (measured inside the
eval_kitti runs, when the machine was not idle -- the ordering matters, not the last digit); stage costs come
from the quiet speed benchmark (speed.json).
Also derives, from measured stage times only, what a depth-free pipeline would cost (an ESTIMATE)."""
from __future__ import annotations

import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

RAW = "out/eval/raw/test_kitti_tiled"
OUT = "docs/evaluation/results/variant_cost.json"


def main() -> int:
    tt = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RAW}/*/track_timing.csv"))], ignore_index=True)
    ft = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RAW}/*/timing.csv"))], ignore_index=True)
    assoc = {v: {"mean_ms": float(g.track_ms.mean()), "median_ms": float(g.track_ms.median()),
                 "p95_ms": float(g.track_ms.quantile(0.95)), "frames": int(len(g))} for v, g in tt.groupby("variant")}
    speed = json.load(open("docs/evaluation/results/speed.json"))
    cpu, cuda = speed["shipped_tiled_local_fs"], speed["cuda_run1"]
    left_decode = cpu["io"]["mean"] / 2.0
    est = {}
    for name, det in (("CPU detector", cpu["detect"]["mean"]), ("CUDA detector (run 1)", cuda["detect"]["mean"])):
        total = left_decode + det + assoc["A5_plain_iou"]["mean_ms"]
        est[name] = {"ms_per_frame": total, "fps": 1000.0 / total}
    out = {"association_ms_per_frame": assoc,
           "note": "association times come from the replays inside the eval_kitti runs (machine not idle then); stage times below come from the quiet speed benchmark",
           "promote_plus_appearance_ms_per_frame": {"mean": float(cpu["promote"]["mean"])},
           "depth_map_ms_per_frame": {"mean": float(cpu["depth"]["mean"])},
           "stereo_ms_per_frame_tiled": {"mean": float(cpu["stereo"]["mean"])},
           "estimate_depth_free_pipeline": {"assumptions": "left image decode = half of the measured 2x imread; detector and tracker times as measured; no stereo, no depth map, no 3D promotion; NOT measured end to end",
                                            **est}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    for v, d in sorted(assoc.items()):
        print(f"{v:24s} tracker.update mean {d['mean_ms']:.3f} ms  p95 {d['p95_ms']:.3f}")
    print("promote + appearance ms/frame:", out["promote_plus_appearance_ms_per_frame"], "| estimate depth-free:", est)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
