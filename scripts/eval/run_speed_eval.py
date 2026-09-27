#!/usr/bin/env python3
"""SP-1 .. SP-3 and the optimisation-cost rows: latency distributions from the per-frame CSVs that
scripts/eval/run_speed.sh (benchmark_e2e) wrote. Warm-up frames (warm == 1) are excluded from every
statistic; the very first frame's detect time is reported separately (one-time CUDA initialisation).
FPS is throughput = 1000 / mean latency of the strictly sequential loop."""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

STAGES = ("io_ms", "stereo_ms", "depth_ms", "detect_ms", "promote_ms", "track_ms", "compute_ms", "total_ms")


def stats_of(d: pd.DataFrame) -> dict:
    out = {"frames": int(len(d))}
    for c in STAGES:
        out[c.replace("_ms", "")] = {"mean": float(d[c].mean()), "median": float(d[c].median()),
                                     "p95": float(d[c].quantile(0.95)), "p99": float(d[c].quantile(0.99)),
                                     "max": float(d[c].max())}
    out["fps_with_io_mean"] = 1000.0 / float(d["total_ms"].mean())
    out["fps_with_io_median"] = 1000.0 / float(d["total_ms"].median())
    out["fps_compute_mean"] = 1000.0 / float(d["compute_ms"].mean())
    out["median_total_ms"] = float(d["total_ms"].median())
    out["p95_total_ms"] = float(d["total_ms"].quantile(0.95))
    out["p99_total_ms"] = float(d["total_ms"].quantile(0.99))
    out["detect_over_40ms_fraction"] = float((d["detect_ms"] > 40).mean())
    return out


def load(root: str, tag: str):
    files = sorted(glob.glob(os.path.join(root, f"{tag}_rep*_*.csv")))
    frames = []
    for f in files:
        base = os.path.basename(f)[len(tag) + 4:-4]        # "<rep>_<seq>"
        rep, seq = base.split("_")
        d = pd.read_csv(f)
        first_detect = float(d["detect_ms"].iloc[0])
        w = d[d["warm"] == 0].copy()
        w["rep"], w["seq"], w["first_detect_ms"] = int(rep), seq, first_detect
        frames.append(w)
    return pd.concat(frames, ignore_index=True) if frames else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/eval/raw/speed")
    ap.add_argument("--out", default="docs/evaluation/results/speed.json")
    a = ap.parse_args()
    res = {}
    for name, tag in (("shipped_tiled_local_fs", "tiled_local"), ("untiled_local_fs", "untiled_local"),
                      ("tiled_bind_mount_seq0000", "tiled_bind"), ("cpu_in_gpu_image", "cpu_gpuimage"),
                      ("cuda_run1", "cuda_local"), ("cuda_run2_two_reps", "cuda_clean")):
        d = load(a.root, tag)
        if d is None:
            continue
        s = stats_of(d)
        s["per_repetition"] = [{"rep": int(r), **{k: v for k, v in stats_of(g).items() if k in ("fps_with_io_mean", "p95_total_ms", "median_total_ms", "fps_compute_mean")}}
                               for r, g in d.groupby("rep")]
        s["first_frame_detect_ms_per_run"] = sorted({round(x, 1) for x in d["first_detect_ms"]})
        res[name] = s
        print(f"{name:26s} {s['frames']:5d} frames | total mean {s['total']['mean']:6.1f} median {s['total']['median']:6.1f} p95 {s['total']['p95']:6.1f} p99 {s['total']['p99']:6.1f} ms "
              f"-> {s['fps_with_io_mean']:.2f} FPS | compute-only {s['fps_compute_mean']:.2f} FPS | stereo {s['stereo']['mean']:.1f} detect {s['detect']['mean']:.1f} (median {s['detect']['median']:.1f}) "
              f"decode {s['io']['mean']:.1f} ms | per-rep FPS {[round(r['fps_with_io_mean'], 2) for r in s['per_repetition']]}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
