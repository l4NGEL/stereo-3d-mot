#!/usr/bin/env python3
"""VO-1 .. VO-4: score benchmark_vo runs with the independent scorer (PROTOCOL.md §3.6).

Per sequence: KITTI segment metric (100-400 m), ATE (rigid and with scale), RPE at 1/10/50 frames,
path length, tracked-frame fraction. Pooled over the held-out sequences: segments of all sequences
pooled per length, then the devkit-style mean over lengths, and the plain mean over all segments.
The ground truth is read from the dataset's own poses file (and checked against the dump).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import kitti_io as K            # noqa: E402
import vo_eval as V             # noqa: E402

LENGTHS = (100, 200, 300, 400)


def load_run(run_dir: str, odo_root: str, seq: str):
    est = K.read_kitti_poses(os.path.join(run_dir, "poses_est.txt"))
    gt_all = K.read_kitti_poses(os.path.join(odo_root, "poses", f"{seq}.txt"))
    n = len(est)
    gt = gt_all[:n]
    dumped = K.read_kitti_poses(os.path.join(run_dir, "poses_gt.txt"))
    assert np.allclose(dumped, gt, atol=1e-6), "dumped GT differs from the dataset poses file"
    frames = K.read_timing(os.path.join(run_dir, "frames.csv"))
    return gt, est, frames


def cpp_numbers(log_path: str) -> dict:
    txt = open(log_path).read()
    g = lambda pat: float(re.search(pat, txt).group(1))  # noqa: E731
    return {"ate_rmse": g(r"ATE RMSE:\s+([\d.]+)"), "rpe1_trans": g(r"RPE trans RMSE:\s+([\d.]+)"),
            "rpe1_rot_deg": g(r"RPE rot RMSE:\s+([\d.]+)")}


def pooled_segments(all_segments: list[dict]) -> dict:
    return V.summarize_segments(all_segments)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--odo-root", default="data/kitti_odometry")
    ap.add_argument("--vo-root", default="out/eval/raw/vo")
    ap.add_argument("--out", default="docs/evaluation/results/vo.json")
    a = ap.parse_args()

    out = {"lengths_m": LENGTHS, "dev": {}, "test": {}}
    runs = [("dev", "04", "dev_tiled_04"), ("dev", "04", "dev_untiled_04")] + \
           [("test", s, f"test_tiled_{s}") for s in ("00", "06", "07", "08", "10")]
    test_segments, dev_segments = [], []
    for kind, seq, name in runs:
        d = os.path.join(a.vo_root, name)
        if not os.path.exists(os.path.join(d, "poses_est.txt")):
            print(f"[missing] {name}")
            continue
        gt, est, frames = load_run(d, a.odo_root, seq)
        rep = V.sequence_report(gt, est, frames, deltas=(1, 10, 50), lengths=LENGTHS)
        segs = V.kitti_segment_errors(gt, est, LENGTHS)
        rep["n_segments_raw"] = len(segs)
        cpp = cpp_numbers(os.path.join(a.vo_root, name + ".log")) if kind == "dev" else None
        if cpp:
            rep["cpp_evaluateTrajectory"] = cpp
            rep["cpp_vs_python_ate_rmse_abs_diff"] = abs(cpp["ate_rmse"] - rep["ate_rigid"]["rmse"])
        out[kind][name] = rep
        if kind == "test":
            test_segments += [dict(s, seq=seq) for s in segs]
        s = rep["segments"]
        print(f"{name:16s} {rep['n_frames']:4d} frames, path {rep['path_length_m']:.0f} m | ATE {rep['ate_rigid']['rmse']:.2f} m "
              f"({rep['ate_rigid_pct_of_path']:.2f}% of path; with scale {rep['ate_with_scale']['rmse']:.2f}, s={rep['ate_with_scale']['scale']:.3f}) | "
              f"RPE1 {rep['rpe']['1']['trans_rmse']:.3f} m/{rep['rpe']['1']['rot_rmse_deg']:.3f} deg | KITTI t_rel {s.get('t_rel_pct_devkit', float('nan')):.2f}% "
              f"r_rel {s.get('r_deg_per_m_devkit', float('nan')):.4f} deg/m ({s['n_segments']} seg) | ok {100*rep['frames_ok_fraction']:.1f}% "
              f"scale rel.err median {100*rep.get('scale_rel_err_median', float('nan')):.1f}%"
              + (f" | C++ ATE {cpp['ate_rmse']:.2f}" if cpp else ""))
    out["test_pooled"] = pooled_segments(test_segments)
    out["test_pooled"]["per_sequence_t_rel_pct_devkit"] = {n: r["segments"].get("t_rel_pct_devkit") for n, r in out["test"].items()}
    if out["test"]:
        p = out["test_pooled"]
        print(f"POOLED held-out: {p['n_segments']} segments | t_rel devkit-style {p['t_rel_pct_devkit']:.2f}% (plain {p['t_rel_pct_pooled']:.2f}%) | "
              f"r_rel {p['r_deg_per_m_devkit']:.4f} deg/m (plain {p['r_deg_per_m_pooled']:.4f}) | per length {json.dumps(p['per_length'])}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
