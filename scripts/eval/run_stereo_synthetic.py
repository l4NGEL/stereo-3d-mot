#!/usr/bin/env python3
"""ST-1: synthetic-scene depth accuracy from dumped disparity maps (PROTOCOL.md §3.5).

Reports, per seed and pooled: the project-convention numbers (ROI without the left
`num_disparities` columns, error only where both maps are valid -- cross-checked against
apps/benchmark_depth), and the same errors over the FULL image with missing estimates counted as
errors; plus errors split by the known constant depth of each surface in the scene.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import kitti_io as K            # noqa: E402
import stereo_eval as SE        # noqa: E402

SEEDS = [42, 1, 2, 3, 4]
CROP = 128


def depth_from_disp(d, fxb, doffs):
    den = d + doffs
    ok = np.isfinite(d) & (d != SE.INVALID) & (den > 0)
    out = np.zeros_like(d)
    out[ok] = fxb / den[ok]
    return out


def run(root: str, tiles: int) -> dict:
    per_seed = {}
    for seed in SEEDS:
        d = os.path.join(root, f"synth_seed{seed}_tiles{tiles}")
        meta = K.read_meta(os.path.join(d, "meta.txt"))
        fxb = meta["fx"] * meta["baseline"]
        frames = sorted(glob.glob(os.path.join(d, "est_*.pfm")))
        rows, full_rows, dep_rows = [], [], []
        by_depth: dict[float, list] = {}
        for est_path in frames:
            idx = os.path.basename(est_path)[4:-4]
            est = K.read_pfm(est_path)
            gt = K.read_pfm(os.path.join(d, f"gt_{idx}.pfm"))
            gtd = K.read_pfm(os.path.join(d, f"gtdepth_{idx}.pfm"))
            rows.append(SE.project_style(est, gt, CROP))
            full_rows.append(SE.full_image(est, gt))
            ed = depth_from_disp(est, fxb, meta["doffs"])
            m = np.zeros(est.shape, bool)
            m[:, CROP:] = True
            ok = m & (ed > 0.1) & (ed < 100) & np.isfinite(gtd) & (gtd > 0.1) & (gtd < 100)
            err = (ed - gtd)[ok]
            dep_rows.append({"rmse": float(np.sqrt((err ** 2).mean())),
                             "abs_rel": float((np.abs(err) / gtd[ok]).mean())})
            # per surface depth (cards and background have constant GT depth)
            ed_full, valid_est = est, (est != SE.INVALID) & np.isfinite(est)
            gvalid = np.isfinite(gt) & (gt > 0.05)
            for depth_val in np.unique(np.round(gtd[gvalid & (gtd > 0)], 2)):
                sel = gvalid & (np.abs(gtd - depth_val) < 0.005)
                sel[:, :CROP] = False
                if sel.sum() < 500:
                    continue
                e = np.abs(est - gt)
                by_depth.setdefault(float(depth_val), []).append({
                    "n": int(sel.sum()),
                    "rmse_valid": float(np.sqrt(((e[sel & valid_est]) ** 2).mean())) if (sel & valid_est).any() else float("nan"),
                    "bad2_holes_as_error": float((sel & (~valid_est | (e > 2))).sum() / sel.sum()),
                    "coverage": float((sel & valid_est).sum() / sel.sum())})
        mean = lambda key, rs: float(np.mean([r[key] for r in rs]))  # noqa: E731
        per_seed[seed] = {
            "frames": len(frames),
            "disparity_rmse_px": mean("rmse", rows),
            "disparity_bad_2.0": float(np.mean([r["bad"]["2.0"] for r in rows])),
            "density": mean("density", rows),
            "depth_rmse_m": mean("rmse", dep_rows), "depth_abs_rel": mean("abs_rel", dep_rows),
            "full_image_bad_2.0_holes_as_error": mean("bad_2_holes_as_error", full_rows),
            "full_image_coverage": mean("coverage", full_rows),
            "by_gt_depth_m": {str(k): {"n_per_frame": int(np.mean([r["n"] for r in v])),
                                       "rmse_valid_px": float(np.nanmean([r["rmse_valid"] for r in v])),
                                       "bad2_holes_as_error": float(np.mean([r["bad2_holes_as_error"] for r in v])),
                                       "coverage": float(np.mean([r["coverage"] for r in v]))}
                              for k, v in sorted(by_depth.items())},
        }
    agg = {k: float(np.mean([s[k] for s in per_seed.values()])) for k in
           ("disparity_rmse_px", "disparity_bad_2.0", "density", "depth_rmse_m", "depth_abs_rel",
            "full_image_bad_2.0_holes_as_error", "full_image_coverage")}
    worst = {"disparity_rmse_px_max": max(s["disparity_rmse_px"] for s in per_seed.values()),
             "disparity_bad_2.0_max": max(s["disparity_bad_2.0"] for s in per_seed.values()),
             "density_min": min(s["density"] for s in per_seed.values())}
    return {"tiles": tiles, "seeds": SEEDS, "mean_over_seeds": agg, "worst_seed": worst, "per_seed": per_seed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/eval/raw/stereo")
    ap.add_argument("--out", default="docs/evaluation/results/stereo_synthetic.json")
    a = ap.parse_args()
    res = {f"tiles_{t}": run(a.root, t) for t in (1, 4)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res.items():
        m, w = v["mean_over_seeds"], v["worst_seed"]
        print(f"{k}: mean over 5 seeds x 30 frames -> disparity RMSE {m['disparity_rmse_px']:.3f} px, bad-2.0 "
              f"{100*m['disparity_bad_2.0']:.3f}%, density {100*m['density']:.2f}%, depth RMSE {m['depth_rmse_m']:.3f} m, "
              f"abs-rel {m['depth_abs_rel']:.4f} | FULL image (holes=errors) bad-2.0 {100*m['full_image_bad_2.0_holes_as_error']:.2f}%, "
              f"coverage {100*m['full_image_coverage']:.1f}% | worst seed: RMSE {w['disparity_rmse_px_max']:.3f}, "
              f"bad2 {100*w['disparity_bad_2.0_max']:.3f}%, density {100*w['density_min']:.2f}%")
    s42 = res["tiles_1"]["per_seed"][42]
    print("by GT depth (seed 42, untiled):", json.dumps(s42["by_gt_depth_m"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
