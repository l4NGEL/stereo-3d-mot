#!/usr/bin/env python3
"""ST-2 / ST-4: real-scene disparity accuracy on the KITTI Stereo 2015 training set (200 pairs).

Scores the dumped disparity maps of apps/dump_disparity against the official ground truth
(16-bit PNG, disparity = value / 256, 0 = no GT): D1 with missing estimates counted as errors
(primary) and over estimated pixels only, on the occluded-included GT (`disp_occ_0`) and the
non-occluded subset (`disp_noc_0`), pooled over all pixels of all images, plus GT-depth bins.
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

CONFIGS = {"sgbm_tiles4": "SGBM, 4 tiles (shipped)", "sgbm_tiles1": "SGBM, untiled",
           "bm15": "BM, block 15 (baseline)"}


def score(raw_root: str, data_root: str, name: str) -> dict:
    est_dir = os.path.join(raw_root, f"kitti2015_{name}", "disp_est")
    occ_rows, noc_rows, occ_crop_rows, ms = [], [], [], []
    ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(est_dir, "*.pfm")))
    for sid in ids:
        est = K.read_pfm(os.path.join(est_dir, sid + ".pfm"))
        cal = K.read_calib_cam_to_cam(os.path.join(data_root, "calib_cam_to_cam", sid[:6] + ".txt"))
        fxb = cal["fx"] * cal["baseline"]
        for gt_dir, rows in (("disp_occ_0", occ_rows), ("disp_noc_0", noc_rows)):
            gt = SE.read_kitti_disp_png(os.path.join(data_root, gt_dir, sid + ".png"))
            if gt.shape != est.shape:
                raise SystemExit(f"{sid}: GT {gt.shape} vs estimate {est.shape}")
            rows.append(SE.d1_counts(est, gt, fxb))
            if gt_dir == "disp_occ_0":       # same pixels, left num_disparities columns excluded
                occ_crop_rows.append(SE.d1_counts(est[:, 128:], gt[:, 128:], fxb))
    timing = os.path.join(raw_root, f"kitti2015_{name}", "timing.csv")
    if os.path.exists(timing):
        import pandas as pd
        ms = pd.read_csv(timing)["match_ms"].tolist()
    return {"n_images": len(ids),
            "occ_gt": SE.d1_from_counts(SE.pool_counts(occ_rows)),
            "noc_gt": SE.d1_from_counts(SE.pool_counts(noc_rows)),
            "occ_gt_excluding_left_128px": SE.d1_from_counts(SE.pool_counts(occ_crop_rows)),
            "per_image_d1_all_occ": [r["n_d1_bad_holes_as_error"] / max(r["n_gt"], 1) for r in occ_rows],
            "match_ms_mean_bind_mount_io_excluded": float(np.mean(ms)) if ms else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="out/eval/raw/stereo")
    ap.add_argument("--data", default="data/kitti_stereo2015")
    ap.add_argument("--out", default="docs/evaluation/results/stereo_kitti2015.json")
    a = ap.parse_args()
    res = {n: score(a.raw, a.data, n) for n in CONFIGS}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for n, label in CONFIGS.items():
        o, c = res[n]["occ_gt"], res[n]["noc_gt"]
        print(f"{label:26s} occ-GT: D1-all(holes=err) {100*o['d1_all_holes_as_error']:.2f}%  valid-only {100*o['d1_valid_only']:.2f}%  "
              f"density {100*o['density']:.1f}%  EPE {o['epe_valid_only']:.2f}px | noc-GT: D1-all(holes=err) {100*c['d1_all_holes_as_error']:.2f}%  "
              f"valid-only {100*c['d1_valid_only']:.2f}% | n={res[n]['n_images']}")
    c = res["sgbm_tiles4"]["occ_gt_excluding_left_128px"]
    print(f"shipped SGBM with the left 128 columns excluded: D1-all(holes=err) {100*c['d1_all_holes_as_error']:.2f}%  "
          f"density {100*c['density']:.1f}%  valid-only {100*c['d1_valid_only']:.2f}%")
    bd = res["sgbm_tiles4"]["occ_gt"]["by_depth"]
    print("shipped SGBM D1-all(holes=err) by GT depth:", {k: (round(100*v["d1_holes_as_error"], 1), round(100*v["density"], 1)) for k, v in bd.items()})
    t, u = res["sgbm_tiles4"]["occ_gt"]["d1_all_holes_as_error"], res["sgbm_tiles1"]["occ_gt"]["d1_all_holes_as_error"]
    print(f"tiling neutrality: |D1(4 tiles) - D1(untiled)| = {100*abs(t-u):.3f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
