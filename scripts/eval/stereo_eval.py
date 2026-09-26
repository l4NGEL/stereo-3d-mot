"""Depth scoring (PROTOCOL.md §3.5), computed from dumped disparity maps so that pixels
with no estimate stay visible.

* project_style(): replicates what apps/benchmark_depth reports (ROI with the left
  `num_disparities` columns cropped, error only where both maps are valid) -- the cross-check.
* full_image(): the same errors over every GT pixel with missing estimates counted as errors.
* d1(): KITTI Stereo 2015 D1 (|d - d_gt| > 3 px and > 5 % of d_gt).
* object_depth(): detection depth vs the label's 3D cuboid on the tracking sequences.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import det_eval as D
import protocol as P
from geometry import cuboid_front_depth, iou_matrix

INVALID = -1.0


def _est_valid(est: np.ndarray) -> np.ndarray:
    return np.isfinite(est) & (est != INVALID)


def project_style(est: np.ndarray, gt: np.ndarray, crop: int, valid_min: float = 0.05,
                  valid_max: float = 1e6, bad=(1.0, 2.0, 4.0)) -> dict:
    e, g = est[:, crop:], gt[:, crop:]
    gt_ok = np.isfinite(g) & (g >= valid_min) & (g <= valid_max)
    ok = gt_ok & _est_valid(e) & (e >= valid_min) & (e <= valid_max)
    err = np.abs(e - g)[ok]
    return {"rmse": float(np.sqrt((err ** 2).mean())) if err.size else float("nan"),
            "mae": float(err.mean()) if err.size else float("nan"),
            "bad": {str(t): float((err > t).mean()) if err.size else float("nan") for t in bad},
            "density": float(ok.sum() / max(gt_ok.sum(), 1)),
            "n_eval": int(ok.sum()), "n_gt": int(gt_ok.sum())}


def full_image(est: np.ndarray, gt: np.ndarray, bad=(1.0, 2.0, 4.0), valid_min: float = 0.05) -> dict:
    gt_ok = np.isfinite(gt) & (gt >= valid_min)
    ev = _est_valid(est)
    err = np.abs(est - gt)
    res = {"coverage": float((gt_ok & ev).sum() / max(gt_ok.sum(), 1)), "n_gt": int(gt_ok.sum())}
    for t in bad:
        bad_px = gt_ok & (~ev | (err > t))
        res[f"bad_{t:g}_holes_as_error"] = float(bad_px.sum() / max(gt_ok.sum(), 1))
        res[f"bad_{t:g}_valid_only"] = float((gt_ok & ev & (err > t)).sum() / max((gt_ok & ev).sum(), 1))
    return res


def d1(est: np.ndarray, gt: np.ndarray, fx_baseline: float | None = None) -> dict:
    """KITTI stereo 2015. `gt` in pixels, <= 0 where there is no ground truth."""
    gt_ok = np.isfinite(gt) & (gt > 0)
    ev = _est_valid(est)
    err = np.abs(est - gt)
    bad_px = (err > 3.0) & (err / np.where(gt > 0, gt, 1.0) > 0.05)
    n = max(int(gt_ok.sum()), 1)
    both = gt_ok & ev
    res = {
        "n_gt": int(gt_ok.sum()),
        "density": float(both.sum() / n),
        "d1_all_holes_as_error": float((gt_ok & (~ev | bad_px)).sum() / n),
        "d1_valid_only": float((both & bad_px).sum() / max(int(both.sum()), 1)),
        "epe_valid_only": float(err[both].mean()) if both.any() else float("nan"),
    }
    for t in (1.0, 2.0, 3.0, 5.0):
        res[f"bad_{t:g}px_holes_as_error"] = float((gt_ok & (~ev | (err > t))).sum() / n)
    if fx_baseline is not None:
        depth = np.where(gt_ok, fx_baseline / np.where(gt > 0, gt, 1.0), np.inf)
        for lo, hi in ((0, 10), (10, 20), (20, 40), (40, np.inf)):
            m = gt_ok & (depth >= lo) & (depth < hi)
            key = f"{lo}-{hi if np.isfinite(hi) else 'inf'}m"
            res.setdefault("by_depth", {})[key] = {
                "n_gt": int(m.sum()),
                "d1_holes_as_error": float((m & (~ev | bad_px)).sum() / max(int(m.sum()), 1)),
                "density": float((m & ev).sum() / max(int(m.sum()), 1)),
            }
    return res


def read_kitti_disp_png(path: str) -> np.ndarray:
    import cv2
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return raw.astype(np.float32) / 256.0


def object_depth(dets: pd.DataFrame, labels: pd.DataFrame, gt_seq, meta: dict,
                 score_thr: float = 0.25) -> pd.DataFrame:
    """One row per (detection matched to an evaluable GT at IoU >= 0.5): the detection's promoted
    depth vs the GT front-surface depth (ray through the detection-box centre into the label's
    cuboid) and vs the cuboid-centre depth."""
    rows = []
    lab = labels[labels["type"] != "DontCare"]
    for grp in P.GROUPS:
        gd = D.group_dets(dets, grp)
        gd = gd[gd["score"] >= score_thr]
        cat = P.gt_categories(lab, grp)
        valid = lab[cat == "valid"]
        for f, g in gd.groupby("frame"):
            gt = valid[valid["frame"] == f]
            if gt.empty:
                continue
            boxes = g[["x", "y", "x2", "y2"]].to_numpy()
            iou = iou_matrix(boxes, gt[["x1", "y1", "x2", "y2"]].to_numpy())
            used = set()
            for i in np.argsort(-g["score"].to_numpy()):
                j = int(np.argmax(np.where(np.isin(np.arange(iou.shape[1]), list(used)), -1, iou[i])))
                if iou[i, j] < 0.5 or j in used:
                    continue
                used.add(j)
                d, o = g.iloc[i], gt.iloc[j]
                u, v = d["x"] + d["w"] / 2.0, d["y"] + d["h"] / 2.0
                front = cuboid_front_depth(u, v, meta["fx"], meta["fy"], meta["cx"], meta["cy"],
                                           o["h"], o["w"], o["l"], o["X"], o["Y"], o["Z"], o["ry"])
                rows.append({"seq_frame": int(f), "group": grp, "type": o["type"], "iou": float(iou[i, j]),
                             "valid_depth": bool(d["valid"] == 1), "est_depth": float(d["depth"]) if d["valid"] == 1 else np.nan,
                             "gt_front": front if front is not None else np.nan, "gt_center": float(o["Z"]),
                             "gt_track": int(o["track_id"])})
    return pd.DataFrame(rows)


def summarize_object_depth(df: pd.DataFrame, bins=((5, 15), (15, 30), (30, 50))) -> dict:
    out = {"n_matched": int(len(df)), "n_missing_depth": int((~df["valid_depth"]).sum()) if len(df) else 0,
           "n_ray_missed": int(df["gt_front"].isna().sum()) if len(df) else 0}
    ok = df[df["valid_depth"] & df["gt_front"].notna()]
    for grp in list(P.GROUPS) + ["All"]:
        sub = ok if grp == "All" else ok[ok["group"] == grp]
        res = {}
        for lo, hi in bins:
            b = sub[(sub["gt_front"] >= lo) & (sub["gt_front"] < hi)]
            if b.empty:
                res[f"{lo}-{hi}m"] = {"n": 0}
                continue
            e = b["est_depth"] - b["gt_front"]
            ec = b["est_depth"] - b["gt_center"]
            res[f"{lo}-{hi}m"] = {
                "n": int(len(b)), "median_signed_err_m": float(e.median()), "mae_m": float(e.abs().mean()),
                "rmse_m": float(np.sqrt((e ** 2).mean())),
                "median_rel_err": float((e.abs() / b["gt_front"]).median()),
                "within_5pct": float(((e.abs() / b["gt_front"]) <= 0.05).mean()),
                "vs_center_median_signed_err_m": float(ec.median()), "vs_center_rmse_m": float(np.sqrt((ec ** 2).mean())),
            }
        out[grp] = res
    return out
