#!/usr/bin/env python3
"""DE-1 .. DE-3: detection precision / recall / AP under PROTOCOL.md §3.4.

Pooled over the given sequences (raw counts), per class group: operating point at the shipped
score 0.25, R40 and all-point AP (Vehicle at IoU 0.5 and 0.7, Person at 0.5), recall ceiling at the
score floor, per-sequence tables, Wilson intervals, recall by object height / range / left margin.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import det_eval as D            # noqa: E402
import stats as ST              # noqa: E402
from bundle import load_bundle  # noqa: E402

SETTINGS = [("Vehicle", 0.5), ("Vehicle", 0.7), ("Person", 0.5)]
OP_SCORE = 0.25
HEIGHT_BINS = [(25, 40), (40, 80), (80, 1e9)]
RANGE_BINS = [(0, 15), (15, 30), (30, 1e9)]


def pr_points(prec, rec, n=25):
    if len(prec) == 0:
        return []
    idx = np.unique(np.linspace(0, len(prec) - 1, n).astype(int))
    return [[float(rec[i]), float(prec[i])] for i in idx]


def _label(a, b):
    return f"{a}-{'inf' if b >= 1e9 else b}"


def _recall_at(tab: pd.DataFrame, sel) -> float:
    return float((tab.loc[sel, "match_score"] >= OP_SCORE).mean()) if sel.sum() else float("nan")


def evaluate(raw: str, kitti_root: str, seqs: list[str]) -> dict:
    bundles = {s: load_bundle(raw, kitti_root, s) for s in seqs}
    out = {"sequences": seqs, "operating_score": OP_SCORE, "results": {}}
    for grp, thr in SETTINGS:
        per_seq = {s: D.score_sequence(b.gt, b.dets, grp, thr) for s, b in bundles.items()}
        pooled, n_gt = D.pooled(list(per_seq.values()))
        prec, rec, _ = D.pr_curve(pooled, n_gt)
        op = D.operating_point(pooled, n_gt, OP_SCORE)
        lo_p, hi_p = ST.wilson(op["tp"], op["tp"] + op["fp"])
        lo_r, hi_r = ST.wilson(op["tp"], n_gt)
        tab = pd.concat([r.attrs["gt_table"].assign(seq=s) for s, r in per_seq.items()],
                        ignore_index=True)
        res = {
            "n_gt": n_gt, "operating_point": op,
            "precision_ci95": [lo_p, hi_p], "recall_ci95": [lo_r, hi_r],
            "ap_r40": D.ap_r40(prec, rec), "ap_all_points": D.ap_all_points(prec, rec),
            "recall_ceiling_at_floor": float(tab["match_score"].notna().mean()) if len(tab) else float("nan"),
            "recall_by_height_px": {_label(a, b): {"n": int(((tab.height_px >= a) & (tab.height_px < b)).sum()),
                                                   "recall": _recall_at(tab, (tab.height_px >= a) & (tab.height_px < b))}
                                    for a, b in HEIGHT_BINS},
            "recall_by_range_m": {_label(a, b): {"n": int(((tab.Z >= a) & (tab.Z < b)).sum()),
                                                 "recall": _recall_at(tab, (tab.Z >= a) & (tab.Z < b))}
                                  for a, b in RANGE_BINS},
            "recall_left_margin_x_lt_128": {"n": int((tab.cx < 128).sum()), "recall": _recall_at(tab, tab.cx < 128)},
            "pr_curve_points": pr_points(prec, rec),
            "per_sequence": {},
        }
        for s, r in per_seq.items():
            n = r.attrs["n_gt"]
            p_s, r_s, _ = D.pr_curve(r, n)
            res["per_sequence"][s] = {"n_gt": n, **D.operating_point(r, n, OP_SCORE), "ap_r40": D.ap_r40(p_s, r_s)}
        out["results"][f"{grp}@IoU{thr:g}"] = res
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--kitti-root", default="data/kitti")
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = evaluate(a.raw, a.kitti_root, a.seqs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for k, r in res["results"].items():
        op = r["operating_point"]
        print(f"{k:16s} n_gt={r['n_gt']:5d}  @score>={OP_SCORE}: P={op['precision']:.3f} "
              f"[{r['precision_ci95'][0]:.3f},{r['precision_ci95'][1]:.3f}] R={op['recall']:.3f} "
              f"[{r['recall_ci95'][0]:.3f},{r['recall_ci95'][1]:.3f}] F1={op['f1']:.3f} | AP R40={r['ap_r40']:.3f} "
              f"AP-all={r['ap_all_points']:.3f} | recall ceiling {r['recall_ceiling_at_floor']:.3f} | "
              f"meanIoU(TP)={op['mean_tp_iou']:.3f}")
        print("     recall by height:", {k2: (v['n'], round(v['recall'], 3)) for k2, v in r["recall_by_height_px"].items()},
              "| by range:", {k2: (v['n'], round(v['recall'], 3)) for k2, v in r["recall_by_range_m"].items()},
              "| left margin:", (r["recall_left_margin_x_lt_128"]["n"], round(r["recall_left_margin_x_lt_128"]["recall"], 3)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
