#!/usr/bin/env python3
"""MOT17 outside-data check (docs/evaluation/IDSW_PLAN.md round 5).

Scores eval_mot2d dumps (E = 0: confirmed and matched this frame) and SORT on the same public
FRCNN detections. MOTChallenge-style preprocessing, as in TrackEval's MOT17 benchmark:
  * scored GT = class 1 (pedestrian) with the consider flag set;
  * tracker boxes matched (Hungarian, IoU >= 0.5) to a distractor GT -- classes 2, 7, 8, 12,
    or a pedestrian with consider flag 0 -- are removed before counting.
Counting: mot_eval.counts_motmetrics (py-motmetrics), IoU >= 0.5, pooled from raw counts.
Not the official server.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__))
import kitti_io as K           # noqa: E402
import mot_eval as M           # noqa: E402
import sort_baseline as S      # noqa: E402
import stats as ST             # noqa: E402
from geometry import iou_matrix  # noqa: E402

DISTRACTOR_CLASSES = {2, 7, 8, 12}
KEEP = ("mota", "idf1", "recall", "precision", "num_switches", "num_fragmentations",
        "num_false_positives", "num_misses", "idtp", "idfp", "idfn", "num_objects")


def seq_len(root: str, seq: str) -> int:
    for line in open(os.path.join(root, seq, "seqinfo.ini")):
        if line.startswith("seqLength="):
            return int(line.split("=")[1])
    raise RuntimeError(seq)


def load_gt(root: str, seq: str) -> pd.DataFrame:
    gt = pd.read_csv(os.path.join(root, seq, "gt", "gt.txt"), header=None,
                     names=["frame", "id", "x", "y", "w", "h", "consider", "cls", "vis"])
    gt["frame"] -= 1
    return gt


def load_dets(root: str, seq: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(root, seq, "det", "det.txt"), header=None).iloc[:, :7]
    d.columns = ["frame", "id", "x", "y", "w", "h", "score"]
    d["frame"] -= 1
    d["cls"] = 0
    return d[(d.w > 0) & (d.h > 0)]


def xyxy(df: pd.DataFrame) -> np.ndarray:
    return np.stack([df.x, df.y, df.x + df.w, df.y + df.h], axis=1) if len(df) else np.zeros((0, 4))


def score(gt: pd.DataFrame, hyp: pd.DataFrame, n_frames: int) -> dict:
    """hyp: emitted rows (frame, id, x, y, w, h)."""
    gtf = {f: g for f, g in gt.groupby("frame")}
    hf = {f: h for f, h in hyp.groupby("frame")}
    inputs = []
    for f in range(n_frames):
        g = gtf.get(f, gt.iloc[0:0])
        h = hf.get(f, hyp.iloc[0:0])
        hb = xyxy(h)
        hid = h["id"].to_numpy() if len(h) else np.zeros(0, int)
        if len(g) and len(h):
            iou = iou_matrix(xyxy(g), hb)
            r, c = linear_sum_assignment(-iou)
            distractor = ((g["cls"].isin(DISTRACTOR_CLASSES)) | ((g["cls"] == 1) & (g["consider"] == 0))).to_numpy()
            drop = np.zeros(len(h), bool)
            for i, j in zip(r, c):
                if iou[i, j] >= 0.5 and distractor[i]:
                    drop[j] = True
            hb, hid = hb[~drop], hid[~drop]
        valid = g[(g["cls"] == 1) & (g["consider"] == 1)]
        vb = xyxy(valid)
        iou = iou_matrix(vb, hb) if len(vb) and len(hb) else np.zeros((len(vb), len(hb)))
        dist = np.where(iou >= 0.5, 1.0 - iou, np.nan)
        inputs.append((valid["id"].to_numpy(), hid, dist))
    return M.counts_motmetrics(inputs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot-root", default="data/mot17/train")
    ap.add_argument("--raw", required=True, help="eval_mot2d output root (<raw>/<seq>/tracks_*.txt)")
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--mode", choices=["select", "report"], required=True)
    ap.add_argument("--order", nargs="+", help="select: candidate order for the final tie-break")
    ap.add_argument("--compare", nargs="+", default=[], help="report: a:b pairs for paired bootstrap")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    per_seq = {v: {} for v in a.variants + ["SORT"]}
    for seq in a.seqs:
        n = seq_len(a.mot_root, seq)
        gt = load_gt(a.mot_root, seq)
        for v in a.variants:
            tr = K.read_tracks(os.path.join(a.raw, seq, f"tracks_{v}.txt"))
            hyp = tr[(tr.confirmed == 1) & (tr.tsu == 0)]
            per_seq[v][seq] = score(gt, hyp, n)
        sort = S.sort_track(load_dets(a.mot_root, seq), n)
        per_seq["SORT"][seq] = score(gt, sort, n)

    out = {"sequences": a.seqs, "systems": {}}
    for name, d in per_seq.items():
        out["systems"][name] = {"pooled": {k: float(v) for k, v in M.pool(list(d.values())).items() if k in KEEP},
                                "per_sequence": {s: {k: float(v) for k, v in M.pool([c]).items() if k in KEEP}
                                                 for s, c in d.items()}}
    if a.mode == "select":
        order = a.order or a.variants
        best = max(out["systems"][v]["pooled"]["idf1"] for v in order)
        tied = [v for v in order if out["systems"][v]["pooled"]["idf1"] >= best - 0.005]
        tied.sort(key=lambda v: (out["systems"][v]["pooled"]["num_switches"],
                                 -out["systems"][v]["pooled"]["mota"], order.index(v)))
        out["selection"] = {"rule": "IDSW_PLAN.md round 5: max pooled dev IDF1, ties within 0.005 -> fewer IDSW "
                                    "-> higher MOTA -> listed order", "selected": tied[0], "tied": tied}
        print("selected:", tied[0], "tied:", tied)
    else:
        out["comparisons"] = []
        for pair in a.compare:
            x, y = pair.split(":")
            for m in ("idf1", "mota", "idsw_rate", "recall"):
                out["comparisons"].append({"a": x, "b": y, **ST.paired_bootstrap(per_seq[x], per_seq[y], m,
                                                                                 n_boot=a.n_boot)})
    json.dump(out, open(a.out, "w"), indent=1)
    for name in per_seq:
        p = out["systems"][name]["pooled"]
        print(f"{name:12s} IDF1 {p['idf1']:.3f} MOTA {p['mota']:+.3f} IDSW {int(p['num_switches']):5d} "
              f"Frag {int(p['num_fragmentations']):5d} rec {p['recall']:.3f} prec {p['precision']:.3f} "
              f"GT {int(p['num_objects'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
