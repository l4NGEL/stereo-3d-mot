"""Tracking scorers (docs/evaluation/PROTOCOL.md §3.2, §3.3, §3.8).

* P-K2D   -- primary, KITTI-like: per class group, 2D IoU >= 0.5, moderate-difficulty
             ground truth, ignore rules, emission policy E.
* P-LEGACY -- what apps/benchmark_kitti.cpp reports: every non-DontCare label vs
             every confirmed track, class-agnostic, 3D distance <= 2 m.

Counting is done by py-motmetrics; mot_reference.py re-implements the same
CLEAR-MOT/IDF1 counting independently and the two must agree exactly.
Aggregation across sequences/classes is always from raw counts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import motmetrics as mm
from scipy.spatial.distance import cdist

import protocol as P
from geometry import iou_matrix, project_point

GT_DIST_SENTINEL = np.nan

MM_METRICS = ["num_frames", "num_objects", "num_matches", "num_switches",
              "num_false_positives", "num_misses", "num_detections", "num_fragmentations",
              "mota", "motp", "idf1", "idtp", "idfp", "idfn", "precision", "recall"]

COUNT_KEYS = ["num_objects", "num_misses", "num_false_positives", "num_switches",
              "num_detections", "num_fragmentations", "idtp", "idfp", "idfn", "dist_sum"]


# ----------------------------------------------------------------------------- inputs
class SequenceGT:
    """Per-frame ground truth of one sequence, split by class group and category."""

    def __init__(self, labels: pd.DataFrame, n_frames: int, rules: bool = True):
        """`rules=False` switches every ignore rule off (no moderate filter, no neutral
        classes, no DontCare, no hypothesis pre-filter) -- used only to measure how much
        the rules themselves change the counts (hypothesis H-scope)."""
        self.n_frames = n_frames
        self.rules = rules
        lab = labels[labels["frame"] < n_frames]
        dc = lab[lab["type"] == "DontCare"]
        real = lab[lab["type"] != "DontCare"].copy()
        self.real = real
        self.dontcare = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in dc.groupby("frame")}             if rules else {}
        self.per_group: dict[str, dict[str, dict]] = {}
        for grp in P.GROUPS:
            cat = P.gt_categories(real, grp)
            if not rules:
                cat = cat.where(cat != "ignored", "valid").where(cat != "neutral", "none")
            frames = {}
            for f, idx in real.groupby("frame").groups.items():
                rows = real.loc[idx]
                c = cat.loc[idx]
                valid = rows[c == "valid"]
                ign = rows[(c == "ignored") | (c == "neutral")]
                frames[f] = {
                    "valid_ids": valid["track_id"].to_numpy(),
                    "valid_boxes": valid[["x1", "y1", "x2", "y2"]].to_numpy(),
                    "ign_boxes": ign[["x1", "y1", "x2", "y2"]].to_numpy(),
                }
            self.per_group[grp] = frames

    def frame(self, grp: str, f: int) -> dict:
        return self.per_group[grp].get(f) or {"valid_ids": np.zeros(0, int),
                                              "valid_boxes": np.zeros((0, 4)),
                                              "ign_boxes": np.zeros((0, 4))}


def emission_hypotheses(tracks: pd.DataFrame, meta: dict, emit_max_coast: int) -> pd.DataFrame:
    """Confirmed tracks with time_since_update <= E, with an image-plane box per row.

    tsu == 0: the matched detection's box, as dumped. tsu > 0: the Kalman-predicted 3D
    position projected into the image, size = last matched size scaled by z_last/z_now
    (PROTOCOL.md §3.3 step 2). Adds columns x1,y1,x2,y2 and `group`."""
    if tracks.empty:
        return tracks.assign(x1=[], y1=[], x2=[], y2=[], group=[])
    t = tracks[(tracks["confirmed"] == 1) & (tracks["tsu"] <= emit_max_coast)].copy()
    if t.empty:
        return t.assign(x1=[], y1=[], x2=[], y2=[], group=[])
    t = t.sort_values(["id", "frame"])
    full = tracks.sort_values(["id", "frame"])
    z_upd = full["Z"].where(full["tsu"] == 0)
    z_last = z_upd.groupby(full["id"]).ffill()
    t["z_last"] = z_last.loc[t.index]
    coast = t["tsu"] > 0
    x1, y1 = t["x"].to_numpy().copy(), t["y"].to_numpy().copy()
    x2, y2 = (t["x"] + t["w"]).to_numpy().copy(), (t["y"] + t["h"]).to_numpy().copy()
    if coast.any():
        c = t[coast]
        u, v = project_point(c["X"].to_numpy(), c["Y"].to_numpy(), c["Z"].to_numpy(),
                             meta["fx"], meta["fy"], meta["cx"], meta["cy"])
        scale = np.clip((c["z_last"] / c["Z"]).fillna(1.0).to_numpy(), 0.25, 4.0)
        w, h = c["w"].to_numpy() * scale, c["h"].to_numpy() * scale
        m = coast.to_numpy()
        x1[m], y1[m], x2[m], y2[m] = u - w / 2, v - h / 2, u + w / 2, v + h / 2
    t["x1"], t["y1"], t["x2"], t["y2"] = x1, y1, x2, y2
    t["group"] = t["cls"].map(P.COCO_GROUP)
    return t


def k2d_inputs(seq: SequenceGT, hyps: pd.DataFrame, grp: str) -> list[tuple]:
    """Per-frame (gt_ids, hyp_ids, dist) for one class group under P-K2D. dist = 1 - IoU,
    NaN where IoU < 0.5. Hypotheses that cannot be TPs and are ignorable are dropped."""
    h = hyps[hyps["group"] == grp] if len(hyps) else hyps
    by_frame = {f: g for f, g in h.groupby("frame")} if len(h) else {}
    out = []
    for f in range(seq.n_frames):
        g = seq.frame(grp, f)
        hf = by_frame.get(f)
        if hf is None or hf.empty:
            hb, hid = np.zeros((0, 4)), np.zeros(0, int)
        else:
            hb = hf[["x1", "y1", "x2", "y2"]].to_numpy()
            hid = hf["id"].to_numpy()
            if seq.rules:
                drop = P.prefilter_mask(hb, g["valid_boxes"], g["ign_boxes"],
                                        seq.dontcare.get(f, np.zeros((0, 4))))
                hb, hid = hb[~drop], hid[~drop]
        iou = iou_matrix(g["valid_boxes"], hb)
        dist = np.where(iou >= P.MATCH_IOU, 1.0 - iou, np.nan)
        out.append((g["valid_ids"], hid, dist))
    return out


def legacy_inputs(seq: SequenceGT, tracks: pd.DataFrame, gate: float = 2.0,
                  emit_max_coast: int | None = None) -> list[tuple]:
    """Per-frame (gt_ids, hyp_ids, dist) for the strict legacy protocol. `emit_max_coast`
    None = every confirmed track (what benchmark_kitti reports); an integer restricts
    hypotheses to time_since_update <= E (used only to test hypothesis H-ghost)."""
    real = seq.real
    gt_by_frame = {f: g for f, g in real.groupby("frame")}
    hyp = tracks[tracks["confirmed"] == 1]
    if emit_max_coast is not None:
        hyp = hyp[hyp["tsu"] <= emit_max_coast]
    hyp_by_frame = {f: g for f, g in hyp.groupby("frame")} if len(hyp) else {}
    out = []
    for f in range(seq.n_frames):
        g = gt_by_frame.get(f)
        if g is None or g.empty:
            gt_ids, gt_pos = np.zeros(0, int), np.zeros((0, 3))
        else:
            gt_ids = g["track_id"].to_numpy()
            gt_pos = np.stack([g["X"], g["Y"] - g["h"] / 2.0, g["Z"]], axis=1)
        hf = hyp_by_frame.get(f)
        if hf is None or hf.empty:
            hid, hpos = np.zeros(0, int), np.zeros((0, 3))
        else:
            hid = hf["id"].to_numpy()
            hpos = hf[["X", "Y", "Z"]].to_numpy()
        if len(gt_ids) and len(hid):
            d = cdist(gt_pos, hpos)
            dist = np.where(d <= gate, d, np.nan)
        else:
            dist = np.full((len(gt_ids), len(hid)), np.nan)
        out.append((gt_ids, hid, dist))
    return out


# ---------------------------------------------------------------------------- scoring
def counts_motmetrics(inputs: list[tuple], return_events: bool = False):
    """Run py-motmetrics over per-frame inputs. Returns a dict of raw counts (+ events)."""
    acc = mm.MOTAccumulator(auto_id=False)
    for f, (gt_ids, hyp_ids, dist) in enumerate(inputs):
        acc.update(list(map(int, gt_ids)), list(map(int, hyp_ids)),
                   np.asarray(dist, dtype=float).reshape(len(gt_ids), len(hyp_ids)), frameid=f)
    s = mm.metrics.create().compute(acc, metrics=MM_METRICS, name="x").iloc[0]
    c = {k: int(round(float(s[k]))) for k in
         ("num_objects", "num_matches", "num_switches", "num_false_positives", "num_misses",
          "num_detections", "num_fragmentations", "idtp", "idfp", "idfn")}
    motp = float(s["motp"]) if np.isfinite(s["motp"]) else 0.0
    c["num_misses"] = int(c["num_misses"])
    c["dist_sum"] = motp * c["num_detections"]
    c["num_frames"] = len(inputs)
    if return_events:
        return c, acc.mot_events
    return c


def derived(c: dict) -> dict:
    """Ratios from raw counts (sums), never from per-sequence ratios."""
    n = c["num_objects"]
    det = c["num_detections"]
    fp = c["num_false_positives"]
    d = dict(c)
    d["mota"] = 1.0 - (c["num_misses"] + fp + c["num_switches"]) / n if n else float("nan")
    d["motp"] = c["dist_sum"] / det if det else float("nan")
    denom = 2 * c["idtp"] + c["idfp"] + c["idfn"]
    d["idf1"] = 2.0 * c["idtp"] / denom if denom else float("nan")
    d["precision"] = det / (det + fp) if (det + fp) else float("nan")
    d["recall"] = det / n if n else float("nan")
    d["idsw_rate"] = c["num_switches"] / n if n else float("nan")
    return d


def pool(count_dicts: list[dict]) -> dict:
    """Sum raw counts over sequences and/or classes, then derive the ratios."""
    total = {k: 0 for k in COUNT_KEYS}
    for c in count_dicts:
        for k in COUNT_KEYS:
            total[k] += c.get(k, 0)
    return derived(total)


def evaluate_k2d(seq: SequenceGT, tracks: pd.DataFrame, meta: dict, emit_max_coast: int,
                 return_events: bool = False) -> dict:
    """{group: raw counts} for one sequence under P-K2D with emission policy E."""
    hyps = emission_hypotheses(tracks, meta, emit_max_coast)
    res = {}
    for grp in P.GROUPS:
        inp = k2d_inputs(seq, hyps, grp)
        res[grp] = counts_motmetrics(inp, return_events=return_events)
    return res


def evaluate_legacy(seq: SequenceGT, tracks: pd.DataFrame, gate: float = 2.0,
                    emit_max_coast: int | None = None, return_events: bool = False):
    return counts_motmetrics(legacy_inputs(seq, tracks, gate, emit_max_coast),
                             return_events=return_events)
