"""Automatic parts of the negative-MOTA investigation (PROTOCOL.md §6): FP attribution,
the H-ghost / H-scope / H-bias / H-depth tests, and the stage attribution of every miss.

Everything is computed from the eval_kitti dumps and the KITTI labels; the visual review of
the residual (A/B/C/D labels) is done separately from images (review_frames.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import det_eval as D
import mot_eval as M
import protocol as P
from bundle import Bundle
from geometry import ioa_matrix, iou_matrix

TWO_WHEELERS = (1, 3)
LEFT_MARGIN_PX = 128        # SGBM cannot estimate disparity in the first num_disparities columns
NEAR_RANGE_M = 4.0


# ------------------------------------------------------------------ legacy FP attribution
def legacy_fp_breakdown(b: Bundle, variant: str) -> dict:
    """Where the P-LEGACY false positives come from, each assigned to the FIRST rule that fits:
    ghost (coasting track) -> two-wheeler class -> in a DontCare region -> on an ignored/neutral
    label -> 2D-overlaps a valid label the 3D gate rejected -> too small -> residual."""
    tracks = b.tracks(variant)
    counts, events = M.evaluate_legacy(b.gt, tracks, return_events=True)
    fp = events[events["Type"] == "FP"].reset_index()
    trk = tracks.set_index(["frame", "id"])
    labels = b.labels
    real = labels[labels["type"] != "DontCare"]
    cat_by_group = {g: P.gt_categories(real, g) for g in P.GROUPS}
    out = {k: 0 for k in ("ghost_coasting", "two_wheeler_class", "in_dontcare", "on_ignored_or_neutral_label",
                          "overlaps_valid_label_3d_gate_rejected", "too_small", "residual")}
    residual_rows = []
    for _, e in fp.iterrows():
        f, h = int(e["FrameId"]), int(e["HId"])
        t = trk.loc[(f, h)]
        if isinstance(t, pd.DataFrame):
            t = t.iloc[0]
        box = np.array([[t["x"], t["y"], t["x"] + t["w"], t["y"] + t["h"]]])
        if t["tsu"] > 0:
            out["ghost_coasting"] += 1
            continue
        if int(t["cls"]) in TWO_WHEELERS:
            out["two_wheeler_class"] += 1
            continue
        dc = labels[(labels["frame"] == f) & (labels["type"] == "DontCare")][["x1", "y1", "x2", "y2"]].to_numpy()
        if len(dc) and (ioa_matrix(box, dc) >= P.DONTCARE_IOA).any():
            out["in_dontcare"] += 1
            continue
        grp = P.COCO_GROUP.get(int(t["cls"]))
        gt_f = real[real["frame"] == f]
        cats = cat_by_group[grp].loc[gt_f.index] if grp else None
        gboxes = gt_f[["x1", "y1", "x2", "y2"]].to_numpy()
        iou = iou_matrix(box, gboxes)[0] if len(gboxes) else np.zeros(0)
        if grp and len(iou):
            valid_hit = ((cats.to_numpy() == "valid") & (iou >= P.MATCH_IOU)).any()
            other_hit = ((cats.to_numpy() != "valid") & (cats.to_numpy() != "none") & (iou >= P.MATCH_IOU)).any() \
                or ((cats.to_numpy() == "none") & (iou >= P.MATCH_IOU)).any()
        else:
            valid_hit = other_hit = False
        if other_hit and not valid_hit:
            out["on_ignored_or_neutral_label"] += 1
        elif valid_hit:
            out["overlaps_valid_label_3d_gate_rejected"] += 1
        elif t["h"] < P.MIN_HEIGHT:
            out["too_small"] += 1
        else:
            out["residual"] += 1
            residual_rows.append({"frame": f, "id": h, "cls": int(t["cls"]), "x": t["x"], "y": t["y"],
                                  "w": t["w"], "h": t["h"], "Z": t["Z"]})
    total = int(sum(out.values()))
    return {"total_fp": total, "categories": out, "residual": pd.DataFrame(residual_rows),
            "counts": counts}


# ------------------------------------------------------------- hypothesis-level tests
def h_ghost(bundles: list[Bundle], variant: str) -> dict:
    """P-LEGACY re-scored with only tracks updated this frame."""
    rows = {}
    for label, E in (("all_confirmed(E=30)", None), ("updated_only(E=0)", 0)):
        cs = [M.evaluate_legacy(b.gt, b.tracks(variant), emit_max_coast=E) for b in bundles]
        rows[label] = M.pool(cs)
    a, z = rows["all_confirmed(E=30)"], rows["updated_only(E=0)"]
    return {"legacy_E30": _short(a), "legacy_E0": _short(z),
            "fp_drop_pct": 100.0 * (1 - z["num_false_positives"] / max(a["num_false_positives"], 1))}


def h_scope(bundles_rules: list[Bundle], bundles_norules: list[Bundle], variant: str) -> dict:
    """K2D with the ignore rules on vs off (same 2D IoU matching, same class groups)."""
    out = {}
    for E in (30, 0):
        on = M.pool([c for b in bundles_rules for c in M.evaluate_k2d(b.gt, b.tracks(variant), b.meta, E).values()])
        off = M.pool([c for b in bundles_norules for c in M.evaluate_k2d(b.gt, b.tracks(variant), b.meta, E).values()])
        out[f"E={E}"] = {"rules_on": _short(on), "rules_off": _short(off),
                         "fp_drop_pct": 100.0 * (1 - on["num_false_positives"] / max(off["num_false_positives"], 1))}
    return out


def h_bias(bundles: list[Bundle], variant: str) -> dict:
    """Signed depth bias of matched Vehicle pairs, and legacy recall at the 2 m vs a 4 m gate."""
    signed = []
    for b in bundles:
        tracks = b.tracks(variant)
        hyps = M.emission_hypotheses(tracks, b.meta, 0)
        veh = hyps[hyps["group"] == "Vehicle"]
        gt = b.gt.real
        cat = P.gt_categories(gt, "Vehicle")
        gv = gt[cat == "valid"]
        for f, g in veh.groupby("frame"):
            gf = gv[gv["frame"] == f]
            if gf.empty:
                continue
            iou = iou_matrix(g[["x1", "y1", "x2", "y2"]].to_numpy(), gf[["x1", "y1", "x2", "y2"]].to_numpy())
            for i in range(len(g)):
                j = int(np.argmax(iou[i]))
                if iou[i, j] >= P.MATCH_IOU:
                    signed.append(float(g.iloc[i]["Z"] - gf.iloc[j]["Z"]))
    rec = {}
    for gate in (2.0, 4.0):
        rec[f"gate_{gate:g}m"] = M.pool([M.evaluate_legacy(b.gt, b.tracks(variant), gate=gate) for b in bundles])["recall"]
    s = np.array(signed)
    return {"n_pairs": int(len(s)), "median_signed_depth_err_m": float(np.median(s)) if len(s) else float("nan"),
            "mean_signed_depth_err_m": float(np.mean(s)) if len(s) else float("nan"),
            "legacy_recall": rec, "recall_gain_points": 100.0 * (rec["gate_4m"] - rec["gate_2m"])}


def _short(c: dict) -> dict:
    return {k: c[k] for k in ("mota", "idf1", "num_objects", "num_false_positives", "num_misses",
                              "num_switches", "precision", "recall")}


# --------------------------------------------------------------------- FN attribution

def _stage_for_gt(g, df: pd.DataFrame, grp: str, score_thr: float) -> str:
    """First pipeline stage that lost one evaluable GT object, given that frame's detections."""
    gbox = np.array([[g["x1"], g["y1"], g["x2"], g["y2"]]])
    if df.empty:
        return "no_detection"
    iou = iou_matrix(gbox, df[["x", "y", "x2", "y2"]].to_numpy())[0]
    near = df[iou >= 0.3].assign(iou=iou[iou >= 0.3])
    if near.empty:
        return "no_detection"
    hi = near[near["score"] >= score_thr]
    if hi.empty:
        return "below_score_threshold"
    same = hi[hi["cls"].map(P.COCO_GROUP) == grp]
    if same.empty:
        return "wrong_or_excluded_class"
    if same["iou"].max() < P.MATCH_IOU:
        return "poor_localization"
    if (same[same["iou"] >= P.MATCH_IOU]["valid"] == 0).all():
        return "no_depth"
    return "tracker_lost_or_unconfirmed"


def legacy_fn_breakdown(b: Bundle, variant: str, score_thr: float = 0.25) -> pd.DataFrame:
    """Every P-LEGACY miss, split into: label not evaluable under the moderate protocol (with the
    reason) or evaluable, in which case the pipeline stage that lost it (plus `3d_gate_rejected`:
    a confirmed track overlapped it in 2D but sat > 2 m away in 3D)."""
    tracks = b.tracks(variant)
    _, ev = M.evaluate_legacy(b.gt, tracks, return_events=True)
    miss = ev[ev["Type"] == "MISS"].reset_index()
    real = b.gt.real.set_index(["frame", "track_id"])
    conf = tracks[tracks["confirmed"] == 1]
    dets = b.dets
    rows = []
    for _, e in miss.iterrows():
        f, oid = int(e["FrameId"]), int(e["OId"])
        g = real.loc[(f, oid)]
        if isinstance(g, pd.DataFrame):
            g = g.iloc[0]
        grp = next((k for k, v in P.VALID_TYPES.items() if g["type"] in v), None)
        height = g["y2"] - g["y1"]
        if grp is None:
            cat, reason = "not_evaluated_type", g["type"]
        elif height < P.MIN_HEIGHT:
            cat, reason = "fails_moderate_filter", "small(<25px)"
        elif g["occluded"] > P.MAX_OCCLUSION:
            cat, reason = "fails_moderate_filter", f"occluded={int(g['occluded'])}"
        elif g["truncated"] > P.MAX_TRUNCATION:
            cat, reason = "fails_moderate_filter", "truncated>0.3"
        else:
            cat = "evaluable"
            reason = _stage_for_gt(g, dets[dets["frame"] == f], grp, score_thr)
            if reason == "tracker_lost_or_unconfirmed":
                c = conf[conf["frame"] == f]
                if len(c) and (iou_matrix(np.array([[g["x1"], g["y1"], g["x2"], g["y2"]]]),
                                          np.stack([c["x"], c["y"], c["x"] + c["w"], c["y"] + c["h"]], axis=1))[0] >= 0.5).any():
                    reason = "3d_gate_rejected"
        rows.append({"seq": b.seq, "frame": f, "gt_id": oid, "type": g["type"], "category": cat,
                     "reason": reason, "height_px": float(height), "Z": float(g["Z"])})
    return pd.DataFrame(rows)

STAGES = ("no_detection", "below_score_threshold", "wrong_or_excluded_class", "poor_localization",
          "no_depth", "tracker_lost_or_unconfirmed")


def fn_stage_attribution(b: Bundle, variant: str, emit_max_coast: int, score_thr: float) -> pd.DataFrame:
    """For every evaluable GT object-frame that ends as a MISS under P-K2D, the first pipeline
    stage that lost it (PROTOCOL.md §6, in this order):
      no_detection            no detection (any class) with IoU >= 0.3 at the score floor
      below_score_threshold   overlapping detections exist but all score < the operating threshold
      wrong_or_excluded_class detections >= threshold overlap it but none is of its class group
      poor_localization       same-group detection >= threshold but best IoU < 0.5
      no_depth                same-group detection with IoU >= 0.5 but valid == 0
      tracker_lost_or_unconfirmed  a usable detection existed; the emitted tracks did not cover it
    """
    tracks = b.tracks(variant)
    hyps = M.emission_hypotheses(tracks, b.meta, emit_max_coast)
    dets = b.dets
    rows = []
    for grp in P.GROUPS:
        inp = M.k2d_inputs(b.gt, hyps, grp)
        _, ev = M.counts_motmetrics(inp, return_events=True)
        miss = ev[ev["Type"] == "MISS"].reset_index()
        lab = b.gt.real
        cat = P.gt_categories(lab, grp)
        gv = lab[cat == "valid"].set_index(["frame", "track_id"])
        for _, e in miss.iterrows():
            f, oid = int(e["FrameId"]), int(e["OId"])
            g = gv.loc[(f, oid)]
            if isinstance(g, pd.DataFrame):
                g = g.iloc[0]
            stage = _stage_for_gt(g, dets[dets["frame"] == f], grp, score_thr)
            cx = (g["x1"] + g["x2"]) / 2.0
            rows.append({"seq": b.seq, "frame": f, "gt_id": oid, "group": grp, "type": g["type"],
                         "stage": stage, "height_px": g["y2"] - g["y1"], "occluded": int(g["occluded"]),
                         "truncated": float(g["truncated"]), "Z": float(g["Z"]), "cx": float(cx),
                         "left_margin": bool(cx < LEFT_MARGIN_PX), "near": bool(g["Z"] < NEAR_RANGE_M)})
    return pd.DataFrame(rows)
