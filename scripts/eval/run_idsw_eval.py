#!/usr/bin/env python3
"""ID-switch follow-up (docs/evaluation/IDSW_PLAN.md, IDSW_REPORT.md).

Scores tracker variants and SORT with the unchanged P-K2D scorer (mot_eval.evaluate_k2d, E = 0)
and classifies every identity switch py-motmetrics emits into the plan's classes A-D, using the
tracker's own dump for internal state (tsu, birth) -- never the scorer's output -- and, when the
dump has them, the per-pair association diagnostics (assoc_<variant>.txt, eval_kitti --diag).

  --mode select : dev; applies the plan's selection rule, writes the selection JSON
  --mode report : any split; metrics, per-sequence table, event classes, paired bootstrap
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import kitti_io as K            # noqa: E402
import mot_eval as M            # noqa: E402
import protocol as P            # noqa: E402
import sort_baseline as S       # noqa: E402
import stats as ST              # noqa: E402
from bundle import load_bundle  # noqa: E402
from geometry import iou_matrix  # noqa: E402

SCORE = 0.25
KEEP = ("mota", "idf1", "recall", "precision", "num_switches", "num_fragmentations",
        "num_false_positives", "num_misses", "idtp", "idfp", "idfn", "num_objects")
ASSOC_COLS = ["frame", "track_id", "det", "tsu_before", "confirmed_before", "iou_last", "iou_pred",
              "mahal_sq", "app", "stage"]


def read_assoc(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep=" ", comment="#", header=None, names=ASSOC_COLS)
    return df


def seq_counts(res: dict) -> dict:
    """Raw counts of one sequence summed over the two class groups."""
    tot = {k: 0 for k in M.COUNT_KEYS}
    for grp in P.GROUPS:
        for k in M.COUNT_KEYS:
            tot[k] += res[grp][0].get(k, 0)
    return tot


def short(d: dict) -> dict:
    return {k: (float(d[k]) if isinstance(d[k], (float, np.floating)) else int(d[k])) for k in KEEP}


def run_system(b, kind: str, variant: str | None):
    """-> (tracks for scoring, internal state df [frame,id,tsu,age], assoc df or None)."""
    if kind == "dump":
        tr = b.tracks(variant)
        return tr, tr[["frame", "id", "tsu", "age"]], read_assoc(os.path.join(b.raw_dir, f"assoc_{variant}.txt"))
    d = b.dets[b.dets["score"] >= SCORE]
    if kind == "sort_valid":
        d = d[d["valid"] == 1]
    max_age = 30 if kind == "sort_age30" else 1
    tr, internal = S.sort_track(d, b.n_frames, max_age=max_age, return_internal=True)
    return tr, internal, None


def classify(b, name, tr, internal, assoc, res) -> list[dict]:
    lab = b.labels[b.labels["type"] != "DontCare"]
    lab_by = {(int(r.frame), int(r.track_id)): r for r in lab.itertuples()}
    tsu_at = {(int(r.frame), int(r.id)): int(r.tsu) for r in internal.itertuples()}
    birth = (internal["frame"] - internal["age"]).groupby(internal["id"]).min().to_dict()
    dets = b.dets
    dets_by_frame = {int(f): g for f, g in dets.groupby("frame")}
    a_idx = None
    if assoc is not None:
        a_idx = {(int(r.frame), int(r.track_id), int(r.det)): r for r in assoc.itertuples()}
        a_match = {(int(r.frame), int(r.track_id)): r for r in assoc[assoc.stage > 0].itertuples()}
    out = []
    for grp in P.GROUPS:
        ev = res[grp][1].reset_index()
        ev = ev[ev.Type.isin(["MATCH", "SWITCH", "MISS"])]
        hit = ev[ev.Type.isin(["MATCH", "SWITCH"])]
        miss_frames = {(int(r.FrameId), int(r.OId)) for r in ev[ev.Type == "MISS"].itertuples()}
        hyp_hist = {}
        for r in hit.itertuples():
            hyp_hist.setdefault(int(r.HId), []).append((int(r.FrameId), int(r.OId)))
        last = {}
        for r in hit.sort_values("FrameId", kind="stable").itertuples():
            o, h, f = int(r.OId), int(r.HId), int(r.FrameId)
            if r.Type == "SWITCH" and o in last:
                f0, h_old = last[o]
                gap = f - f0
                tsu_new_prev = tsu_at.get((f - 1, h))
                if gap == 1:
                    cls = "A" if (tsu_new_prev is not None and tsu_new_prev >= 1) else "D"
                    sub = ("lost track took over" if cls == "A" else
                           "swap with active track" if tsu_new_prev == 0 else "new track at f")
                else:
                    cls = "B" if birth.get(h, 10**9) <= f0 else "C"
                    sub = "existing track" if cls == "B" else "new identity"
                prev_other = sorted({oo for (ff, oo) in hyp_hist.get(h, []) if ff < f and oo != o})
                prev_same = any(ff < f0 and oo == o for (ff, oo) in hyp_hist.get(h, []))
                row = dict(sys=name, seq=b.seq, grp=grp, frame=f, gt=o, h_old=h_old, h_new=h, gap=gap,
                           cls=cls, sub=sub, tsu_new_prev=tsu_new_prev,
                           tsu_old_at_f=tsu_at.get((f, h_old)),
                           h_new_born=birth.get(h), h_new_prev_other_gt=len(prev_other) > 0,
                           h_new_prev_same_gt=prev_same)
                # gap anatomy (gap > 1)
                if gap > 1:
                    n_scored = n_filtered = n_absent = 0
                    det_kind = {"no_det": 0, "below_score": 0, "no_depth": 0, "usable": 0}
                    occ = []
                    for g in range(f0 + 1, f):
                        lr = lab_by.get((g, o))
                        if lr is not None:
                            occ.append(int(lr.occluded))
                        if (g, o) in miss_frames:
                            n_scored += 1
                            gd = dets_by_frame.get(g)
                            box = np.array([[lr.x1, lr.y1, lr.x2, lr.y2]])
                            if gd is None or gd.empty:
                                det_kind["no_det"] += 1
                                continue
                            ious = iou_matrix(box, gd[["x", "y", "x2", "y2"]].to_numpy())[0]
                            m = gd[ious >= P.MATCH_IOU]
                            if m.empty:
                                det_kind["no_det"] += 1
                            elif (m.score >= SCORE).sum() == 0:
                                det_kind["below_score"] += 1
                            elif ((m.score >= SCORE) & (m.valid == 1)).sum() == 0:
                                det_kind["no_depth"] += 1
                            else:
                                det_kind["usable"] += 1
                        elif lr is not None:
                            n_filtered += 1
                        else:
                            n_absent += 1
                    row.update(gap_scored_missed=n_scored, gap_filtered=n_filtered, gap_absent=n_absent,
                               gap_max_occ=max(occ) if occ else -1,
                               **{f"gap_{k}": v for k, v in det_kind.items()})
                # association diagnostics at the switch frame
                if a_idx is not None:
                    m = a_match.get((f, h))
                    if m is not None:
                        row.update(stage=int(m.stage), new_iou_last=m.iou_last, new_iou_pred=m.iou_pred,
                                   new_mahal_sq=m.mahal_sq, new_app=m.app, new_tsu_before=int(m.tsu_before))
                        c = a_idx.get((f, h_old, int(m.det)))
                        if c is not None:
                            row.update(old_iou_last=c.iou_last, old_iou_pred=c.iou_pred,
                                       old_mahal_sq=c.mahal_sq, old_app=c.app, old_tsu_before=int(c.tsu_before))
                out.append(row)
            last[o] = (f, h)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--kitti-root", default="data/kitti")
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", required=True, help="dump variant names")
    ap.add_argument("--mode", choices=["select", "report"], required=True)
    ap.add_argument("--selection", help="report: selection JSON from --mode select (recorded only)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--events-csv")
    ap.add_argument("--round", type=int, choices=[1, 2], default=1,
                    help="2: IDSW_PLAN.md round 2 (reconfirm x max_age grid, base = R0_age30)")
    a = ap.parse_args()

    bundles = {s: load_bundle(a.raw, a.kitti_root, s) for s in a.seqs}
    systems = [(v, "dump", v) for v in a.variants] + [("SORT", "sort", None), ("SORT_valid", "sort_valid", None)]
    if a.round == 2:
        systems.append(("SORT_age30", "sort_age30", None))  # diagnostic, not a candidate
    per_seq, events = {}, []
    for name, kind, variant in systems:
        per_seq[name] = {}
        for s, b in bundles.items():
            tr, internal, assoc = run_system(b, kind, variant)
            res = M.evaluate_k2d(b.gt, tr, b.meta, 0, return_events=True)
            per_seq[name][s] = seq_counts(res)
            events += classify(b, name, tr, internal, assoc, res)
    ev = pd.DataFrame(events)
    if a.events_csv:
        ev.to_csv(a.events_csv, index=False)

    out = {"sequences": a.seqs, "score": SCORE, "emission_E": 0, "systems": {}}
    for name, _, _ in systems:
        pooled = M.pool(list(per_seq[name].values()))
        e = ev[ev.sys == name] if len(ev) else ev
        cls = {c: int((e.cls == c).sum()) for c in "ABCD"} if len(e) else {c: 0 for c in "ABCD"}
        assert sum(cls.values()) == pooled["num_switches"], (name, cls, pooled["num_switches"])
        wo = [s for s in a.seqs if s != "0001"]
        out["systems"][name] = {
            "pooled": short(pooled),
            "idsw_classes": cls,
            "pooled_without_0001": short(M.pool([per_seq[name][s] for s in wo])) if "0001" in a.seqs else None,
            "per_sequence": {s: short(M.pool([c])) for s, c in per_seq[name].items()},
        }

    if a.mode == "select" and a.round == 2:
        def closeness(n):  # reconfirm off first, then larger max_age (closer to C)
            r, age = n[1], int(n.split("_age")[1])
            return (int(r), -age)
        best = max(out["systems"][n]["pooled"]["idf1"] for n in a.variants)
        tied = [n for n in a.variants if out["systems"][n]["pooled"]["idf1"] >= best - 0.005]
        tied.sort(key=lambda n: (out["systems"][n]["pooled"]["num_switches"],
                                 -out["systems"][n]["pooled"]["mota"], closeness(n)))
        out["selection"] = {"rule": "docs/evaluation/IDSW_PLAN.md round 2: max pooled dev IDF1, ties within 0.005 -> "
                                    "fewer IDSW -> higher MOTA -> reconfirm off, larger max_age",
                            "selected": tied[0], "tied_within_0.005": tied}
        print("round-2 selected:", tied[0], "| tied:", tied)
    elif a.mode == "select":
        def pick(cands, order_key):
            best = max(out["systems"][n]["pooled"]["idf1"] for n in cands)
            tied = [n for n in cands if out["systems"][n]["pooled"]["idf1"] >= best - 0.005]
            tied.sort(key=lambda n: (out["systems"][n]["pooled"]["num_switches"],
                                     -out["systems"][n]["pooled"]["mota"], order_key(n)))
            return tied[0], tied

        grid = [v for v in a.variants if v.startswith("D2_")]

        def strictness(n):
            chi = float(n.split("_chi")[1].split("_app")[0])
            app = float(n.split("_app")[1])
            return (chi, app)

        d2, d2_tied = pick(grid, strictness)
        order = ["A_iou_last", "B_iou_last_age1", "C_iou_pred", "D0_two_stage_none", "D1_two_stage_iou", d2]
        rec, rec_tied = pick(order, order.index)
        out["selection"] = {"rule": "docs/evaluation/IDSW_PLAN.md: max pooled dev IDF1, ties within 0.005 -> "
                                    "fewer IDSW -> higher MOTA -> stricter / simpler",
                            "d2_selected": d2, "d2_tied_within_0.005": d2_tied,
                            "recommended": rec, "recommended_tied_within_0.005": rec_tied}
        print("D2 selected:", d2, "| recommended:", rec)
    else:
        comps = []
        base = "A_iou_last" if a.round == 1 else "R0_age30"
        others = [v for v in a.variants if v != base]
        for v in others + ["SORT", "SORT_valid"]:
            comps.append((v, base))
        for v in a.variants:
            comps.append((v, "SORT"))
        # separating D's components: priority alone (D1 vs C), the stricter lost-track rule
        # (D2 vs D1), and removing recovery altogether (D0 vs C)
        d2 = [v for v in a.variants if v.startswith("D2_")]
        for x, y in [("D1_two_stage_iou", "C_iou_pred"), ("D0_two_stage_none", "C_iou_pred")] + \
                [(v, "C_iou_pred") for v in d2] + [(v, "D1_two_stage_iou") for v in d2]:
            if x in a.variants and y in a.variants:
                comps.append((x, y))
        if a.round == 2:  # factor contrasts: each factor with the other held fixed
            for x, y in [("R1_age30", "R0_age30"), ("R1_age1", "R0_age1"),
                         ("R0_age1", "R0_age30"), ("R1_age1", "R1_age30")]:
                if x in a.variants and y in a.variants:
                    comps.append((x, y))
        out["comparisons"] = []
        for x, y in comps:
            for metric in ("idf1", "mota", "idsw_rate", "recall"):
                r = ST.paired_bootstrap(per_seq[x], per_seq[y], metric, n_boot=a.n_boot)
                out["comparisons"].append({"a": x, "b": y, **r})
        if a.selection:
            out["selection_used"] = json.load(open(a.selection))["selection"]

    # input difference between the project's tracker and SORT: valid-depth filter
    inp = {}
    for s, b in bundles.items():
        d = b.dets[b.dets["score"] >= SCORE]
        inp[s] = {"dets_score_ge_0.25": int(len(d)), "of_which_valid_depth": int((d.valid == 1).sum())}
    out["input_detections"] = inp

    json.dump(out, open(a.out, "w"), indent=1)
    for name, _, _ in systems:
        p = out["systems"][name]["pooled"]
        c = out["systems"][name]["idsw_classes"]
        print(f"{name:22s} IDF1 {p['idf1']:.3f} MOTA {p['mota']:+.3f} IDSW {p['num_switches']:4d} "
              f"Frag {p['num_fragmentations']:4d} rec {p['recall']:.3f} prec {p['precision']:.3f} "
              f"FP {p['num_false_positives']:5d} FN {p['num_misses']:5d} | A{c['A']} B{c['B']} C{c['C']} D{c['D']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
