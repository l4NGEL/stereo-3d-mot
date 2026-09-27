#!/usr/bin/env python3
"""Tracking evaluation of PROTOCOL.md §3.3 / §5 / §3.8 over a set of sequences (dev or held-out).

Systems (names -> eval_kitti variant + emission policy E):
  S*          dev-selected system: plain IoU association, E = 0
  F, A1..A7   the fused system and its leave-one-out ablations (E = 0)
  *_E30       the same variants as shipped (emission E = 30: every confirmed track, coasting included)
  SORT        the depth-free 2D baseline (sort_baseline.py) on the same detections
For each: P-K2D pooled from raw counts (Vehicle, Person, both), and P-LEGACY (README-comparable).
Paired bootstrap over sequences for the comparisons the criteria need.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import mot_eval as M            # noqa: E402
import protocol as P            # noqa: E402
import sort_baseline as S       # noqa: E402
import stats as ST              # noqa: E402
from bundle import load_bundle  # noqa: E402

# system -> (variant name in the dump, emission policy)
TEST_SYSTEMS = {
    "S*": ("A5_plain_iou", 0), "F": ("F_fused", 0), "A1_no_appearance": ("A1_no_appearance", 0),
    "A2_no_iou_cost": ("A2_no_iou_cost", 0), "A3_no_3d_cost": ("A3_no_3d_cost", 0),
    "A4_plain_mahalanobis": ("A4_plain_mahalanobis", 0), "A7_greedy": ("A7_greedy", 0),
    "S*_E30(shipped iou2d)": ("A5_plain_iou", 30), "F_E30(shipped fused)": ("F_fused", 30),
    "A4_E30(shipped mahalanobis3d)": ("A4_plain_mahalanobis", 30),
}
DEV_SYSTEMS = {
    "S*": ("iou2d_s0.25_h3", 0), "F": ("fused_s0.25_h3", 0), "A4_plain_mahalanobis": ("mahalanobis3d_s0.25_h3", 0),
    "S*_E30(shipped iou2d)": ("iou2d_s0.25_h3", 30), "F_E30(shipped fused)": ("fused_s0.25_h3", 30),
    "A4_E30(shipped mahalanobis3d)": ("mahalanobis3d_s0.25_h3", 30),
}
COMPARISONS = [  # (a, b, why)
    ("S*", "SORT", "dev-selected system vs the depth-free baseline (TR-5)"),
    ("F", "SORT", "fused system vs the baseline"),
    ("A4_plain_mahalanobis", "SORT", "3D Mahalanobis vs the baseline"),
    ("F", "S*", "fusion of 3D+IoU+appearance vs the selected plain IoU"),
    ("F", "A1_no_appearance", "appearance cue"),
    ("F", "A2_no_iou_cost", "IoU term of the fused cost"),
    ("F", "A3_no_3d_cost", "3D term of the fused cost"),
    ("F", "A7_greedy", "Hungarian vs greedy assignment"),
    ("A4_plain_mahalanobis", "S*", "3D Mahalanobis vs 2D IoU association"),
    ("S*", "S*_E30(shipped iou2d)", "emission E=0 vs shipped E=30"),
    ("F", "F_E30(shipped fused)", "emission E=0 vs shipped E=30 (fused)"),
]
METRICS = ["mota", "idf1"]


def group_counts(res: dict) -> dict:
    return {g: res[g] for g in P.GROUPS}


def score_system(bundles, variant, E):
    per_seq = {}
    for s, b in bundles.items():
        per_seq[s] = M.evaluate_k2d(b.gt, b.tracks(variant), b.meta, E)
    return per_seq


def summarise(per_seq: dict) -> dict:
    both = {s: M.pool(list(r.values())) for s, r in per_seq.items()}
    veh = M.pool([r["Vehicle"] for r in per_seq.values()])
    per = M.pool([r["Person"] for r in per_seq.values()])
    allp = M.pool(list(both.values()))
    keep = ("mota", "idf1", "precision", "recall", "num_objects", "num_false_positives", "num_misses",
            "num_switches", "num_fragmentations", "idsw_rate", "motp")
    short = lambda d: {k: (float(d[k]) if isinstance(d[k], (float, np.floating)) else int(d[k])) for k in keep}  # noqa: E731
    return {"pooled": short(allp), "vehicle": short(veh), "person": short(per),
            "mean_iou_of_matches": 1.0 - allp["motp"] if allp["motp"] == allp["motp"] else None,
            "per_sequence": {s: {"mota": both[s]["mota"], "idf1": both[s]["idf1"], "num_objects": both[s]["num_objects"],
                                 "fp": both[s]["num_false_positives"], "fn": both[s]["num_misses"],
                                 "idsw": both[s]["num_switches"]} for s in both}}, both


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--kitti-root", default="data/kitti")
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--kind", choices=["dev", "test"], required=True)
    ap.add_argument("--sort-score", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()

    systems = DEV_SYSTEMS if a.kind == "dev" else TEST_SYSTEMS
    bundles = {s: load_bundle(a.raw, a.kitti_root, s) for s in a.seqs}
    out = {"kind": a.kind, "sequences": a.seqs, "n_frames": {s: b.n_frames for s, b in bundles.items()}, "systems": {}}
    both_by_system: dict[str, dict] = {}

    for name, (variant, E) in systems.items():
        per_seq = score_system(bundles, variant, E)
        summary, both = summarise(per_seq)
        summary["variant"], summary["emission_E"] = variant, E
        out["systems"][name] = summary
        both_by_system[name] = both
        p = summary["pooled"]
        print(f"{name:32s} pooled MOTA {p['mota']:+.3f} IDF1 {p['idf1']:.3f} FP {p['num_false_positives']:5d} FN {p['num_misses']:5d} "
              f"IDSW {p['num_switches']:3d} GT {p['num_objects']} | Vehicle MOTA {summary['vehicle']['mota']:+.3f} IDF1 {summary['vehicle']['idf1']:.3f}"
              f" | Person MOTA {summary['person']['mota']:+.3f} IDF1 {summary['person']['idf1']:.3f}", flush=True)

    # SORT-2D on identical detections (same score threshold, all kept classes, no depth)
    sort_per_seq = {}
    for s, b in bundles.items():
        d = b.dets[b.dets["score"] >= a.sort_score]
        tr = S.sort_track(d, b.n_frames)
        sort_per_seq[s] = M.evaluate_k2d(b.gt, tr, b.meta, 0)
    summary, both = summarise(sort_per_seq)
    summary["variant"], summary["emission_E"] = f"SORT(score>={a.sort_score})", 0
    out["systems"]["SORT"] = summary
    both_by_system["SORT"] = both
    p = summary["pooled"]
    print(f"{'SORT':32s} pooled MOTA {p['mota']:+.3f} IDF1 {p['idf1']:.3f} FP {p['num_false_positives']:5d} FN {p['num_misses']:5d} IDSW {p['num_switches']:3d}"
          f" GT {p['num_objects']} | Vehicle MOTA {summary['vehicle']['mota']:+.3f} | Person MOTA {summary['person']['mota']:+.3f}", flush=True)

    # P-LEGACY for the as-shipped triple (README-comparable) and with only-updated tracks
    out["legacy"] = {}
    for name, variant in (("mahalanobis3d", "A4_plain_mahalanobis" if a.kind == "test" else "mahalanobis3d_s0.25_h3"),
                          ("iou2d", "A5_plain_iou" if a.kind == "test" else "iou2d_s0.25_h3"),
                          ("fused", "F_fused" if a.kind == "test" else "fused_s0.25_h3")):
        per = {s: M.evaluate_legacy(b.gt, b.tracks(variant)) for s, b in bundles.items()}
        per0 = {s: M.evaluate_legacy(b.gt, b.tracks(variant), emit_max_coast=0) for s, b in bundles.items()}
        out["legacy"][name] = {
            "as_shipped_all_confirmed": {**{k: v for k, v in M.pool(list(per.values())).items()
                                             if k in ("mota", "motp", "idf1", "precision", "recall", "num_objects", "num_false_positives",
                                                      "num_misses", "num_switches", "num_fragmentations")},
                                          "per_sequence": {s: {k: M.derived(c)[k] for k in ("mota", "idf1", "num_false_positives", "num_misses", "num_switches", "num_objects")}
                                                           for s, c in per.items()}},
            "updated_tracks_only": {k: v for k, v in M.pool(list(per0.values())).items()
                                    if k in ("mota", "idf1", "precision", "recall", "num_objects", "num_false_positives", "num_misses", "num_switches")}}
        lg = out["legacy"][name]["as_shipped_all_confirmed"]
        print(f"LEGACY {name:14s} pooled MOTA {lg['mota']:+.3f} IDF1 {lg['idf1']:.3f} FP {lg['num_false_positives']} FN {lg['num_misses']} IDSW {lg['num_switches']} GT {lg['num_objects']}", flush=True)

    # paired bootstrap over sequences
    out["comparisons"] = []
    for x, y, why in COMPARISONS:
        if x in both_by_system and y in both_by_system:
            for m in METRICS:
                r = ST.paired_bootstrap(both_by_system[x], both_by_system[y], m, n_boot=a.n_boot)
                r.update({"a": x, "b": y, "why": why})
                out["comparisons"].append(r)
    out["ci95_pooled"] = {}
    for name in ("S*", "F", "SORT"):
        if name in both_by_system:
            out["ci95_pooled"][name] = {m: ST.bootstrap_ci(both_by_system[name], m, n_boot=a.n_boot) for m in METRICS}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    for c in out["comparisons"]:
        print(f"  {c['a']:24s} - {c['b']:30s} {c['metric']:5s} {c['diff']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}] "
              f"{'EXCLUDES 0' if c['excludes_zero'] else 'inside noise':12s} wins {c['wins_a']}/{c['n_seq']} vs {c['wins_b']}  ({c['why']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
