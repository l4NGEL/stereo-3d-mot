#!/usr/bin/env python3
"""The automatic parts of the negative-MOTA investigation for one set of sequences (PROTOCOL.md §6):
legacy FP attribution, H-ghost, H-scope, H-bias, FN stage attribution (K2D) and the breakdown of
every legacy miss (the 0003 low-recall question). Writes JSON + CSV."""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import diagnose as X            # noqa: E402
from bundle import load_bundle  # noqa: E402

NAMES = {
    "dev": {"legacy": "mahalanobis3d_s0.25_h3", "selected": "iou2d_s0.25_h3"},
    "test": {"legacy": "A4_plain_mahalanobis", "selected": "A5_plain_iou"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--kitti-root", default="data/kitti")
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--kind", choices=["dev", "test"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    legacy_v, sel_v = NAMES[a.kind]["legacy"], NAMES[a.kind]["selected"]

    B = [load_bundle(a.raw, a.kitti_root, s) for s in a.seqs]
    BN = [load_bundle(a.raw, a.kitti_root, s, rules=False) for s in a.seqs]
    out = {"kind": a.kind, "sequences": a.seqs, "legacy_variant": legacy_v, "selected_variant": sel_v}

    # legacy false positives, by the first rule that explains each
    per_seq, tot = {}, {}
    for b in B:
        r = X.legacy_fp_breakdown(b, legacy_v)
        per_seq[b.seq] = {"total_fp": r["total_fp"], **r["categories"]}
        for k, v in r["categories"].items():
            tot[k] = tot.get(k, 0) + v
    n = sum(tot.values())
    out["legacy_fp_breakdown"] = {"total": n, "categories": tot,
                                  "shares": {k: v / n for k, v in tot.items()}, "per_sequence": per_seq}
    print(f"legacy FP ({legacy_v}): total {n}; " + ", ".join(f"{k} {v} ({100*v/n:.1f}%)" for k, v in tot.items()))

    out["H_ghost"] = X.h_ghost(B, legacy_v)
    print("H-ghost: legacy FP", out["H_ghost"]["legacy_E30"]["num_false_positives"], "->", out["H_ghost"]["legacy_E0"]["num_false_positives"],
          f"({out['H_ghost']['fp_drop_pct']:.1f}% drop); MOTA {out['H_ghost']['legacy_E30']['mota']:+.3f} -> {out['H_ghost']['legacy_E0']['mota']:+.3f}")
    out["H_scope"] = X.h_scope(B, BN, legacy_v)
    for k, v in out["H_scope"].items():
        print(f"H-scope {k}: K2D FP rules-off {v['rules_off']['num_false_positives']} -> rules-on {v['rules_on']['num_false_positives']} ({v['fp_drop_pct']:.1f}% drop)")
    out["H_bias"] = X.h_bias(B, legacy_v)
    hb = out["H_bias"]
    print(f"H-bias: median signed depth error (track Z - label centre Z) {hb['median_signed_depth_err_m']:+.2f} m over {hb['n_pairs']} matched Vehicles; "
          f"legacy recall {hb['legacy_recall']['gate_2m']:.3f} (2 m) -> {hb['legacy_recall']['gate_4m']:.3f} (4 m), +{hb['recall_gain_points']:.1f} points")

    # K2D misses of the dev-selected system: which stage lost each one
    fn = pd.concat([X.fn_stage_attribution(b, sel_v, 0, 0.25) for b in B], ignore_index=True)
    fn.to_csv(a.out.replace(".json", "_k2d_fn_stages.csv"), index=False)
    out["K2D_fn_stages_selected_system"] = {"total": int(len(fn)), "by_stage": fn["stage"].value_counts().to_dict(),
                                            "by_sequence": {s: fn[fn.seq == s]["stage"].value_counts().to_dict() for s in a.seqs},
                                            "no_depth_left_margin": int(fn[(fn.stage == "no_depth") & fn.left_margin].shape[0]),
                                            "no_depth_total": int((fn.stage == "no_depth").sum())}
    print("K2D FN stages (selected system, E=0):", out["K2D_fn_stages_selected_system"]["by_stage"], "| no_depth in left margin:",
          out["K2D_fn_stages_selected_system"]["no_depth_left_margin"], "of", out["K2D_fn_stages_selected_system"]["no_depth_total"])

    # every legacy miss
    lf = pd.concat([X.legacy_fn_breakdown(b, legacy_v) for b in B], ignore_index=True)
    lf.to_csv(a.out.replace(".json", "_legacy_fn.csv"), index=False)
    out["legacy_fn_breakdown"] = {"total": int(len(lf)),
                                  "by_category": lf["category"].value_counts().to_dict(),
                                  "by_category_reason": {f"{c}/{r}": int(v) for (c, r), v in lf.groupby(["category", "reason"]).size().items()},
                                  "by_sequence": {s: {f"{c}/{r}": int(v) for (c, r), v in lf[lf.seq == s].groupby(["category", "reason"]).size().items()} for s in a.seqs}}
    print("legacy misses:", out["legacy_fn_breakdown"]["total"], out["legacy_fn_breakdown"]["by_category"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
