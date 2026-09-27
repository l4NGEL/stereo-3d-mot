#!/usr/bin/env python3
"""Phase 2 final measurement (docs/evaluation/REPORT.md remediation log): the ONE dev-selected
range-gate configuration (range_gate_max_m=10, docs/evaluation/results/phase2_range_gate_selection.json),
measured once on held-out data, against SORT-2D, the fixed plain-IoU tracker (S*) and the fixed plain-3D
tracker (A4_plain_mahalanobis) -- all four already-frozen or already-fixed systems, held-out P-K2D pooled,
paired bootstrap over sequences. Also reports P-LEGACY as a secondary, more intuitive view."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import mot_eval as M            # noqa: E402
import sort_baseline as S       # noqa: E402
import stats as ST              # noqa: E402
from bundle import load_bundle  # noqa: E402

TEST = ["0001", "0002", "0004", "0007", "0008", "0010", "0012", "0016"]
N_BOOT = 10000


def main() -> int:
    b_main = {s: load_bundle("out/eval/raw/test_kitti_tiled", "data/kitti", s) for s in TEST}
    b_gate = {s: load_bundle("out/eval/raw/test_range_gate", "data/kitti", s) for s in TEST}

    systems = {}
    for name, source, variant in (
        ("range_gated_10", b_gate, "range_gate_10"),
        ("S*_iou2d_E0", b_main, "A5_plain_iou"),
        ("A4_mahalanobis3d_E0", b_main, "A4_plain_mahalanobis"),
    ):
        k2d, legacy = {}, {}
        for s in TEST:
            bnd = source[s]
            k2d[s] = M.pool(list(M.evaluate_k2d(bnd.gt, bnd.tracks(variant), bnd.meta, 0).values()))
            legacy[s] = M.evaluate_legacy(bnd.gt, bnd.tracks(variant), gate=2.0, emit_max_coast=0)
        systems[name] = {"k2d": k2d, "legacy": legacy}

    sort_k2d, sort_legacy = {}, {}
    for s in TEST:
        bnd = b_main[s]
        d = bnd.dets[bnd.dets["score"] >= 0.25]
        tr = S.sort_track(d, bnd.n_frames)
        sort_k2d[s] = M.pool(list(M.evaluate_k2d(bnd.gt, tr, bnd.meta, 0).values()))
        sort_legacy[s] = M.evaluate_legacy(bnd.gt, tr, gate=2.0, emit_max_coast=0)  # informational only, see caveat
    systems["SORT"] = {"k2d": sort_k2d, "legacy": sort_legacy}

    def summarise(per_seq):
        p = M.pool(list(per_seq.values()))
        return {k: p[k] for k in ("mota", "idf1", "precision", "recall", "num_objects",
                                  "num_false_positives", "num_misses", "num_switches")}

    out = {"pooled": {}, "comparisons": []}
    for name, d in systems.items():
        out["pooled"][name] = {"K2D": summarise(d["k2d"]), "LEGACY": summarise(d["legacy"])}
        p = out["pooled"][name]["K2D"]
        print(f"{name:22s} K2D  MOTA {p['mota']:+.3f} IDF1 {p['idf1']:.3f} FP {p['num_false_positives']:5d} "
              f"FN {p['num_misses']:5d} IDSW {p['num_switches']:4d}")

    pairs = [("range_gated_10", "SORT"), ("range_gated_10", "S*_iou2d_E0"), ("range_gated_10", "A4_mahalanobis3d_E0")]
    for a, bb in pairs:
        for metric in ("mota", "idf1"):
            r = ST.paired_bootstrap(systems[a]["k2d"], systems[bb]["k2d"], metric, N_BOOT)
            out["comparisons"].append({"a": a, "b": bb, "protocol": "K2D", **r})
            print(f"  {a} - {bb:20s} [K2D] {metric}: {r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] "
                  f"{'EXCLUDES 0' if r['excludes_zero'] else 'inside noise'} wins {r['wins_a']}/{r['n_seq']} vs {r['wins_b']}")

    print("\nNote: SORT's P-LEGACY numbers are not meaningful (SORT has no 3D output, see run_phase1_ghost_fix.py) "
          "-- reported under 'pooled' for completeness only, not compared.")
    os.makedirs("docs/evaluation/results", exist_ok=True)
    json.dump(out, open("docs/evaluation/results/phase2_range_gate_measurement.json", "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
