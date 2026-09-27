#!/usr/bin/env python3
"""Phase 2 dev-only selection of kRangeGated3D's range_gate_max_m (docs/evaluation/REPORT.md
remediation log). Selection rule, fixed before this script's output was looked at: maximise
pooled dev P-K2D MOTA at emission policy E=0 (the product's own post-Phase-1 behavior);
ties within 0.005 broken toward the SMALLER range_gate_max_m (falls back to IoU more often --
the more conservative choice given SORT's own held-out edge, fcde488 TR-5). score=0.25,
min_hits=3 held fixed at fcde488's dev-selected values (configs/eval/variants_range_gate_dev.txt).

Held-out data is not touched by this script."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import mot_eval as M            # noqa: E402
from bundle import load_bundle  # noqa: E402

DEV = ["0000", "0003", "0017"]
RAW = "out/eval/raw/dev_range_gate"
TIE = 0.005


def main() -> int:
    bundles = {s: load_bundle(RAW, "data/kitti", s) for s in DEV}
    variants = bundles[DEV[0]].variants()
    rows = []
    for name in sorted(variants):
        counts = []
        for s, b in bundles.items():
            counts += list(M.evaluate_k2d(b.gt, b.tracks(name), b.meta, 0).values())
        p = M.pool(counts)
        r_val = 1e9 if name.endswith("unlimited") else int(name.rsplit("_", 1)[1])
        rows.append({"variant": name, "range_gate_max_m": r_val, "mota": p["mota"], "idf1": p["idf1"],
                    "fp": p["num_false_positives"], "fn": p["num_misses"], "idsw": p["num_switches"],
                    "num_objects": p["num_objects"]})
        print(f"{name:22s} range={r_val:>10.0f} m  MOTA {p['mota']:+.4f}  IDF1 {p['idf1']:.4f}  "
              f"FP {p['num_false_positives']:4d}  FN {p['num_misses']:4d}  IDSW {p['num_switches']:3d}")

    best = max(r["mota"] for r in rows)
    tied = [r for r in rows if r["mota"] >= best - TIE]
    chosen = min(tied, key=lambda r: r["range_gate_max_m"])
    out = {"rule": "max pooled dev K2D MOTA (E=0); ties within 0.005 -> smallest range_gate_max_m",
          "grid": rows, "best_mota": best, "n_tied": len(tied), "chosen": chosen}
    os.makedirs("docs/evaluation/results", exist_ok=True)
    json.dump(out, open("docs/evaluation/results/phase2_range_gate_selection.json", "w"), indent=1)
    print(f"\nCHOSEN: {chosen['variant']} (range_gate_max_m={chosen['range_gate_max_m']}), "
          f"MOTA={chosen['mota']:.4f}, {len(tied)} variant(s) within the tie band")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
