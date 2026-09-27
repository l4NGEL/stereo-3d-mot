#!/usr/bin/env python3
"""Phase 1 of the fcde488 remediation: measures the emission fix (isEmittable() in
tracker.hpp -- a track is only reported if confirmed AND matched by a real detection
this frame; internal state/matching ability is untouched, see the code comment).

Three systems, same detections, same sequences as fcde488:
  OLD    shipped default (association=mahalanobis3d, E=30 -- every confirmed track,
         coasting included up to max_age) -- what the tracker emitted before this fix
  FIXED  same association, E=0 -- what the fixed binary now emits (isEmittable())
  SORT   the depth-free baseline (unchanged)

Two independent pieces of evidence:
  (a) native: the ACTUAL fixed apps/benchmark_kitti binary run on real dev-sequence data
      (out/eval/raw/cpp_benchmark_kitti_fixed/, produced by a separate docker run, not
      by this script) -- cross-checked here against the Python E=0 filter on the SAME
      underlying eval_kitti dumps (which apply score>=0.25, min_hits=3, identically).
  (b) held-out: P-K2D and P-LEGACY, OLD vs FIXED vs SORT, from the frozen fcde488 dumps
      (out/eval/raw/{dev,test}_kitti_tiled) -- nothing about the dumps, the sequences,
      the seed, or the metric definitions changes; only which of the already-dumped
      per-frame track states counts as "emitted" changes, which is exactly what E does.

Writes docs/evaluation/results/phase1_ghost_fix.json. Does NOT touch tracking_test.json,
tracking_dev.json, or any other fcde488 artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import mot_eval as M            # noqa: E402
import stats as ST              # noqa: E402
from bundle import load_bundle  # noqa: E402

DEV = ["0000", "0003", "0017"]
TEST = ["0001", "0002", "0004", "0007", "0008", "0010", "0012", "0016"]


def parse_cpp(path: str) -> dict:
    kv = {}
    for line in open(path):
        m = re.match(r"^(3D Mahalanobis|2D IoU)\s*:\s*(.+)$", line)
        if not m:
            continue
        name = "mahalanobis3d" if m.group(1) == "3D Mahalanobis" else "iou2d"
        fields = dict(re.findall(r"(\w+)=([-\d.e+]+)", m.group(2)))
        kv[name] = {"mota": float(fields["MOTA"]), "idf1": float(fields["IDF1"]),
                    "fp": int(fields["FP"]), "fn": int(fields["FN"]), "idsw": int(fields["IDSW"])}
    return kv


def native_cross_check(dev_raw: str, kitti_root: str, cpp_dir: str) -> dict:
    """(a): the fixed C++ binary's own stdout vs the independent Python E=0 filter on
    the identical eval_kitti dumps (untiled config, matching cpp_dir's config)."""
    out = {}
    n_checked = n_mismatch = 0
    for seq in DEV:
        b = load_bundle(dev_raw, kitti_root, seq)
        cpp = parse_cpp(os.path.join(cpp_dir, f"dev_untiled_{seq}.txt"))
        out[seq] = {}
        for assoc, variant in (("mahalanobis3d", "mahalanobis3d_s0.25_h3"), ("iou2d", "iou2d_s0.25_h3")):
            d = M.derived(M.evaluate_legacy(b.gt, b.tracks(variant), gate=2.0, emit_max_coast=0))
            ref = cpp[assoc]
            ok = (abs(d["mota"] - ref["mota"]) < 5e-5 and abs(d["idf1"] - ref["idf1"]) < 5e-5
                  and d["num_false_positives"] == ref["fp"] and d["num_misses"] == ref["fn"]
                  and d["num_switches"] == ref["idsw"])
            n_checked += 1
            n_mismatch += 0 if ok else 1
            out[seq][assoc] = {"python_e0_filter": {"mota": d["mota"], "idf1": d["idf1"],
                                                     "fp": d["num_false_positives"], "fn": d["num_misses"],
                                                     "idsw": d["num_switches"]},
                               "fixed_cpp_binary": ref, "exact_match": ok}
    return {"n_checked": n_checked, "n_mismatch": n_mismatch, "per_sequence": out}


def three_way(raw: str, kitti_root: str, seqs: list[str], n_boot: int) -> dict:
    """(b): OLD (E=30) vs FIXED (E=0) vs SORT, P-K2D pooled and P-LEGACY, on the frozen dumps."""
    import sort_baseline as S
    bundles = {s: load_bundle(raw, kitti_root, s) for s in seqs}
    k2d_old, k2d_fixed, legacy_old, legacy_fixed, sort_k2d = {}, {}, {}, {}, {}
    for s, b in bundles.items():
        tr = b.tracks("mahalanobis3d_s0.25_h3" if raw.endswith("dev_kitti_tiled") else "A4_plain_mahalanobis")
        k2d_old[s] = M.pool(list(M.evaluate_k2d(b.gt, tr, b.meta, 30).values()))
        k2d_fixed[s] = M.pool(list(M.evaluate_k2d(b.gt, tr, b.meta, 0).values()))
        legacy_old[s] = M.evaluate_legacy(b.gt, tr, gate=2.0, emit_max_coast=30)
        legacy_fixed[s] = M.evaluate_legacy(b.gt, tr, gate=2.0, emit_max_coast=0)
        d = b.dets[b.dets["score"] >= 0.25]
        strk = S.sort_track(d, b.n_frames)
        sort_k2d[s] = M.pool(list(M.evaluate_k2d(b.gt, strk, b.meta, 0).values()))
        # SORT has NO 3D output (X=Y=Z=0 always, see sort_baseline.py) -- P-LEGACY matches
        # by 3D Euclidean distance, which would score every SORT hypothesis as being at the
        # camera origin and never within the 2 m gate of a real object. That is a protocol/
        # system mismatch, not a tracking result, so SORT is deliberately NOT scored under
        # P-LEGACY here; it is scored under P-K2D, which SORT's 2D IoU output does support.

    def summarise(per_seq: dict) -> dict:
        p = M.pool(list(per_seq.values()))
        return {k: p[k] for k in ("mota", "idf1", "precision", "recall", "num_objects",
                                  "num_false_positives", "num_misses", "num_switches")}

    comparisons = []
    for metric in ("mota", "idf1"):
        comparisons.append({"a": "FIXED", "b": "OLD", "protocol": "K2D", **ST.paired_bootstrap(k2d_fixed, k2d_old, metric, n_boot)})
        comparisons.append({"a": "FIXED", "b": "SORT", "protocol": "K2D", **ST.paired_bootstrap(k2d_fixed, sort_k2d, metric, n_boot)})
        comparisons.append({"a": "FIXED", "b": "OLD", "protocol": "LEGACY", **ST.paired_bootstrap(legacy_fixed, legacy_old, metric, n_boot)})
    return {"K2D": {"OLD_E30": summarise(k2d_old), "FIXED_E0": summarise(k2d_fixed), "SORT": summarise(sort_k2d)},
            "LEGACY": {"OLD_E30": summarise(legacy_old), "FIXED_E0": summarise(legacy_fixed),
                      "SORT": "not applicable -- SORT has no 3D output, see the code comment above"},
            "comparisons": comparisons,
            "per_sequence_K2D_mota": {s: {"OLD": k2d_old[s]["mota"], "FIXED": k2d_fixed[s]["mota"], "SORT": sort_k2d[s]["mota"]} for s in seqs}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", default="data/kitti")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default="docs/evaluation/results/phase1_ghost_fix.json")
    a = ap.parse_args()

    native = native_cross_check("out/eval/raw/dev_kitti", a.kitti_root, "out/eval/raw/cpp_benchmark_kitti_fixed")
    print(f"(a) native C++ fix vs independent Python E=0 filter: {native['n_checked']} checked, {native['n_mismatch']} mismatches")

    dev = three_way("out/eval/raw/dev_kitti_tiled", a.kitti_root, DEV, a.n_boot)
    test = three_way("out/eval/raw/test_kitti_tiled", a.kitti_root, TEST, a.n_boot)
    for label, res in (("DEV", dev), ("HELD-OUT", test)):
        print(f"\n=== {label} ===")
        for proto in ("K2D", "LEGACY"):
            r = res[proto]
            sort_str = f"SORT MOTA {r['SORT']['mota']:+.3f} IDF1 {r['SORT']['idf1']:.3f}" if isinstance(r["SORT"], dict) else "SORT n/a (no 3D output)"
            print(f"  {proto}: OLD MOTA {r['OLD_E30']['mota']:+.3f} IDF1 {r['OLD_E30']['idf1']:.3f}  ->  "
                  f"FIXED MOTA {r['FIXED_E0']['mota']:+.3f} IDF1 {r['FIXED_E0']['idf1']:.3f}  |  {sort_str}")
        for c in res["comparisons"]:
            print(f"    {c['a']} - {c['b']:5s} [{c['protocol']}] {c['metric']}: {c['diff']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}] "
                  f"{'EXCLUDES 0' if c['excludes_zero'] else 'inside noise'} wins {c['wins_a']}/{c['n_seq']} vs {c['wins_b']}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"native_cpp_vs_python_cross_check": native, "dev": dev, "held_out": test}, open(a.out, "w"), indent=1, default=float)
    print(f"\nwrote {a.out}")
    return 0 if native["n_mismatch"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
