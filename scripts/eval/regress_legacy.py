#!/usr/bin/env python3
"""TR-6: does the independent scorer reproduce the published legacy numbers?

Compares, per dev sequence and association method (score 0.25, min_hits 3, every confirmed
track -- the as-shipped legacy setting): (a) py-motmetrics on the eval_kitti dumps,
(b) the independent counter mot_reference.py on the same inputs, (c) the numbers printed
by the C++ benchmark_kitti (re-run for this check, if its output files are present) and
(d) the values published in README.md / docs/roadmap.md.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import mot_eval as M            # noqa: E402
import mot_reference as R       # noqa: E402
from bundle import load_bundle  # noqa: E402

ASSOC = {"3D Mahalanobis": "mahalanobis3d", "2D IoU": "iou2d", "3D+IoU+ReID": "fused"}

PUBLISHED = {
    "0000": {"mahalanobis3d": dict(mota=-2.304, motp=0.995, idf1=0.182, idsw=33, frag=11, fp=2195, fn=121, prec=0.212, rec=0.830),
             "iou2d": dict(mota=-2.135, motp=1.020, idf1=0.187, idsw=35, frag=23, fp=1980, fn=214, prec=0.201, rec=0.699),
             "fused": dict(idf1=0.203, idsw=27, frag=13)},
    "0003": {"mahalanobis3d": dict(idf1=0.223, idsw=0, frag=2), "iou2d": dict(idf1=0.301, idsw=2, frag=3),
             "fused": dict(idf1=0.283, idsw=0, frag=2)},
    "0017": {"mahalanobis3d": dict(idf1=0.486, idsw=5, frag=1), "iou2d": dict(idf1=0.386, idsw=13, frag=6),
             "fused": dict(idf1=0.462, idsw=8, frag=2)},
}
KEYMAP = {"mota": "mota", "motp": "motp", "idf1": "idf1", "prec": "precision", "rec": "recall",
          "idsw": "num_switches", "frag": "num_fragmentations", "fp": "num_false_positives",
          "fn": "num_misses"}


def parse_cpp(path: str) -> dict:
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        for label, assoc in ASSOC.items():
            if line.startswith(label):
                kv = dict(re.findall(r"(\w+)=([-\d.e+]+)", line))
                out[assoc] = {"mota": float(kv["MOTA"]), "motp": float(kv["MOTP"]), "idf1": float(kv["IDF1"]),
                              "num_switches": int(kv["IDSW"]), "num_fragmentations": int(kv["Frag"]),
                              "num_false_positives": int(kv["FP"]), "num_misses": int(kv["FN"]),
                              "precision": float(kv["Prec"]), "recall": float(kv["Rec"])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="eval_kitti output root, e.g. out/eval/raw/dev_kitti")
    ap.add_argument("--kitti-root", default="data/kitti")
    ap.add_argument("--cpp-dir", default=None, help="dir with benchmark_kitti stdout per sequence: <seq>.txt")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    report, failures = {}, 0
    for seq in ("0000", "0003", "0017"):
        b = load_bundle(a.raw, a.kitti_root, seq)
        cpp = parse_cpp(os.path.join(a.cpp_dir, f"{seq}.txt")) if a.cpp_dir else {}
        report[seq] = {}
        for assoc in ("mahalanobis3d", "iou2d", "fused"):
            tracks = b.tracks(f"{assoc}_s0.25_h3")
            inputs = M.legacy_inputs(b.gt, tracks)
            mm_counts = M.counts_motmetrics(inputs)
            ref_counts = R.count(inputs)
            d = M.derived(mm_counts)
            agree = all(mm_counts[k] == ref_counts[k] for k in
                        ("num_objects", "num_switches", "num_misses", "num_false_positives",
                         "num_fragmentations", "idtp", "idfp", "idfn"))
            row = {"motmetrics": {k: d[k] for k in ("mota", "motp", "idf1", "precision", "recall", "num_objects",
                                                    "num_switches", "num_fragmentations", "num_false_positives", "num_misses")},
                   "reference_counter_agrees": agree, "cpp": cpp.get(assoc), "published": PUBLISHED[seq][assoc]}
            checks = []
            for pk, val in PUBLISHED[seq][assoc].items():
                mine = d[KEYMAP[pk]]
                tol = 0.0011 if pk in ("mota", "motp", "idf1", "prec", "rec") else 0.5
                checks.append((pk, val, mine, abs(mine - val) <= tol))
            if cpp.get(assoc):
                for k, v in cpp[assoc].items():
                    tol = 0.0011 if isinstance(v, float) else 0.5
                    checks.append((f"cpp:{k}", v, d[k], abs(d[k] - v) <= tol))
            row["checks"] = [{"key": k, "expected": e, "reproduced": m, "ok": bool(ok)} for k, e, m, ok in checks]
            failures += sum(0 if ok else 1 for *_, ok in checks) + (0 if agree else 1)
            report[seq][assoc] = row
            bad = [c for c in row["checks"] if not c["ok"]]
            print(f"{seq} {assoc:14s} MOTA={d['mota']:+.3f} MOTP={d['motp']:.3f} IDF1={d['idf1']:.3f} "
                  f"IDSW={d['num_switches']} Frag={d['num_fragmentations']} FP={d['num_false_positives']} "
                  f"FN={d['num_misses']} | ref-counter {'OK' if agree else 'DISAGREES'} | "
                  f"{len(row['checks']) - len(bad)}/{len(row['checks'])} checks ok"
                  + (f"  MISMATCH: {bad}" if bad else ""))
    print("TR-6 legacy regression:", "PASS" if failures == 0 else f"FAIL ({failures} mismatches)")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump({"failures": failures, "report": report}, open(a.out, "w"), indent=1)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
