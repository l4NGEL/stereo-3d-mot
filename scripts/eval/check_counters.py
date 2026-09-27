#!/usr/bin/env python3
"""TR-6(b): the independent CLEAR-MOT/IDF1 counter (mot_reference.py) must agree EXACTLY with
py-motmetrics on every scored input -- every system, sequence, class group and protocol used for the
reported numbers (dev + held-out). Exits non-zero on any disagreement."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import mot_eval as M            # noqa: E402
import mot_reference as R       # noqa: E402
import protocol as P            # noqa: E402
import sort_baseline as S       # noqa: E402
from bundle import load_bundle  # noqa: E402

KEYS = ("num_objects", "num_matches", "num_switches", "num_misses", "num_false_positives",
        "num_detections", "num_fragmentations", "idtp", "idfp", "idfn")


def agree(inputs) -> bool:
    a, b = M.counts_motmetrics(inputs), R.count(inputs)
    return all(a[k] == b[k] for k in KEYS) and abs(a["dist_sum"] - b["dist_sum"]) < 1e-6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", default="data/kitti")
    a = ap.parse_args()
    sets = [("out/eval/raw/dev_kitti_tiled", ["0000", "0003", "0017"],
             ["iou2d_s0.25_h3", "fused_s0.25_h3", "mahalanobis3d_s0.25_h3"]),
            ("out/eval/raw/test_kitti_tiled", ["0001", "0002", "0004", "0007", "0008", "0010", "0012", "0016"],
             ["F_fused", "A1_no_appearance", "A2_no_iou_cost", "A3_no_3d_cost", "A4_plain_mahalanobis",
              "A5_plain_iou", "A7_greedy"])]
    n = bad = 0
    for raw, seqs, variants in sets:
        for s in seqs:
            b = load_bundle(raw, a.kitti_root, s)
            for v in variants:
                tr = b.tracks(v)
                for E in (0, 30):
                    hyps = M.emission_hypotheses(tr, b.meta, E)
                    for grp in P.GROUPS:
                        n += 1
                        if not agree(M.k2d_inputs(b.gt, hyps, grp)):
                            bad += 1
                            print(f"DISAGREE k2d {raw} {s} {v} E={E} {grp}")
                n += 1
                if not agree(M.legacy_inputs(b.gt, tr)):
                    bad += 1
                    print(f"DISAGREE legacy {raw} {s} {v}")
            tr = S.sort_track(b.dets[b.dets["score"] >= 0.25], b.n_frames)
            hyps = M.emission_hypotheses(tr, b.meta, 0)
            for grp in P.GROUPS:
                n += 1
                if not agree(M.k2d_inputs(b.gt, hyps, grp)):
                    bad += 1
                    print(f"DISAGREE sort {raw} {s} {grp}")
        print(f"{raw}: checked so far {n} inputs, {bad} disagreements", flush=True)
    import json
    os.makedirs("docs/evaluation/results", exist_ok=True)
    json.dump({"compared": n, "disagreements": bad}, open("docs/evaluation/results/counters_check.json", "w"))
    print(f"TR-6(b): {n} scored inputs compared, {bad} disagreements ->", "PASS" if bad == 0 else "FAIL")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
