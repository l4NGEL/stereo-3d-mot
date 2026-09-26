#!/usr/bin/env python3
"""Dev-only selection of the tracker configuration (PROTOCOL.md §4).

Scores every (association x score threshold x min_hits) variant dumped by eval_kitti for the
DEV sequences under every emission policy E in {0, 2, 5, 30} with P-K2D, pools raw counts over
sequences and both classes, and applies the pre-registered rule:

    maximise pooled K2D MOTA; among candidates within 0.005 of the best, prefer the one
    closest to the shipped setting (association mahalanobis3d, E=30, score 0.25, min_hits 3),
    measured as the number of parameters that differ; then lexicographic name.

Nothing here reads any held-out sequence.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import mot_eval as M            # noqa: E402
from bundle import load_bundle  # noqa: E402

EMISSIONS = [0, 2, 5, 30]
SHIPPED = {"assoc": "mahalanobis3d", "E": 30, "score": 0.25, "min_hits": 3}
DEV_SEQS = ["0000", "0003", "0017"]
TIE = 0.005


def parse_variant(name: str) -> dict:
    m = re.fullmatch(r"(\w+?)_s([\d.]+)_h(\d+)", name)
    return {"assoc": m.group(1), "score": float(m.group(2)), "min_hits": int(m.group(3))}


def distance_from_shipped(v: dict) -> int:
    return sum(v[k] != SHIPPED[k] for k in SHIPPED)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--kitti-root", default="data/kitti")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()

    bundles = {s: load_bundle(a.raw, a.kitti_root, s) for s in DEV_SEQS}
    variants = bundles[DEV_SEQS[0]].variants()
    rows = []
    for name in variants:
        v = parse_variant(name)
        tracks = {s: b.tracks(name) for s, b in bundles.items()}
        for E in EMISSIONS:
            counts, per_seq = [], {}
            for s, b in bundles.items():
                res = M.evaluate_k2d(b.gt, tracks[s], b.meta, E)
                counts += list(res.values())
                per_seq[s] = M.pool(list(res.values()))
            pooled = M.pool(counts)
            rows.append({"variant": name, **v, "E": E, "dist_from_shipped": distance_from_shipped({**v, "E": E}),
                         "mota": pooled["mota"], "idf1": pooled["idf1"], "precision": pooled["precision"],
                         "recall": pooled["recall"], "num_objects": pooled["num_objects"],
                         "fp": pooled["num_false_positives"], "fn": pooled["num_misses"],
                         "idsw": pooled["num_switches"],
                         **{f"mota_{s}": per_seq[s]["mota"] for s in DEV_SEQS}})
        print(f"scored {name}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)
    df.to_csv(a.out_csv, index=False, float_format="%.6f")

    best = df["mota"].max()
    cand = df[df["mota"] >= best - TIE].sort_values(["dist_from_shipped", "variant", "E"])
    chosen = cand.iloc[0]
    shipped_rows = df[(df["score"] == 0.25) & (df["min_hits"] == 3) & (df["E"] == 30)]
    out = {
        "rule": "max pooled K2D MOTA on dev; ties within 0.005 -> closest to shipped "
                "(mahalanobis3d, E=30, score 0.25, min_hits 3); then name",
        "best_mota": float(best),
        "n_candidates_within_tie": int(len(cand)),
        "chosen": {k: (chosen[k].item() if hasattr(chosen[k], "item") else chosen[k])
                   for k in ("variant", "assoc", "score", "min_hits", "E", "mota", "idf1",
                             "num_objects", "fp", "fn", "idsw", "dist_from_shipped")},
        "shipped_on_dev": {r["assoc"]: {"mota": float(r["mota"]), "idf1": float(r["idf1"])}
                           for _, r in shipped_rows.iterrows()},
        "top10": df.sort_values("mota", ascending=False).head(10)[["variant", "E", "mota", "idf1"]].to_dict("records"),
    }
    json.dump(out, open(a.out_json, "w"), indent=1)
    print(json.dumps(out["chosen"], indent=1))
    print("best pooled dev MOTA", round(best, 4), "| candidates within tie:", len(cand))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
