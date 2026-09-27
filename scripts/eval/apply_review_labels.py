#!/usr/bin/env python3
"""Apply the manual review labels (single reviewer: the author, an AI model, looking at the contact
sheets) to the review tables, summarise them, and export the images as JPEG into docs/.

The label tables below ARE the review: each entry was assigned by looking at the numbered crop
(and, for the rule-selected frames, the annotated full frame) under the category definitions of
docs/evaluation/PROTOCOL.md §6. Categories for the residual false positives:
  A visible, unlabelled real object of the claimed class   B true false alarm (no such object /
  duplicate)   C label-convention or class-mapping issue (a real, labelled object the protocol does
  not credit)   D ambiguous.   Audit of the automatic categories: Y real object of the claimed class,
  N not, U unclear.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import stats as ST  # noqa: E402

# --- rule-selected frames: only the `residual` items need a manual label (image, item) -> (label, note)
RULE_SELECTED = {
    ("0000_000153.png", 1): ("A", "parked car in a row, no label"),
    ("0000_000153.png", 2): ("A", "parked car in a row, no label"),
    ("0000_000153.png", 3): ("D", "narrow sliver between two parked cars"),
    ("0000_000147.png", 1): ("A", "silver parked car, no label"),
    ("0000_000151.png", 1): ("C", "real pedestrian labelled Ped12; detection IoU 0.37 with the label box"),
    ("0000_000151.png", 3): ("A", "parked car in a row, no label"),
    ("0000_000151.png", 5): ("A", "silver parked car, no label"),
    ("0000_000151.png", 6): ("D", "cars and a cyclist overlapping, partial box"),
    ("0000_000148.png", 3): ("A", "parked car in a row, no label"),
    ("0000_000141.png", 3): ("D", "narrow partial car"),
    ("0000_000145.png", 4): ("D", "narrow partial car next to a person"),
    ("0000_000146.png", 4): ("D", "upper body of a cyclist / cars, unclear"),
}
for f in range(2, 10):        # 0017 frames 2-9: the same parked blue truck at an identical box
    RULE_SELECTED[(f"0017_{f:06d}.png", 1)] = ("A", "blue parked truck, no label (same box as frame 2, which was inspected)")

TRUCK = "blue parked truck (Buechele), unlabelled, static, 0017"
RESIDUAL_SAMPLE = {
    1: "BBBBBCDCCDAB", 2: "DAAAADDAAAAA", 3: "AAAAAAAAAAAA", 4: "AAADAAAAAAAA", 5: "AAAAAAAAAAAA"}
RESIDUAL_NOTES = {
    (1, 1): "covered scooter/bike, not a car", (1, 2): "covered scooter/bike, not a car",
    (1, 3): "market stall / bench", (1, 4): "scooter + bicycle", (1, 5): "scooter + bicycle",
    (1, 6): "rider on bicycle: KITTI labels rider+bike as Cyclist", (1, 8): "rider on bicycle (Cyclist)",
    (1, 9): "rider on bicycle (Cyclist)", (1, 7): "small figure at a kiosk", (1, 10): "small figure at a kiosk",
    (1, 11): "white van partly hidden by another car", (1, 12): "market stall with blue canopy",
    (2, 1): "figure at a pole / kiosk, unclear", (2, 6): "narrow partial car", (2, 7): "cars and a person mixed",
    (3, 7): "walking pedestrian, no matching label", (4, 1): "person cut by the image border, no label",
    (4, 4): "dark figures among bicycles, unclear"}
AUDIT = {1: "YYYYUYYYYYUY", 2: "YNYYYYYYYYYY", 3: "YYUYYYYYYYYY", 4: "YYUY"}
AUDIT_NOTES = {(2, 2): "two bicycles labelled motorcycle", (1, 5): "blurred", (1, 11): "blurred, dark car + person",
               (3, 3): "figure inside a kiosk", (4, 3): "tiny saturated white blob"}


# --- held-out sample (dev-independent), same definitions; legacy FPs of the as-shipped 3D Mahalanobis system
RESIDUAL_SAMPLE_TEST = {1: "DAAAABBBBABA", 2: "AABBBBBBBBBB", 3: "BBBBCCAAAAAC"}
RESIDUAL_NOTES_TEST = {
    (1, 1): "dark hood/roof very close to the camera, unclear", (1, 6): "glass bus shelter, not a bus",
    (1, 7): "round traffic sign on a pole", (1, 8): "round traffic sign on a pole", (1, 9): "round traffic sign on a pole",
    (1, 11): "round traffic sign on a pole", (1, 10): "moving white van", (1, 2): "parked cars", (1, 3): "parked white van",
    (3, 5): "person on a bicycle (KITTI Cyclist = rider + bike)", (3, 6): "cyclist, close", (3, 12): "cyclist at distance",
    (3, 7): "blurred dark car", (3, 8): "blurred dark car", (3, 9): "blurred dark car", (3, 10): "blurred dark car"}
for _sh, _rng in ((2, range(3, 13)), (3, range(1, 5))):
    for _i in _rng:
        RESIDUAL_NOTES_TEST[(_sh, _i)] = "red-white road delineator post with an orange lamp, detected as person"
AUDIT_TEST = {1: "UYYYYYYYYYYY", 2: "YUYY"}
AUDIT_NOTES_TEST = {(1, 1): "thin figure by a tree, unclear", (2, 2): "tiny blurred cars"}


def wilson_row(k, n):
    lo, hi = ST.wilson(k, n)
    return {"k": k, "n": n, "share": k / n if n else float("nan"), "ci95": [lo, hi]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-dev", default="out/eval/review_dev")
    ap.add_argument("--review-sample", default="out/eval/review_dev_sample")
    ap.add_argument("--review-test-sample", default="out/eval/review_test_sample")
    ap.add_argument("--out", default="docs/evaluation/review")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- rule-selected frames
    rs = pd.read_csv(os.path.join(a.review_dev, "review_items.csv"))
    labels, notes = [], []
    for _, r in rs.iterrows():
        lab, note = RULE_SELECTED.get((r["image"], int(r["item"])), ("", ""))
        labels.append(lab)
        notes.append(note)
    rs["label"], rs["note"] = labels, notes
    assert (rs.loc[rs.auto_category == "residual", "label"] != "").all(), "unlabelled residual item"
    rs.to_csv(os.path.join(a.out, "rule_selected_items.csv"), index=False)
    pd.read_csv(os.path.join(a.review_dev, "review_frames.csv")).to_csv(os.path.join(a.out, "rule_selected_frames.csv"), index=False)

    # ---- random residual sample
    rsm = pd.read_csv(os.path.join(a.review_sample, "residual_items.csv"))
    rsm["label"] = [RESIDUAL_SAMPLE[int(s)][int(i) - 1] for s, i in zip(rsm["sheet"], rsm["item"])]
    rsm["note"] = [TRUCK if (int(s) >= 3 and int(i) not in (7,) and c == 7) or (int(s) == 2 and int(i) == 12)
                   else RESIDUAL_NOTES.get((int(s), int(i)), "")
                   for s, i, c in zip(rsm["sheet"], rsm["item"], rsm["cls"])]
    rsm["is_truck"] = (rsm["cls"] == 7) & (rsm["w"] > 250)
    rsm.to_csv(os.path.join(a.out, "residual_sample_items.csv"), index=False)

    # ---- audit
    au = pd.read_csv(os.path.join(a.review_sample, "audit_items.csv"))
    au["label"] = [AUDIT[int(s)][int(i) - 1] for s, i in zip(au["sheet"], au["item"])]
    au["note"] = [AUDIT_NOTES.get((int(s), int(i)), "") for s, i in zip(au["sheet"], au["item"])]
    au.to_csv(os.path.join(a.out, "audit_items.csv"), index=False)

    # ---- held-out sample
    rt = pd.read_csv(os.path.join(a.review_test_sample, "residual_items.csv"))
    rt["label"] = [RESIDUAL_SAMPLE_TEST[int(sh)][int(i) - 1] for sh, i in zip(rt["sheet"], rt["item"])]
    rt["note"] = [RESIDUAL_NOTES_TEST.get((int(sh), int(i)), "") for sh, i in zip(rt["sheet"], rt["item"])]
    rt.to_csv(os.path.join(a.out, "residual_sample_heldout_items.csv"), index=False)
    at = pd.read_csv(os.path.join(a.review_test_sample, "audit_items.csv"))
    at["label"] = [AUDIT_TEST[int(sh)][int(i) - 1] for sh, i in zip(at["sheet"], at["item"])]
    at["note"] = [AUDIT_NOTES_TEST.get((int(sh), int(i)), "") for sh, i in zip(at["sheet"], at["item"])]
    at.to_csv(os.path.join(a.out, "audit_heldout_items.csv"), index=False)

    # ---- summaries
    def dist(df, col="label", cats="ABCD"):
        n = len(df)
        return {c: wilson_row(int((df[col] == c).sum()), n) for c in cats}

    summary = {
        "reviewer": "single reviewer (the author, an AI model); no second annotator",
        "rule_selected": {"frames": int(pd.read_csv(os.path.join(a.review_dev, "review_frames.csv")).shape[0]),
                          "items": int(len(rs)), "auto_category_counts": rs["auto_category"].value_counts().to_dict(),
                          "residual_manual": dist(rs[rs.auto_category == "residual"]),
                          "residual_distinct_objects_note": "8 of the 20 residual items are the same parked truck in consecutive frames"},
        "residual_random_sample": {"n": int(len(rsm)), "item_weighted": dist(rsm),
                                   "excluding_the_static_truck": dist(rsm[~rsm.is_truck]),
                                   "n_truck_items": int(rsm.is_truck.sum())},
        "residual_random_sample_heldout": {"n": int(len(rt)), "item_weighted": dist(rt),
                                           "seed": 20260928, "system": "A4_plain_mahalanobis, legacy, all confirmed tracks"},
        "audit_heldout": {"n": int(len(at)), "overall": dist(at, cats="YNU"),
                          "by_category": {c: dist(at[at.auto_category == c], cats="YNU") for c in sorted(at.auto_category.unique())}},
        "audit_of_automatic_categories": {
            "overall": dist(au, cats="YNU"),
            "by_category": {c: dist(au[au.auto_category == c], cats="YNU") for c in sorted(au.auto_category.unique())}},
    }
    json.dump(summary, open(os.path.join(a.out, "review_summary.json"), "w"), indent=1)

    # ---- images: PNG -> JPEG (the review is auditable, the repo stays small)
    n_img = 0
    for src_dir, prefix in ((a.review_dev, "frame_"), (a.review_sample, "sheet_"), (a.review_test_sample, "heldout_sheet_")):
        for name in sorted(os.listdir(src_dir)):
            if name.endswith(".png"):
                img = cv2.imread(os.path.join(src_dir, name))
                cv2.imwrite(os.path.join(a.out, prefix + name[:-4] + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                n_img += 1
    print(json.dumps({k: summary[k] for k in ("rule_selected", "residual_random_sample")}, indent=1)[:3000])
    print("audit:", json.dumps(summary["audit_of_automatic_categories"]["overall"]))
    for c, d in summary["audit_of_automatic_categories"]["by_category"].items():
        print(f"  {c:40s}", {k: v['k'] for k, v in d.items()})
    print(f"{n_img} images exported to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
