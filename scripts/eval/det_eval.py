"""Detection scoring (PROTOCOL.md §3.4): KITTI-style greedy matching, precision/recall at
an operating score, and 40-point / all-point AP.

A detection's TP/FP/ignored status depends only on higher-scored detections in its own
frame, so it can be computed once at the score floor and thresholded afterwards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import protocol as P
from geometry import ioa_matrix, iou_matrix
from mot_eval import SequenceGT


def group_dets(dets: pd.DataFrame, group: str) -> pd.DataFrame:
    cls_ids = [c for c, g in P.COCO_GROUP.items() if g == group]
    return dets[dets["cls"].isin(cls_ids)]


def match_frame(det_boxes: np.ndarray, det_scores: np.ndarray, valid_boxes: np.ndarray,
                ign_boxes: np.ndarray, dontcare: np.ndarray, iou_thr: float):
    """Status of each detection in one frame: 'tp' | 'fp' | 'ign', plus the TP's IoU."""
    order = np.argsort(-det_scores, kind="stable")
    status = np.empty(len(det_boxes), dtype=object)
    tp_iou = np.zeros(len(det_boxes))
    matched = np.zeros(len(valid_boxes), dtype=bool)
    iou_valid = iou_matrix(det_boxes, valid_boxes)
    iou_ign = iou_matrix(det_boxes, ign_boxes)
    in_dc = (ioa_matrix(det_boxes, dontcare) >= P.DONTCARE_IOA).any(axis=1) \
        if len(dontcare) and len(det_boxes) else np.zeros(len(det_boxes), dtype=bool)
    too_small = (det_boxes[:, 3] - det_boxes[:, 1]) < P.MIN_HEIGHT if len(det_boxes) \
        else np.zeros(0, dtype=bool)
    for i in order:
        cand = np.where(matched, -1.0, iou_valid[i]) if len(valid_boxes) else np.zeros(0)
        j = int(np.argmax(cand)) if len(cand) else -1
        if j >= 0 and cand[j] >= iou_thr:
            matched[j] = True
            status[i] = "tp"
            tp_iou[i] = cand[j]
        elif len(ign_boxes) and iou_ign[i].max() >= P.MATCH_IOU:
            status[i] = "ign"
        elif in_dc[i] or too_small[i]:
            status[i] = "ign"
        else:
            status[i] = "fp"
    return status, tp_iou


def score_sequence(seq: SequenceGT, dets: pd.DataFrame, group: str, iou_thr: float) -> pd.DataFrame:
    """One row per detection: frame, score, status, iou (TP only). Also `n_gt` in attrs."""
    gd = group_dets(dets, group)
    rows = []
    for f, g in gd.groupby("frame"):
        gt = seq.frame(group, int(f))
        boxes = g[["x", "y", "x2", "y2"]].to_numpy()
        status, tp_iou = match_frame(boxes, g["score"].to_numpy(), gt["valid_boxes"],
                                     gt["ign_boxes"], seq.dontcare.get(int(f), np.zeros((0, 4))),
                                     iou_thr)
        rows.append(pd.DataFrame({"frame": int(f), "score": g["score"].to_numpy(),
                                  "status": status, "iou": tp_iou}))
    out = pd.concat(rows, ignore_index=True) if rows else \
        pd.DataFrame(columns=["frame", "score", "status", "iou"])
    out.attrs["n_gt"] = int(sum(len(seq.frame(group, f)["valid_ids"]) for f in range(seq.n_frames)))
    return out


def pr_curve(result: pd.DataFrame, n_gt: int):
    """Precision/recall arrays over detections sorted by descending score (ignored dropped)."""
    r = result[result["status"] != "ign"].sort_values("score", ascending=False, kind="stable")
    tp = (r["status"] == "tp").to_numpy().astype(float)
    fp = 1.0 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    rec = ctp / max(n_gt, 1)
    return prec, rec, r["score"].to_numpy()


def ap_r40(prec: np.ndarray, rec: np.ndarray) -> float:
    """KITTI R40: mean interpolated precision at recall 1/40 ... 1."""
    if len(prec) == 0:
        return 0.0
    interp = np.maximum.accumulate(prec[::-1])[::-1]
    total = 0.0
    for k in range(1, 41):
        idx = np.searchsorted(rec, k / 40.0, side="left")
        total += interp[idx] if idx < len(interp) else 0.0
    return total / 40.0


def ap_all_points(prec: np.ndarray, rec: np.ndarray) -> float:
    """Area under the monotone (interpolated) PR curve."""
    if len(prec) == 0:
        return 0.0
    mrec = np.concatenate([[0.0], rec, [rec[-1]]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def operating_point(result: pd.DataFrame, n_gt: int, score_thr: float) -> dict:
    r = result[(result["status"] != "ign") & (result["score"] >= score_thr)]
    tp = int((r["status"] == "tp").sum())
    fp = int((r["status"] == "fp").sum())
    fn = max(n_gt - tp, 0)
    p = tp / (tp + fp) if tp + fp else float("nan")
    rc = tp / n_gt if n_gt else float("nan")
    f1 = 2 * p * rc / (p + rc) if p == p and rc == rc and (p + rc) > 0 else float("nan")
    mean_iou = float(r.loc[r["status"] == "tp", "iou"].mean()) if tp else float("nan")
    return {"score_thr": score_thr, "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": rc,
            "f1": f1, "mean_tp_iou": mean_iou, "n_gt": n_gt}


def pooled(results: list[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    n_gt = sum(r.attrs.get("n_gt", 0) for r in results)
    frames = []
    for i, r in enumerate(results):
        frames.append(r.assign(seq=i))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), n_gt
