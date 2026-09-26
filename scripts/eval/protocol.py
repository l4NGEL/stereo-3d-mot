"""Class mapping, ground-truth categories and hypothesis pre-filtering of
docs/evaluation/PROTOCOL.md §3.1 and §3.3. One place, so tracking and detection
scoring cannot drift apart."""
from __future__ import annotations

import numpy as np
import pandas as pd

from geometry import ioa_matrix, iou_matrix

GROUPS = ("Vehicle", "Person")

VALID_TYPES = {"Vehicle": {"Car", "Van", "Truck"}, "Person": {"Pedestrian"}}
NEUTRAL_TYPES = {"Vehicle": {"Tram", "Misc"},
                 "Person": {"Person_sitting", "Cyclist", "Tram", "Misc"}}

# COCO class id -> evaluation group. bicycle (1) and motorcycle (3) are deliberately
# absent: excluded from both evaluations (PROTOCOL.md §3.1).
COCO_GROUP = {2: "Vehicle", 5: "Vehicle", 7: "Vehicle", 0: "Person"}

# KITTI object-detection benchmark "moderate" difficulty.
MIN_HEIGHT = 25.0
MAX_OCCLUSION = 1
MAX_TRUNCATION = 0.30

MATCH_IOU = 0.5          # hypothesis <-> GT match threshold
DONTCARE_IOA = 0.5       # intersection over hypothesis area


def gt_categories(labels: pd.DataFrame, group: str) -> pd.Series:
    """'valid' | 'ignored' | 'neutral' | 'none' for every label row, for one group."""
    height = labels["y2"] - labels["y1"]
    passes = ((height >= MIN_HEIGHT) & (labels["occluded"] <= MAX_OCCLUSION)
              & (labels["truncated"] <= MAX_TRUNCATION))
    is_type = labels["type"].isin(VALID_TYPES[group])
    cat = pd.Series("none", index=labels.index, dtype=object)
    cat[is_type & passes] = "valid"
    cat[is_type & ~passes] = "ignored"
    cat[labels["type"].isin(NEUTRAL_TYPES[group])] = "neutral"
    return cat


def hyp_group(cls_id: int) -> str | None:
    return COCO_GROUP.get(int(cls_id))


def prefilter_mask(hyp_boxes: np.ndarray, valid_boxes: np.ndarray, ignorable_boxes: np.ndarray,
                   dontcare_boxes: np.ndarray) -> np.ndarray:
    """True for hypotheses that must be DROPPED before matching (§3.3 step 3): the
    hypothesis cannot be a true positive (no valid GT with IoU >= 0.5) and it is
    ignorable (overlaps an ignored/neutral GT with IoU >= 0.5, or a DontCare region
    with IoA >= 0.5, or is shorter than MIN_HEIGHT)."""
    n = len(hyp_boxes)
    if n == 0:
        return np.zeros(0, dtype=bool)
    can_tp = (iou_matrix(hyp_boxes, valid_boxes) >= MATCH_IOU).any(axis=1) \
        if len(valid_boxes) else np.zeros(n, dtype=bool)
    on_ignorable = (iou_matrix(hyp_boxes, ignorable_boxes) >= MATCH_IOU).any(axis=1) \
        if len(ignorable_boxes) else np.zeros(n, dtype=bool)
    in_dontcare = (ioa_matrix(hyp_boxes, dontcare_boxes) >= DONTCARE_IOA).any(axis=1) \
        if len(dontcare_boxes) else np.zeros(n, dtype=bool)
    too_small = (hyp_boxes[:, 3] - hyp_boxes[:, 1]) < MIN_HEIGHT
    return ~can_tp & (on_ignorable | in_dontcare | too_small)

