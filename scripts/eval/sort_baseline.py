"""SORT (Bewley et al., 2016) -- the depth-free 2D baseline B1 of PROTOCOL.md §5.

Written from the published algorithm and its reference implementation's constants
(constant-velocity Kalman filter on [u, v, s, r], IoU association solved with the Hungarian
algorithm, track output only while `time_since_update < 1`). No stereo, no depth: it consumes
only the 2D detections. Made class-aware (a track only associates with detections of its own
class id) so it is comparable to the project's tracker, which does the same.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from geometry import iou_matrix


def _to_z(box: np.ndarray) -> np.ndarray:
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2.0, box[1] + h / 2.0, w * h, w / max(h, 1e-9)])


def _to_box(x: np.ndarray) -> np.ndarray:
    s, r = max(x[2], 1e-9), max(x[3], 1e-9)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0])


class _KalmanBox:
    count = 0

    def __init__(self, box: np.ndarray, cls: int):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01
        self.x = np.zeros(7)
        self.x[:4] = _to_z(box)
        _KalmanBox.count += 1
        self.id = _KalmanBox.count
        self.cls = cls
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def predict(self) -> np.ndarray:
        if self.x[6] + self.x[2] <= 0:
            self.x[6] *= 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _to_box(self.x)

    def update(self, box: np.ndarray) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        y = _to_z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(7) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T   # Joseph form, as filterpy


def sort_track(dets: pd.DataFrame, n_frames: int, max_age: int = 1, min_hits: int = 3,
               iou_threshold: float = 0.3, return_internal: bool = False):
    """Run SORT over one sequence's detections (columns frame, cls, score, x, y, w, h).

    Returns rows in the tracker-dump schema (mot_eval.emission_hypotheses input) with
    confirmed = 1 and tsu = 0, i.e. exactly SORT's own reporting rule; 3D fields are 0.
    With `return_internal`, also returns every live tracker after each frame (frame, id,
    tsu, age), reported or not -- for error analysis only; scoring never uses it."""
    _KalmanBox.count = 0
    trackers: list[_KalmanBox] = []
    by_frame = {int(f): g for f, g in dets.groupby("frame")}
    out = []
    internal = []
    for f in range(n_frames):
        g = by_frame.get(f)
        if g is None:
            boxes, cls = np.zeros((0, 4)), np.zeros(0, int)
        else:
            boxes = np.stack([g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"]], axis=1)
            cls = g["cls"].to_numpy().astype(int)

        preds = np.zeros((len(trackers), 4))
        bad = []
        for t, trk in enumerate(trackers):
            preds[t] = trk.predict()
            if np.isnan(preds[t]).any():
                bad.append(t)
        for t in reversed(bad):
            trackers.pop(t)
        preds = np.delete(preds, bad, axis=0) if bad else preds

        matched, un_d = [], list(range(len(boxes)))
        if len(boxes) and len(trackers):
            iou = iou_matrix(boxes, preds)
            same = cls[:, None] == np.array([t.cls for t in trackers])[None, :]
            iou = np.where(same, iou, 0.0)
            r, c = linear_sum_assignment(-iou)
            matched = [(int(i), int(j)) for i, j in zip(r, c) if iou[i, j] >= iou_threshold]
            un_d = [i for i in range(len(boxes)) if i not in {m[0] for m in matched}]
        for i, j in matched:
            trackers[j].update(boxes[i])
        for i in un_d:
            trackers.append(_KalmanBox(boxes[i], int(cls[i])))

        for trk in reversed(trackers):
            if trk.time_since_update < 1 and (trk.hit_streak >= min_hits or f + 1 <= min_hits):
                b = _to_box(trk.x)
                out.append([f, trk.id, trk.cls, b[0], b[1], b[2] - b[0], b[3] - b[1],
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0, trk.hits, trk.age])
        trackers = [t for t in trackers if t.time_since_update <= max_age]
        if return_internal:
            internal.extend([f, t.id, t.time_since_update, t.age] for t in trackers)
    cols = ["frame", "id", "cls", "x", "y", "w", "h", "X", "Y", "Z", "vx", "vy", "vz",
            "confirmed", "tsu", "hits", "age"]
    if return_internal:
        return (pd.DataFrame(out, columns=cols),
                pd.DataFrame(internal, columns=["frame", "id", "tsu", "age"]))
    return pd.DataFrame(out, columns=cols)
