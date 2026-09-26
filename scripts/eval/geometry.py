"""Box and 3D-cuboid geometry shared by the scorers."""
from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every (a_i, b_j); boxes are (n, 4) [x1, y1, x2, y2]."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


def ioa_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over the area of a_i (the hypothesis), for every (a_i, b_j)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(area_a[:, None] > 0, inter / area_a[:, None], 0.0)


def xywh_to_xyxy(x, y, w, h) -> np.ndarray:
    return np.stack([np.asarray(x, float), np.asarray(y, float),
                     np.asarray(x, float) + np.asarray(w, float),
                     np.asarray(y, float) + np.asarray(h, float)], axis=-1)


def project_point(X, Y, Z, fx, fy, cx, cy):
    return fx * X / Z + cx, fy * Y / Z + cy


def cuboid_front_depth(u: float, v: float, fx: float, fy: float, cx: float, cy: float,
                       h: float, w: float, l: float, X: float, Y: float, Z: float,
                       ry: float) -> float | None:
    """Depth (camera Z) at which the ray through pixel (u, v) first enters the KITTI
    label's 3D cuboid, or None if the ray misses it.

    KITTI cuboid: `location` is the bottom-centre, y points down, the box spans
    x in [-l/2, l/2], y in [-h, 0], z in [-w/2, w/2] in the object frame, which is
    the camera frame rotated by `ry` about Y (R = [[c,0,s],[0,1,0],[-s,0,c]]).
    """
    c, s = np.cos(ry), np.sin(ry)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])   # parametrised so t == depth
    o = R.T @ (-np.array([X, Y, Z]))                          # ray origin, object frame
    d = R.T @ d_cam
    lo = np.array([-l / 2.0, -h, -w / 2.0])
    hi = np.array([l / 2.0, 0.0, w / 2.0])
    t_near, t_far = -np.inf, np.inf
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if o[i] < lo[i] or o[i] > hi[i]:
                return None
            continue
        t1, t2 = (lo[i] - o[i]) / d[i], (hi[i] - o[i]) / d[i]
        t_near = max(t_near, min(t1, t2))
        t_far = min(t_far, max(t1, t2))
    if t_near > t_far or t_far <= 0:
        return None
    return float(t_near) if t_near > 0 else float(t_far)
