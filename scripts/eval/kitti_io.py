"""Readers for KITTI files and for the dumps written by apps/eval_kitti,
apps/dump_disparity and apps/benchmark_vo (docs/evaluation/PROTOCOL.md).

Nothing here depends on the C++ code: it parses text/PFM/PNG files, so the
scorer stays independent of the pipeline it is judging.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

LABEL_COLUMNS = ["frame", "track_id", "type", "truncated", "occluded", "alpha",
                 "x1", "y1", "x2", "y2", "h", "w", "l", "X", "Y", "Z", "ry"]
DET_COLUMNS = ["frame", "cls", "score", "x", "y", "w", "h", "valid", "X", "Y", "Z", "depth"]
TRACK_COLUMNS = ["frame", "id", "cls", "x", "y", "w", "h", "X", "Y", "Z", "vx", "vy", "vz",
                 "confirmed", "tsu", "hits", "age"]


def read_labels(path: str) -> pd.DataFrame:
    """KITTI tracking label_02/NNNN.txt (17 fields per line; DontCare rows kept)."""
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 17:
                continue
            rows.append([int(p[0]), int(p[1]), p[2], float(p[3]), int(float(p[4])), float(p[5]),
                         *[float(v) for v in p[6:10]], *[float(v) for v in p[10:17]]])
    return pd.DataFrame(rows, columns=LABEL_COLUMNS)


def _read_table(path: str, columns: list[str]) -> pd.DataFrame:
    if os.path.getsize(path) == 0:
        return pd.DataFrame(columns=columns)
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.size == 0:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(data, columns=columns)
    for c in ("frame", "cls", "valid", "id", "confirmed", "tsu", "hits", "age"):
        if c in df.columns:
            df[c] = df[c].astype(int)
    return df


def read_dets(path: str) -> pd.DataFrame:
    df = _read_table(path, DET_COLUMNS)
    df["x2"] = df["x"] + df["w"]
    df["y2"] = df["y"] + df["h"]
    return df


def read_tracks(path: str) -> pd.DataFrame:
    return _read_table(path, TRACK_COLUMNS)


def read_meta(path: str) -> dict:
    meta: dict = {}
    with open(path) as f:
        for line in f:
            if "=" not in line:
                continue
            k, v = line.rstrip("\n").split("=", 1)
            try:
                meta[k] = float(v) if any(c in v for c in ".e") else int(v)
            except ValueError:
                meta[k] = v
    return meta


def read_timing(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def read_pfm(path: str) -> np.ndarray:
    """Little/big-endian PFM -> float32 array, top row first (PFM stores rows bottom-up)."""
    with open(path, "rb") as f:
        header = f.readline().decode().strip()
        if header not in ("Pf", "PF"):
            raise ValueError(f"{path}: not a PFM ({header!r})")
        w, h = (int(v) for v in f.readline().decode().split())
        scale = float(f.readline().decode().strip())
        channels = 3 if header == "PF" else 1
        data = np.frombuffer(f.read(), dtype=("<f4" if scale < 0 else ">f4"),
                             count=w * h * channels)
    img = data.reshape((h, w, channels) if channels == 3 else (h, w))
    return np.flipud(img).astype(np.float32).copy()


def read_kitti_poses(path: str) -> np.ndarray:
    """KITTI odometry poses file -> (N, 4, 4) camera-to-world matrices."""
    raw = np.loadtxt(path, ndmin=2)
    poses = np.tile(np.eye(4), (raw.shape[0], 1, 1))
    poses[:, :3, :4] = raw.reshape(-1, 3, 4)
    return poses


def read_calib_cam_to_cam(path: str) -> dict:
    """KITTI stereo-2015 calib_cam_to_cam/NNNNNN.txt -> fx, cx, cy, baseline."""
    vals: dict[str, np.ndarray] = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            try:
                vals[k.strip()] = np.array([float(t) for t in v.split()])
            except ValueError:
                continue
    p2, p3 = vals["P_rect_02"], vals["P_rect_03"]
    return {"fx": float(p2[0]), "cx": float(p2[2]), "cy": float(p2[6]),
            "baseline": float((p2[3] - p3[3]) / p2[0])}


def sequence_frame_count(kitti_root: str, sequence: str) -> int:
    d = os.path.join(kitti_root, "image_02", sequence)
    return len([n for n in os.listdir(d) if n.endswith(".png")])
