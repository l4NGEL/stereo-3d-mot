"""Visual-odometry scoring (PROTOCOL.md §3.6), independent of the C++ trajectory_eval.

* KITTI segment metric (Geiger et al. 2012): start frames every `step` frames, segment
  lengths measured along the ground-truth path, translation error / length and rotation
  error / length of the relative pose difference. Re-implemented from the published
  definition of the odometry devkit.
* ATE (rigid Kabsch/Umeyama alignment, optionally with scale) and RPE at several frame
  offsets (Sturm et al. 2012).
Poses are (N, 4, 4) camera-to-world matrices, as in the KITTI poses files.
"""
from __future__ import annotations

import math

import numpy as np

LENGTHS = (100, 200, 300, 400, 500, 600, 700, 800)


def path_lengths(poses: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def _rot_angle(R: np.ndarray) -> float:
    return math.acos(max(-1.0, min(1.0, 0.5 * (np.trace(R) - 1.0))))


def _last_frame(dist: np.ndarray, first: int, length: float) -> int:
    idx = np.nonzero(dist[first:] > dist[first] + length)[0]
    return int(first + idx[0]) if len(idx) else -1


def kitti_segment_errors(gt: np.ndarray, est: np.ndarray, lengths=LENGTHS, step: int = 10) -> list[dict]:
    """One dict per (start frame, length) segment that fits inside the trajectory."""
    n = min(len(gt), len(est))
    gt, est = gt[:n], est[:n]
    dist = path_lengths(gt)
    out = []
    for first in range(0, n, step):
        for length in lengths:
            last = _last_frame(dist, first, length)
            if last == -1:
                continue
            d_gt = np.linalg.inv(gt[first]) @ gt[last]
            d_est = np.linalg.inv(est[first]) @ est[last]
            err = np.linalg.inv(d_est) @ d_gt
            out.append({"first": first, "last": last, "length": length,
                        "t_err": float(np.linalg.norm(err[:3, 3]) / length),      # m per m
                        "r_err": float(_rot_angle(err[:3, :3]) / length)})        # rad per m
    return out


def summarize_segments(segments: list[dict]) -> dict:
    """Per-length means, the devkit-style average (mean of the per-length means) and the plain
    mean over all segments. t in percent, r in deg/m."""
    if not segments:
        return {"n_segments": 0}
    per_len = {}
    for L in sorted({s["length"] for s in segments}):
        sel = [s for s in segments if s["length"] == L]
        per_len[L] = {"n": len(sel), "t_rel_pct": 100.0 * float(np.mean([s["t_err"] for s in sel])),
                      "r_deg_per_m": math.degrees(float(np.mean([s["r_err"] for s in sel])))}
    return {
        "n_segments": len(segments),
        "per_length": per_len,
        "t_rel_pct_devkit": float(np.mean([v["t_rel_pct"] for v in per_len.values()])),
        "r_deg_per_m_devkit": float(np.mean([v["r_deg_per_m"] for v in per_len.values()])),
        "t_rel_pct_pooled": 100.0 * float(np.mean([s["t_err"] for s in segments])),
        "r_deg_per_m_pooled": math.degrees(float(np.mean([s["r_err"] for s in segments]))),
    }


def umeyama(X: np.ndarray, Y: np.ndarray, with_scale: bool = False):
    """Rigid (optionally similarity) transform minimising |Y - (s R X + t)|."""
    n = len(X)
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Yc.T @ Xc / n)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float((D * np.diag(S)).sum() / ((Xc ** 2).sum() / n)) if with_scale else 1.0
    return R, my - s * R @ mx, s


def ate(gt: np.ndarray, est: np.ndarray, with_scale: bool = False) -> dict:
    n = min(len(gt), len(est))
    X, Y = est[:n, :3, 3], gt[:n, :3, 3]
    R, t, s = umeyama(X, Y, with_scale)
    err = np.linalg.norm(Y - (s * (R @ X.T).T + t), axis=1)
    return {"rmse": float(np.sqrt((err ** 2).mean())), "mean": float(err.mean()),
            "max": float(err.max()), "scale": s}


def rpe(gt: np.ndarray, est: np.ndarray, delta: int) -> dict:
    n = min(len(gt), len(est))
    if n <= delta:
        return {"trans_rmse": float("nan"), "rot_rmse_deg": float("nan"), "n_pairs": 0}
    trans, rot = [], []
    for i in range(n - delta):
        d_gt = np.linalg.inv(gt[i]) @ gt[i + delta]
        d_est = np.linalg.inv(est[i]) @ est[i + delta]
        e = np.linalg.inv(d_gt) @ d_est
        trans.append(np.linalg.norm(e[:3, 3]))
        rot.append(_rot_angle(e[:3, :3]))
    return {"trans_rmse": float(np.sqrt(np.mean(np.square(trans)))),
            "rot_rmse_deg": math.degrees(float(np.sqrt(np.mean(np.square(rot))))), "n_pairs": len(trans)}


def sequence_report(gt: np.ndarray, est: np.ndarray, frames_csv=None, deltas=(1, 10, 50),
                    lengths=LENGTHS) -> dict:
    n = min(len(gt), len(est))
    gt, est = gt[:n], est[:n]
    plen = float(path_lengths(gt)[-1])
    a_rigid, a_scale = ate(gt, est, False), ate(gt, est, True)
    rep = {
        "n_frames": n,
        "path_length_m": plen,
        "ate_rigid": a_rigid,
        "ate_rigid_pct_of_path": 100.0 * a_rigid["rmse"] / plen if plen else float("nan"),
        "ate_with_scale": a_scale,
        "rpe": {str(d): rpe(gt, est, d) for d in deltas},
        "segments": summarize_segments(kitti_segment_errors(gt, est, lengths)),
    }
    if frames_csv is not None:
        ok = frames_csv["ok"].to_numpy().astype(bool)
        rep["frames_ok_fraction"] = float(ok[1:].mean()) if len(ok) > 1 else float("nan")
        rep["coasted_frames"] = int((~ok[1:]).sum())
        rep["mean_matches"] = float(frames_csv["matches"][frames_csv["matches"] > 0].mean())
        gt_step = np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1)
        sc = frames_csv["scale"].to_numpy()[1:len(gt)]
        good = ok[1:len(gt)] & (gt_step > 0.05)
        if good.any():
            rel = np.abs(sc[good] - gt_step[good]) / gt_step[good]
            rep["scale_rel_err_median"] = float(np.median(rel))
    return rep
