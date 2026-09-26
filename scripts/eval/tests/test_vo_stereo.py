import math

import numpy as np

import stereo_eval as SE
import vo_eval as V


def _straight(n=200, step=1.0):
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 2, 3] = np.arange(n) * step
    return poses


def test_identical_trajectories_have_zero_error_everywhere():
    gt = _straight()
    r = V.sequence_report(gt, gt.copy())
    assert r["ate_rigid"]["rmse"] < 1e-9
    assert r["rpe"]["1"]["trans_rmse"] < 1e-9
    assert r["segments"]["t_rel_pct_devkit"] < 1e-9 and r["segments"]["r_deg_per_m_devkit"] < 1e-9
    assert r["segments"]["n_segments"] > 0


def test_a_constant_scale_error_is_reported_as_that_percentage():
    gt = _straight(300)
    est = gt.copy()
    est[:, 2, 3] *= 1.05                      # 5 % too long
    seg = V.summarize_segments(V.kitti_segment_errors(gt, est))
    assert abs(seg["t_rel_pct_devkit"] - 5.0) < 0.25
    # a rigid alignment cannot remove a scale error, a similarity alignment can
    assert V.ate(gt, est, with_scale=False)["rmse"] > 1.0
    assert V.ate(gt, est, with_scale=True)["rmse"] < 1e-6


def test_rotation_drift_is_measured_in_degrees_per_metre():
    gt = _straight(300)
    est = gt.copy()
    for i in range(len(est)):                 # yaw drifts 0.01 deg per frame (= per metre)
        a = math.radians(0.01 * i)
        R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        est[i, :3, :3] = R
    seg = V.summarize_segments(V.kitti_segment_errors(gt, est))
    assert abs(seg["r_deg_per_m_devkit"] - 0.01) < 0.002


def test_umeyama_recovers_a_rigid_transform():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    a = 0.7
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Y = X @ R.T + np.array([3.0, -2.0, 1.0])
    R2, t2, s = V.umeyama(X, Y)
    assert np.allclose(R2, R, atol=1e-9) and np.allclose(t2, [3, -2, 1], atol=1e-9) and s == 1.0


def test_d1_counts_holes_as_errors_and_needs_both_thresholds():
    gt = np.full((10, 10), 20.0, dtype=np.float32)
    est = gt.copy()
    est[0, :] = -1.0                          # 10 holes
    est[1, :] = 23.5                          # 3.5 px and 17 % off -> D1 error
    est[2, :] = 21.0                          # 1 px off -> fine
    r = SE.d1(est, gt)
    assert abs(r["d1_all_holes_as_error"] - 0.2) < 1e-9
    assert abs(r["d1_valid_only"] - 10 / 90) < 1e-9
    assert abs(r["density"] - 0.9) < 1e-9
    big = np.full((10, 10), 100.0, dtype=np.float32)
    e2 = big + 4.0                            # 4 px off but only 4 % of 100 -> not a D1 error
    assert SE.d1(e2, big)["d1_all_holes_as_error"] == 0.0


def test_project_style_matches_its_definition():
    gt = np.full((4, 200), 10.0, dtype=np.float32)
    est = gt.copy()
    est[:, 150:] = -1.0
    r = SE.project_style(est, gt, crop=128)   # ROI = columns 128..199, half without estimate
    assert abs(r["density"] - (150 - 128) / (200 - 128)) < 1e-9 and r["rmse"] == 0.0
    f = SE.full_image(est, gt)
    assert abs(f["coverage"] - 150 / 200) < 1e-9
