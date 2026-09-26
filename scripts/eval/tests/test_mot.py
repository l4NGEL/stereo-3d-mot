"""Tests for the tracking scorers: the independent counter must agree with
py-motmetrics exactly, and the K2D protocol rules must do what PROTOCOL.md §3.3 says."""
import numpy as np
import pandas as pd

import mot_eval as M
import mot_reference as R
import protocol as P
from geometry import cuboid_front_depth, iou_matrix, ioa_matrix

KEYS = ["num_objects", "num_matches", "num_switches", "num_misses", "num_false_positives",
        "num_detections", "num_fragmentations", "idtp", "idfp", "idfn"]


def _random_inputs(rng, frames=40, n_gt=6, n_hyp=8):
    inputs = []
    for _ in range(frames):
        ng = rng.integers(0, n_gt + 1)
        nh = rng.integers(0, n_hyp + 1)
        gt_ids = rng.choice(np.arange(1, 9), size=min(ng, 8), replace=False)
        hyp_ids = rng.choice(np.arange(1, 12), size=min(nh, 11), replace=False)
        d = rng.random((len(gt_ids), len(hyp_ids)))
        d[rng.random(d.shape) < 0.5] = np.nan
        inputs.append((gt_ids, hyp_ids, d))
    return inputs


def test_reference_counter_agrees_with_motmetrics_on_random_data():
    rng = np.random.default_rng(0)
    for trial in range(60):
        inputs = _random_inputs(rng)
        a = M.counts_motmetrics(inputs)
        b = R.count(inputs)
        for k in KEYS:
            assert a[k] == b[k], (trial, k, a[k], b[k])
        assert abs(a["dist_sum"] - b["dist_sum"]) < 1e-9


def _labels(rows):
    cols = ["frame", "track_id", "type", "truncated", "occluded", "alpha", "x1", "y1", "x2", "y2",
            "h", "w", "l", "X", "Y", "Z", "ry"]
    return pd.DataFrame(rows, columns=cols)


def _row(frame, tid, typ, box, trunc=0.0, occ=0):
    return [frame, tid, typ, trunc, occ, 0.0, *box, 1.5, 1.7, 4.0, 0.0, 1.5, 20.0, 0.0]


def _tracks(rows):
    cols = ["frame", "id", "cls", "x", "y", "w", "h", "X", "Y", "Z", "vx", "vy", "vz",
            "confirmed", "tsu", "hits", "age"]
    return pd.DataFrame(rows, columns=cols)


def _trk(frame, tid, cls, box, tsu=0, Z=20.0):
    x1, y1, x2, y2 = box
    return [frame, tid, cls, x1, y1, x2 - x1, y2 - y1, 0.0, 1.0, Z, 0, 0, 0, 1, tsu, 5, 5]


META = {"fx": 700.0, "fy": 700.0, "cx": 600.0, "cy": 180.0}


def test_k2d_rules_ignore_what_the_protocol_says_to_ignore():
    car = (100, 100, 200, 160)                       # evaluable car, 60 px high
    occluded_car = (300, 100, 400, 160)              # Car with occluded=2 -> ignored GT
    tram = (500, 100, 600, 160)                      # neutral
    dontcare = (700, 100, 800, 200)
    labels = _labels([
        _row(0, 1, "Car", car),
        _row(0, 2, "Car", occluded_car, occ=2),
        _row(0, 3, "Tram", tram),
        _row(0, -1, "DontCare", dontcare),
    ])
    tracks = _tracks([
        _trk(0, 10, 2, car),                                    # TP
        _trk(0, 11, 2, occluded_car),                           # on ignored GT -> dropped
        _trk(0, 12, 2, tram),                                   # on neutral GT -> dropped
        _trk(0, 13, 2, (710, 110, 790, 190)),                   # inside DontCare -> dropped
        _trk(0, 14, 2, (900, 100, 950, 110)),                   # 10 px high -> dropped
        _trk(0, 15, 2, (900, 200, 1000, 260)),                  # genuine FP
        _trk(0, 16, 1, (1000, 100, 1100, 160)),                 # bicycle -> excluded group
    ])
    seq = M.SequenceGT(labels, 1)
    res = M.evaluate_k2d(seq, tracks, META, emit_max_coast=0)
    v = res["Vehicle"]
    assert v["num_objects"] == 1          # only the evaluable car is in the denominator
    assert v["num_detections"] == 1
    assert v["num_false_positives"] == 1  # exactly the genuine FP
    assert v["num_misses"] == 0
    assert res["Person"]["num_objects"] == 0 and res["Person"]["num_false_positives"] == 0


def test_moderate_filter_categories():
    lab = _labels([
        _row(0, 1, "Car", (0, 0, 50, 30)),                  # 30 px -> valid
        _row(0, 2, "Car", (0, 0, 50, 24)),                  # 24 px -> ignored
        _row(0, 3, "Car", (0, 0, 50, 40), occ=2),           # occluded 2 -> ignored
        _row(0, 4, "Car", (0, 0, 50, 40), trunc=0.4),       # truncated .4 -> ignored
        _row(0, 5, "Van", (0, 0, 50, 40)),                  # van -> valid vehicle
        _row(0, 6, "Cyclist", (0, 0, 50, 40)),              # cyclist -> none for vehicle
        _row(0, 7, "Tram", (0, 0, 50, 40)),                 # neutral
    ])
    assert list(P.gt_categories(lab, "Vehicle")) == \
        ["valid", "ignored", "ignored", "ignored", "valid", "none", "neutral"]
    assert list(P.gt_categories(lab, "Person")) == \
        ["none", "none", "none", "none", "none", "neutral", "neutral"]


def test_coasting_box_is_projected_and_scaled():
    tracks = _tracks([
        _trk(0, 1, 2, (100, 100, 200, 200), tsu=0, Z=10.0),
        # coasting one frame later, now farther away (Z=20): half the size, at the projection
        [1, 1, 2, 100, 100, 100, 100, 0.0, 0.0, 20.0, 0, 0, 0, 1, 1, 5, 6],
    ])
    hyp = M.emission_hypotheses(tracks, META, emit_max_coast=1)
    row = hyp[hyp["frame"] == 1].iloc[0]
    u, v = 600.0, 180.0                      # X=Y=0 projects to the principal point
    assert abs((row["x1"] + row["x2"]) / 2 - u) < 1e-6
    assert abs((row["y1"] + row["y2"]) / 2 - v) < 1e-6
    assert abs((row["x2"] - row["x1"]) - 50.0) < 1e-6   # 100 px * (10/20)
    assert M.emission_hypotheses(tracks, META, emit_max_coast=0)["frame"].tolist() == [0]


def test_pooling_uses_raw_counts_not_mean_of_ratios():
    a = {"num_objects": 100, "num_misses": 10, "num_false_positives": 0, "num_switches": 0,
         "num_detections": 90, "num_fragmentations": 0, "idtp": 90, "idfp": 0, "idfn": 10,
         "dist_sum": 0.0}
    b = {"num_objects": 10, "num_misses": 0, "num_false_positives": 30, "num_switches": 0,
         "num_detections": 10, "num_fragmentations": 0, "idtp": 10, "idfp": 30, "idfn": 0,
         "dist_sum": 0.0}
    pooled = M.pool([a, b])
    assert abs(pooled["mota"] - (1 - 40 / 110)) < 1e-12          # not mean(0.9, -2.0)
    assert abs(pooled["idf1"] - 2 * 100 / (2 * 100 + 30 + 10)) < 1e-12


def test_geometry_helpers():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[5, 0, 15, 10], [20, 20, 30, 30]])
    assert abs(iou_matrix(a, b)[0, 0] - 50 / 150) < 1e-12
    assert iou_matrix(a, b)[0, 1] == 0
    assert abs(ioa_matrix(b[:1], a)[0, 0] - 0.5) < 1e-12
    # car centre at Z=10, length 4 along X, width 2 along Z: front face at Z=9
    z = cuboid_front_depth(600.0, 150.0, 700.0, 700.0, 600.0, 180.0, 1.5, 2.0, 4.0, 0.0, 1.0, 10.0, 0.0)
    assert z is not None and abs(z - 9.0) < 1e-9
    assert cuboid_front_depth(0.0, 0.0, 700.0, 700.0, 600.0, 180.0, 1.5, 2.0, 4.0, 0.0, 1.0, 10.0, 0.0) is None
