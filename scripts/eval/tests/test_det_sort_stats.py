import numpy as np
import pandas as pd

import det_eval as D
import sort_baseline as S
import stats as ST


def test_ap_hand_computed():
    status = ["tp", "tp", "fp", "tp", "fp"]
    res = pd.DataFrame({"frame": 0, "score": [.9, .8, .7, .6, .5], "status": status,
                        "iou": [.8, .8, 0, .8, 0]})
    prec, rec, _ = D.pr_curve(res, n_gt=4)
    assert abs(D.ap_all_points(prec, rec) - 0.6875) < 1e-12
    assert abs(D.ap_r40(prec, rec) - 0.6875) < 1e-12
    op = D.operating_point(res, 4, 0.65)          # keeps the first three detections
    assert (op["tp"], op["fp"], op["fn"]) == (2, 1, 2)


def test_perfect_detector_has_ap_one():
    res = pd.DataFrame({"frame": 0, "score": np.linspace(.99, .5, 10), "status": ["tp"] * 10,
                        "iou": .9})
    prec, rec, _ = D.pr_curve(res, n_gt=10)
    assert abs(D.ap_r40(prec, rec) - 1.0) < 1e-12 and abs(D.ap_all_points(prec, rec) - 1.0) < 1e-12


def test_match_frame_duplicates_ignored_regions_and_small_boxes():
    valid = np.array([[100., 100, 200, 200]])
    ign = np.array([[400., 100, 500, 200]])
    dc = np.array([[700., 100, 800, 200]])
    dets = np.array([[100., 100, 200, 200],      # TP
                     [102., 101, 201, 199],      # duplicate of the same GT -> FP
                     [402., 101, 498, 199],      # on ignored GT -> ign
                     [710., 110, 790, 190],      # inside DontCare -> ign
                     [900., 100, 950, 110],      # 10 px high -> ign
                     [900., 300, 1000, 400]])    # nothing there -> FP
    scores = np.array([.9, .8, .7, .6, .5, .4])
    st, _ = D.match_frame(dets, scores, valid, ign, dc, 0.5)
    assert list(st) == ["tp", "fp", "ign", "ign", "ign", "fp"]


def _dets_from_path(n, cls=2, miss=()):
    rows = []
    for f in range(n):
        if f in miss:
            continue
        rows.append([f, cls, .9, 100 + 5 * f, 100, 60, 40])
    return pd.DataFrame(rows, columns=["frame", "cls", "score", "x", "y", "w", "h"])


def test_sort_keeps_one_identity_on_a_clean_track():
    tr = S.sort_track(_dets_from_path(25), 25)
    assert tr["id"].nunique() == 1 and len(tr) >= 20


def test_sort_reports_only_updated_tracks_and_needs_max_age_to_bridge_gaps():
    dets = _dets_from_path(30, miss=(12, 13, 14))
    short = S.sort_track(dets, 30, max_age=1)
    long_ = S.sort_track(dets, 30, max_age=5)
    assert (short["tsu"] == 0).all()
    assert short["id"].nunique() >= 2          # SORT's default max_age=1 loses the object
    assert long_["id"].nunique() == 1          # a longer memory bridges the gap


def test_bootstrap_detects_a_consistent_improvement_and_no_difference_for_identical_systems():
    base = {"num_objects": 100, "num_misses": 30, "num_false_positives": 20, "num_switches": 5,
            "num_detections": 70, "num_fragmentations": 0, "idtp": 60, "idfp": 30, "idfn": 40,
            "dist_sum": 0.0}
    better = dict(base, num_misses=20, num_false_positives=10, idtp=70, idfp=20, idfn=30)
    a = {f"s{i}": better for i in range(8)}
    b = {f"s{i}": base for i in range(8)}
    r = ST.paired_bootstrap(a, b, "mota", n_boot=500)
    assert r["diff"] > 0 and r["excludes_zero"] and r["wins_a"] == 8
    same = ST.paired_bootstrap(b, b, "mota", n_boot=500)
    assert same["diff"] == 0 and not same["excludes_zero"]


def test_wilson_interval_brackets_the_proportion():
    lo, hi = ST.wilson(30, 100)
    assert lo < 0.30 < hi and 0.2 < lo and hi < 0.4
