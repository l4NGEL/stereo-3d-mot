"""An independent CLEAR-MOT + IDF1 counter (PROTOCOL.md TR-6).

Consumes exactly the per-frame (gt_ids, hyp_ids, dist) triples that mot_eval feeds to
py-motmetrics, but implements the counting itself, written from the metric definitions
(Bernardin & Stiefelhagen 2008; Ristani et al. 2016) with scipy's assignment solver. The
two must agree exactly on FP / FN / IDSW / matches / IDTP / IDFP / IDFN.

Per frame: (1) keep each GT id's previous hypothesis id if that id is present again and
within the gate (persisting across gaps); (2) optimally match the remainder by minimum
total distance; a step-2 match that contradicts the GT id's previous hypothesis id is an
ID switch; (3) leftovers are misses / false positives.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _match_min_cost(dist: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-total-distance matching restricted to finite entries."""
    if dist.size == 0:
        return []
    finite = np.isfinite(dist)
    if not finite.any():
        return []
    big = float(np.nanmax(dist[finite])) * dist.size + 1e6
    cost = np.where(finite, dist, big)
    r, c = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(r, c) if finite[i, j]]


def count(inputs: list[tuple]) -> dict:
    prev: dict[int, int] = {}
    objects = matches = switches = misses = fps = 0
    dist_sum = 0.0
    co: dict[tuple[int, int], int] = {}
    gt_frames: dict[int, int] = {}
    hyp_frames: dict[int, int] = {}
    events: dict[int, list[bool]] = {}          # gt id -> per-appearance matched?

    for gt_ids, hyp_ids, dist in inputs:
        gt_ids = [int(g) for g in gt_ids]
        hyp_ids = [int(h) for h in hyp_ids]
        dist = np.asarray(dist, dtype=float).reshape(len(gt_ids), len(hyp_ids))
        objects += len(gt_ids)
        for g in gt_ids:
            gt_frames[g] = gt_frames.get(g, 0) + 1
        for h in hyp_ids:
            hyp_frames[h] = hyp_frames.get(h, 0) + 1
        for i, g in enumerate(gt_ids):
            for j, h in enumerate(hyp_ids):
                if np.isfinite(dist[i, j]):
                    co[(g, h)] = co.get((g, h), 0) + 1

        used_g, used_h = set(), set()
        hyp_index = {h: j for j, h in enumerate(hyp_ids)}
        for i, g in enumerate(gt_ids):
            h = prev.get(g)
            if h is None or h not in hyp_index:
                continue
            j = hyp_index[h]
            if j in used_h or not np.isfinite(dist[i, j]):
                continue
            used_g.add(i)
            used_h.add(j)
            matches += 1
            dist_sum += dist[i, j]
            events.setdefault(g, []).append(True)

        rem_i = [i for i in range(len(gt_ids)) if i not in used_g]
        rem_j = [j for j in range(len(hyp_ids)) if j not in used_h]
        if rem_i and rem_j:
            sub = dist[np.ix_(rem_i, rem_j)]
            for a, b in _match_min_cost(sub):
                i, j = rem_i[a], rem_j[b]
                g, h = gt_ids[i], hyp_ids[j]
                if g in prev and prev[g] != h:
                    switches += 1
                else:
                    matches += 1
                dist_sum += dist[i, j]
                prev[g] = h
                used_g.add(i)
                used_h.add(j)
                events.setdefault(g, []).append(True)

        for i, g in enumerate(gt_ids):
            if i not in used_g:
                misses += 1
                events.setdefault(g, []).append(False)
        fps += len(hyp_ids) - len(used_h)

    # fragmentations: per GT id, tracked -> not tracked transitions inside [first, last match]
    frag = 0
    for ev in events.values():
        idx = [k for k, m in enumerate(ev) if m]
        if not idx:
            continue
        seg = ev[idx[0]: idx[-1] + 1]
        frag += sum(1 for a, b in zip(seg, seg[1:]) if a and not b)

    # IDF1: one global one-to-one assignment of GT ids to hypothesis ids maximising
    # the number of frames they co-occur within the gate.
    g_ids, h_ids = sorted(gt_frames), sorted(hyp_frames)
    idtp = 0
    if g_ids and h_ids:
        m = np.zeros((len(g_ids), len(h_ids)))
        gi = {g: i for i, g in enumerate(g_ids)}
        hi = {h: j for j, h in enumerate(h_ids)}
        for (g, h), v in co.items():
            m[gi[g], hi[h]] = v
        r, c = linear_sum_assignment(-m)
        idtp = int(m[r, c].sum())
    return {
        "num_objects": objects,
        "num_matches": matches,
        "num_switches": switches,
        "num_misses": misses,
        "num_false_positives": fps,
        "num_detections": matches + switches,
        "num_fragmentations": frag,
        "idtp": idtp,
        "idfp": sum(hyp_frames.values()) - idtp,
        "idfn": sum(gt_frames.values()) - idtp,
        "dist_sum": dist_sum,
    }
