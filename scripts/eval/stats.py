"""Paired bootstrap over sequences (PROTOCOL.md §3.8).

The resampling unit is the sequence. Each system supplies its per-sequence RAW counts
(summed over classes); a resample draws sequences with replacement, re-pools the counts,
recomputes the metric from the sums, and the paired difference is taken on the same draw.
"""
from __future__ import annotations

import numpy as np

import mot_eval as M


def metric_of(counts_by_seq: dict[str, dict], seqs: list[str], metric: str) -> float:
    return M.pool([counts_by_seq[s] for s in seqs])[metric]


def paired_bootstrap(a: dict[str, dict], b: dict[str, dict], metric: str, n_boot: int = 10000,
                     seed: int = 0) -> dict:
    """Difference metric(a) - metric(b) with a 95% percentile interval and per-sequence wins."""
    seqs = sorted(set(a) & set(b))
    rng = np.random.default_rng(seed)
    point = metric_of(a, seqs, metric) - metric_of(b, seqs, metric)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        draw = [seqs[k] for k in rng.integers(0, len(seqs), len(seqs))]
        diffs[i] = metric_of(a, draw, metric) - metric_of(b, draw, metric)
    finite = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(finite, [2.5, 97.5])
    per_seq = [M.pool([a[s]])[metric] - M.pool([b[s]])[metric] for s in seqs]
    return {"metric": metric, "diff": point, "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "wins_a": int(sum(d > 0 for d in per_seq)), "wins_b": int(sum(d < 0 for d in per_seq)),
            "n_seq": len(seqs)}


def bootstrap_ci(counts_by_seq: dict[str, dict], metric: str, n_boot: int = 10000,
                 seed: int = 0) -> dict:
    seqs = sorted(counts_by_seq)
    rng = np.random.default_rng(seed)
    point = metric_of(counts_by_seq, seqs, metric)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        draw = [seqs[k] for k in rng.integers(0, len(seqs), len(seqs))]
        vals[i] = metric_of(counts_by_seq, draw, metric)
    finite = vals[np.isfinite(vals)]
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return {"metric": metric, "value": point, "lo": float(lo), "hi": float(hi), "n_seq": len(seqs)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (used for the manual-review distribution)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
