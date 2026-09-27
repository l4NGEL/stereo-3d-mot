#!/usr/bin/env python3
"""Figures for docs/evaluation/REPORT.md, drawn from the results JSON and the raw dumps.
Writes docs/evaluation/figures/*.png (small, ~100 dpi)."""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import kitti_io as K              # noqa: E402

R = "docs/evaluation/results"
OUT = "docs/evaluation/figures"


def J(name):
    return json.load(open(os.path.join(R, name)))


def fig_fp_attribution():
    fig, ax = plt.subplots(figsize=(8, 3.2))
    cats = ["ghost_coasting", "overlaps_valid_label_3d_gate_rejected", "on_ignored_or_neutral_label", "in_dontcare",
            "two_wheeler_class", "too_small", "residual"]
    labels = ["coasting 'ghost' track", "3D gate rejected a labelled object", "on an ignored/neutral label", "in a DontCare region",
              "bicycle/motorcycle class", "too small (<25 px)", "residual (manual review)"]
    colors = ["#c0392b", "#e67e22", "#f1c40f", "#95a5a6", "#7f8c8d", "#bdc3c7", "#2980b9"]
    for i, (kind, title) in enumerate((("dev", "dev (0000, 0003, 0017)"), ("test", "held-out (8 sequences)"))):
        d = J(f"diagnosis_{kind}.json")["legacy_fp_breakdown"]
        left = 0.0
        for c, lab, col in zip(cats, labels, colors):
            v = 100 * d["shares"][c]
            ax.barh(i, v, left=left, color=col, label=lab if i == 0 else None)
            if v > 4:
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=8)
            left += v
        ax.text(101, i, f"n={d['total']}", va="center", fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["dev", "held-out"])
    ax.set_xlim(0, 112)
    ax.set_xlabel("share of legacy-protocol false positives [%]")
    ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    ax.set_title("Where the negative-MOTA false positives come from (as-shipped 3D Mahalanobis tracker)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_fp_attribution.png"), dpi=110)
    plt.close(fig)


def fig_pr():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, title in zip(axes, ("Vehicle@IoU0.5", "Person@IoU0.5"), ("Vehicle (IoU 0.5)", "Person (IoU 0.5)")):
        for kind, style in (("dev", "--"), ("test", "-")):
            r = J(f"detection_{kind}.json")["results"][key]
            pts = np.array(r["pr_curve_points"])
            ax.plot(pts[:, 0], pts[:, 1], style, label=f"{kind}: AP40={r['ap_r40']:.2f}")
            op = r["operating_point"]
            ax.plot(op["recall"], op["precision"], "o", color=ax.lines[-1].get_color())
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(title + "; dots = score 0.25", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_pr_curves.png"), dpi=110)
    plt.close(fig)


def fig_tracking():
    t = J("tracking_test.json")
    names = [("A4_E30(shipped mahalanobis3d)", "shipped\n3D Mahal."), ("S*_E30(shipped iou2d)", "shipped\nIoU-2D"), ("F_E30(shipped fused)", "shipped\nfused"),
             ("A4_plain_mahalanobis", "3D Mahal.\nE=0"), ("S*", "IoU-2D\nE=0 (S*)"), ("F", "fused\nE=0"), ("SORT", "SORT-2D\n(no depth)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, m, title in zip(axes, ("mota", "idf1"), ("pooled MOTA (P-K2D)", "pooled IDF1 (P-K2D)")):
        vals = [t["systems"][n]["pooled"][m] for n, _ in names]
        cols = ["#c0392b"] * 3 + ["#2980b9"] * 3 + ["#27ae60"]
        ax.bar(range(len(vals)), vals, color=cols)
        for k, (n, _) in enumerate(names):
            ci = t["ci95_pooled"].get(n)
            if ci:
                ax.errorbar(k, ci[m]["value"], yerr=[[ci[m]["value"] - ci[m]["lo"]], [ci[m]["hi"] - ci[m]["value"]]], color="k", capsize=3, lw=1)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([l for _, l in names], fontsize=7)
        ax.set_title(title + "; held-out, error bars = 95% bootstrap over sequences", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_tracking.png"), dpi=110)
    plt.close(fig)


def fig_latency():
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for tag, label, col in (("tiled_local", "CPU, tiled stereo (shipped)", "#2980b9"), ("untiled_local", "CPU, untiled stereo", "#95a5a6"),
                            ("cuda_clean", "CUDA detector, tiled stereo", "#27ae60")):
        fr = []
        for f in sorted(glob.glob(f"out/eval/raw/speed/{tag}_rep*_*.csv")):
            d = pd.read_csv(f)
            fr.append(d[d.warm == 0].total_ms)
        if fr:
            x = pd.concat(fr)
            ax.hist(x, bins=np.arange(80, 340, 4), alpha=0.55, color=col, label=f"{label}: median {x.median():.0f} ms, p95 {x.quantile(.95):.0f} ms", density=True)
    ax.axvline(50, color="r", ls="--")
    ax.text(51, ax.get_ylim()[1] * 0.9, "20 FPS budget (50 ms)", color="r", fontsize=8)
    ax.set_xlabel("end-to-end latency per frame incl. image decode [ms]")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    ax.set_title("Per-frame latency, KITTI 1242x375, i9-9880H (16 threads), sequential loop", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_latency.png"), dpi=110)
    plt.close(fig)


def fig_vo():
    seqs = ["00", "06", "07", "08", "10"]
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    vo = J("vo.json")["test"]
    for ax, s in zip(axes, seqs):
        d = f"out/eval/raw/vo/test_tiled_{s}"
        est = K.read_kitti_poses(os.path.join(d, "poses_est.txt"))
        gt = K.read_kitti_poses(os.path.join(d, "poses_gt.txt"))
        ax.plot(gt[:, 0, 3], gt[:, 2, 3], "k-", label="ground truth")
        ax.plot(est[:, 0, 3], est[:, 2, 3], "r-", label="stereo VO")
        ax.set_aspect("equal")
        t = vo[f"test_tiled_{s}"]["segments"]["t_rel_pct_devkit"]
        ax.set_title(f"seq {s}: t_rel {t:.1f}%", fontsize=9)
        ax.set_xlabel("x [m]")
    axes[0].set_ylabel("z [m]")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_vo.png"), dpi=100)
    plt.close(fig)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for f in (fig_fp_attribution, fig_pr, fig_tracking, fig_latency, fig_vo):
        f()
        print("wrote", f.__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
