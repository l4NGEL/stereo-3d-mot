#!/usr/bin/env python3
"""Regenerate Appendix A of docs/evaluation/REPORT.md (per-sequence tracking numbers and localisation
quality) from results/tracking_{dev,test}.json, between the APPENDIX markers (idempotent)."""
from __future__ import annotations

import json
import re

R = "docs/evaluation/results"
REPORT = "docs/evaluation/REPORT.md"


def table(t: dict, names: list[str], title: str) -> str:
    out = [f"**{title}** — P-K2D, pooled over Vehicle and Person per sequence (raw counts; MOTA is *not* averaged across sequences).", ""]
    out.append("| Sequence | Frames | GT object-frames | " + " | ".join(f"{n}: MOTA / IDF1 / FP / FN / IDSW" for n in names) + " |")
    out.append("|---|---|---|" + "---|" * len(names))
    for s in t["sequences"]:
        cells = []
        for n in names:
            p = t["systems"][n]["per_sequence"][s]
            cells.append(f"{p['mota']:+.3f} / {p['idf1']:.3f} / {p['fp']} / {p['fn']} / {p['idsw']}")
        gt = t["systems"][names[0]]["per_sequence"][s]["num_objects"]
        out.append(f"| {s} | {t['n_frames'][s]} | {gt} | " + " | ".join(cells) + " |")
    p = {n: t["systems"][n]["pooled"] for n in names}
    out.append("| **pooled** | " + f"{sum(t['n_frames'].values())} | {p[names[0]]['num_objects']} | " + " | ".join(
        f"**{p[n]['mota']:+.3f}** / {p[n]['idf1']:.3f} / {p[n]['num_false_positives']} / {p[n]['num_misses']} / {p[n]['num_switches']}" for n in names) + " |")
    out.append("")
    return "\n".join(out)


def cost_section() -> str:
    c = json.load(open(f"{R}/variant_cost.json"))
    t = json.load(open(f"{R}/tracking_test.json"))
    sp = json.load(open(f"{R}/speed.json"))
    a = c["association_ms_per_frame"]
    rows = [
        ("plain IoU-2D association (S\\*, the selected)", "A5_plain_iou", "S*", "tracker.update"),
        ("plain 3D Mahalanobis association (A4)", "A4_plain_mahalanobis", "A4_plain_mahalanobis", "tracker.update"),
        ("fused 3D + IoU + appearance (F)", "F_fused", "F", "tracker.update"),
        ("fused without appearance cost (A1)", "A1_no_appearance", "A1_no_appearance", "tracker.update"),
        ("fused without IoU cost (A2)", "A2_no_iou_cost", "A2_no_iou_cost", "tracker.update"),
        ("fused without 3D cost (A3)", "A3_no_3d_cost", "A3_no_3d_cost", "tracker.update"),
        ("fused, greedy instead of Hungarian (A7)", "A7_greedy", "A7_greedy", "tracker.update"),
    ]
    out = ["Accuracy (held-out, P-K2D, E=0) next to the measured cost of the stage each variant changes (ms per frame, mean of all held-out frames, tiled config):", "",
           "| Variant | MOTA | IDF1 | IDSW | Changed stage | Cost |", "|---|---|---|---|---|---|"]
    for label, var, sys_name, stage in rows:
        p = t["systems"][sys_name]["pooled"]
        out.append(f"| {label} | {p['mota']:+.3f} | {p['idf1']:.3f} | {p['num_switches']} | {stage} | {a[var]['mean_ms']:.2f} ms |")
    p = t["systems"]["SORT"]["pooled"]
    d = c["estimate_depth_free_pipeline"]
    out.append(f"| SORT-2D, no stereo (B1) | {p['mota']:+.3f} | {p['idf1']:.3f} | {p['num_switches']} | whole pipeline: no stereo (−{c['stereo_ms_per_frame_tiled']['mean']:.0f} ms), no depth map, no 3D promotion, left image only | "
               f"**estimate** {d['CPU detector']['ms_per_frame']:.0f} ms ({d['CPU detector']['fps']:.1f} FPS) with the CPU detector, {d['CUDA detector (run 1)']['ms_per_frame']:.0f} ms ({d['CUDA detector (run 1)']['fps']:.0f} FPS) with CUDA — not measured end to end |")
    tl, ul = sp["shipped_tiled_local_fs"], sp["untiled_local_fs"]
    out.append(f"| tiled stereo, 4 strips (A9) | ≈ same (§5.3) | ≈ same | ≈ same | stereo matching | {tl['stereo']['mean']:.0f} ms vs {ul['stereo']['mean']:.0f} ms untiled (−{100 * (1 - tl['stereo']['mean'] / ul['stereo']['mean']):.0f} %) |")
    cu1, cu2 = sp["cuda_run1"], sp["cuda_run2_two_reps"]
    out.append(f"| CUDA detector (A10) | same detections (§5.3) | same | same | detector | median {cu1['detect']['median']:.0f}–{cu2['detect']['median']:.0f} ms vs {tl['detect']['median']:.0f} ms on the CPU |")
    out += ["", f"The association step costs 0.1–0.2 ms per frame in every variant — irrelevant next to stereo ({c['stereo_ms_per_frame_tiled']['mean']:.0f} ms) and the detector ({tl['detect']['mean']:.0f} ms). "
            f"Promotion to 3D plus the HSV appearance descriptors together cost {c['promote_plus_appearance_ms_per_frame']['mean']:.1f} ms per frame (§8) and are computed whether or not the appearance weight is zero. "
            "Since the depth-free SORT baseline needs neither the right image, stereo matching nor depth, its estimated cost is about a third of the shipped pipeline's: **the stereo/3D machinery is what makes the pipeline slow, and on this evidence it buys no tracking accuracy** — "
            "a CPU-detector-plus-SORT pipeline would still miss 20 FPS (≈ 15 FPS), a CUDA-detector one would clear it (≈ 31 FPS) by this estimate.", ""]
    return "\n".join(out)


def main() -> int:
    test = json.load(open(f"{R}/tracking_test.json"))
    dev = json.load(open(f"{R}/tracking_dev.json"))
    names_t = ["S*", "F", "SORT", "S*_E30(shipped iou2d)"]
    names_d = ["S*", "F", "SORT", "S*_E30(shipped iou2d)"]
    parts = ["## Appendix A — per-sequence tracking results and localisation quality", "",
             table(test, names_t, "Held-out sequences"), table(dev, names_d, "Dev sequences (design and selection allowed)")]
    loc = ["**Localisation quality** (held-out, pooled over both classes and all sequences): mean IoU of the matched pairs under P-K2D — "
           + ", ".join(f"{n} {test['systems'][n]['mean_iou_of_matches']:.3f}" for n in ("S*", "F", "SORT", "A4_plain_mahalanobis"))
           + ". Under P-LEGACY the mean 3D distance of matched pairs (MOTP) is "
           + ", ".join(f"{k} {v['as_shipped_all_confirmed']['motp']:.3f} m" for k, v in test["legacy"].items())
           + " — that distance includes the label-centre-versus-visible-surface offset discussed in §3 and §6, so it is a localisation *plus* protocol number, not a depth-error estimate (§3 has that).", ""]
    body = "\n".join(parts + loc)
    text = open(REPORT, encoding="utf-8").read()
    new, n = re.subn(r"<!-- APPENDIX-BEGIN -->.*?<!-- APPENDIX-END -->",
                     lambda _: f"<!-- APPENDIX-BEGIN -->\n{body}\n<!-- APPENDIX-END -->", text, flags=re.S)
    if n != 1:
        raise SystemExit("APPENDIX markers not found exactly once in REPORT.md")
    new, n2 = re.subn(r"<!-- COST-BEGIN -->.*?<!-- COST-END -->",
                      lambda _: f"<!-- COST-BEGIN -->\n{cost_section()}\n<!-- COST-END -->", new, flags=re.S)
    if n2 != 1:
        raise SystemExit("COST markers not found exactly once in REPORT.md")
    open(REPORT, "w", encoding="utf-8").write(new)
    print("appendix and cost table embedded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
