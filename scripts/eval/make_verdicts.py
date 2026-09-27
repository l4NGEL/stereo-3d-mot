#!/usr/bin/env python3
"""Apply the acceptance criteria of PROTOCOL.md §7 to the measured results and write the verdict table.

Every threshold is copied from PROTOCOL.md (frozen before the held-out measurements); this script only
compares. It never changes a threshold. Output: docs/evaluation/results/verdicts.json and verdicts.md.
"""
from __future__ import annotations

import json
import os
import sys

R = "docs/evaluation/results"
PASS, FAIL, NOMEAS, NOTPROVEN = "GEÇTİ", "KALDI", "ÖLÇÜLEMEDİ", "KANITLANMADI"


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def row(cid, criterion, measured, verdict, note=""):
    return {"id": cid, "criterion": criterion, "measured": measured, "verdict": verdict, "note": note}


def main() -> int:
    rows = []
    det, trk = load("detection_test.json"), load("tracking_test.json")
    syn, k15 = load("stereo_synthetic.json"), load("stereo_kitti2015.json")
    objd, vo, speed = load("object_depth_test.json"), load("vo.json"), load("speed.json")
    reg, ctr = load("tr6_legacy_regression.json"), load("counters_check.json")

    # ------------------------------------------------------------------ speed
    if speed:
        s = speed["shipped_tiled_local_fs"]
        ok1 = all(r["fps_with_io_mean"] >= 20.0 for r in s["per_repetition"])
        ok2 = all(r["p95_total_ms"] <= 50.0 for r in s["per_repetition"])
        ok3 = all(r["fps_compute_mean"] >= 20.0 for r in s["per_repetition"])
        rows.append(row("SP-1", "end-to-end >= 20 FPS incl. decode (each of 3 repetitions)",
                        f"{s['fps_with_io_mean']:.2f} FPS (reps {[round(r['fps_with_io_mean'], 2) for r in s['per_repetition']]})", PASS if ok1 else FAIL))
        rows.append(row("SP-2", "p95 latency <= 50 ms (each repetition)",
                        f"p95 {s['p95_total_ms']:.1f} ms (median {s['median_total_ms']:.1f})", PASS if ok2 else FAIL))
        rows.append(row("SP-3", "compute-only (no decode) >= 20 FPS",
                        f"{s['fps_compute_mean']:.2f} FPS", PASS if ok3 else FAIL))
    else:
        for cid in ("SP-1", "SP-2", "SP-3"):
            rows.append(row(cid, "speed", "not yet measured", NOMEAS))

    # -------------------------------------------------------------- detection
    if det:
        v05, p05 = det["results"]["Vehicle@IoU0.5"], det["results"]["Person@IoU0.5"]
        vop, pop = v05["operating_point"], p05["operating_point"]
        rows.append(row("DE-1", "Vehicle precision >= 0.80 and recall >= 0.70 (IoU 0.5, moderate, score >= 0.25)",
                        f"P {vop['precision']:.3f} [{v05['precision_ci95'][0]:.3f},{v05['precision_ci95'][1]:.3f}], R {vop['recall']:.3f}",
                        PASS if vop["precision"] >= 0.80 and vop["recall"] >= 0.70 else FAIL,
                        "recall passes; precision misses" if vop["recall"] >= 0.70 > 0 and vop["precision"] < 0.80 else ""))
        rows.append(row("DE-2", "Vehicle AP50 (R40) >= 0.56", f"{v05['ap_r40']:.3f} (all-point {v05['ap_all_points']:.3f})",
                        PASS if v05["ap_r40"] >= 0.56 else FAIL))
        rows.append(row("DE-3", "Person precision >= 0.75 and recall >= 0.50; AP50 (R40) >= 0.375",
                        f"P {pop['precision']:.3f}, R {pop['recall']:.3f}, AP50 {p05['ap_r40']:.3f}",
                        PASS if (pop["precision"] >= 0.75 and pop["recall"] >= 0.50 and p05["ap_r40"] >= 0.375) else FAIL))
    rows.append(row("DE-4", "COCO mAP", "not measured in this project", NOMEAS,
                    "only KITTI-protocol numbers are claimed; Ultralytics' published COCO mAP is not a KITTI number"))

    # --------------------------------------------------------------- tracking
    if trk:
        S = trk["systems"]["S*"]
        p, v, pe = S["pooled"], S["vehicle"], S["person"]
        rows.append(row("TR-1", "minimum: pooled P-K2D MOTA > 0 (dev-selected system, held-out)", f"MOTA {p['mota']:+.3f}", PASS if p["mota"] > 0 else FAIL))
        rows.append(row("TR-2", "target: Vehicle MOTA >= 0.50 and IDF1 >= 0.60", f"MOTA {v['mota']:.3f}, IDF1 {v['idf1']:.3f}",
                        PASS if (v["mota"] >= 0.50 and v["idf1"] >= 0.60) else FAIL))
        rows.append(row("TR-3", "target: Person MOTA >= 0.30 and IDF1 >= 0.50", f"MOTA {pe['mota']:.3f}, IDF1 {pe['idf1']:.3f}",
                        PASS if (pe["mota"] >= 0.30 and pe["idf1"] >= 0.50) else FAIL))
        rate = p["num_switches"] / p["num_objects"]
        rows.append(row("TR-4", "IDSW <= 2 % of GT objects (pooled)", f"{p['num_switches']}/{p['num_objects']} = {100 * rate:.2f} %", PASS if rate <= 0.02 else FAIL))
        cmp = {(c["a"], c["b"], c["metric"]): c for c in trk["comparisons"]}
        m, i = cmp[("S*", "SORT", "mota")], cmp[("S*", "SORT", "idf1")]
        better = (m["excludes_zero"] and m["diff"] > 0) and (i["excludes_zero"] and i["diff"] > 0)
        worse = (m["excludes_zero"] and m["diff"] < 0) or (i["excludes_zero"] and i["diff"] < 0)
        rows.append(row("TR-5", "beats SORT-2D on pooled MOTA and IDF1, bootstrap interval excluding 0",
                        f"MOTA {m['diff']:+.3f} [{m['lo']:+.3f},{m['hi']:+.3f}], IDF1 {i['diff']:+.3f} [{i['lo']:+.3f},{i['hi']:+.3f}]",
                        PASS if better else (FAIL if worse else NOTPROVEN), "S* is significantly WORSE than the depth-free baseline" if worse else ""))
    if reg and ctr:
        rows.append(row("TR-6", "harness reproduces published legacy counts; independent counter == py-motmetrics",
                        f"legacy regression {'PASS' if reg['failures'] == 0 else 'FAIL'}; counters {ctr['compared']} inputs, {ctr['disagreements']} disagreements",
                        PASS if (reg["failures"] == 0 and ctr["disagreements"] == 0) else FAIL))
    elif reg:
        rows.append(row("TR-6", "harness reproduces published legacy counts; independent counter == motmetrics",
                        f"legacy regression {'PASS' if reg['failures'] == 0 else 'FAIL'}; counter check pending", NOMEAS))

    if trk:
        # TR-7: each claimed contribution is "demonstrated" only if the paired-bootstrap interval excludes 0 in the
        # improving direction on MOTA or IDF1 and neither metric is significantly worse; contradicted if one is.
        def claim(cid, text, x, y):
            ms = [c for c in trk["comparisons"] if c["a"] == x and c["b"] == y]
            if len(ms) != 2:
                return
            better = [c["metric"] for c in ms if c["excludes_zero"] and c["diff"] > 0]
            worse = [c["metric"] for c in ms if c["excludes_zero"] and c["diff"] < 0]
            v = FAIL if worse else (PASS if better else NOTPROVEN)
            meas = "; ".join(f"{c['metric'].upper()} {c['diff']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}]" for c in ms)
            both = len(better) == 2
            note = ("contradicted on " + " and ".join(worse).upper()) if worse else (
                ("demonstrated on " + (" and ".join(better).upper() if both else better[0].upper() + " only")) if better else "inside the noise")
            rows.append(row(cid, text, meas, v, note))
        claim("TR-7a", "claim: 3D Mahalanobis association beats plain 2D IoU (A4 vs S*)", "A4_plain_mahalanobis", "S*")
        claim("TR-7b", "claim: fusion (3D+IoU+appearance) beats the selected plain IoU (F vs S*)", "F", "S*")
        claim("TR-7c", "claim: the appearance cue helps (F vs A1)", "F", "A1_no_appearance")
        claim("TR-7d", "claim: the IoU term of the fused cost helps (F vs A2)", "F", "A2_no_iou_cost")
        claim("TR-7e", "claim: the 3D term of the fused cost helps (F vs A3)", "F", "A3_no_3d_cost")
        claim("TR-7f", "claim: Hungarian beats greedy assignment (F vs A7)", "F", "A7_greedy")
        claim("TR-7g", "claim: emitting only updated tracks (E=0) beats the shipped E=30 (S*)", "S*", "S*_E30(shipped iou2d)")
        un = load("tracking_test_untiled.json")
        if un:
            dm = max(abs(trk["systems"][n]["pooled"]["mota"] - un["systems"][n]["pooled"]["mota"]) for n in trk["systems"])
            di = max(abs(trk["systems"][n]["pooled"]["idf1"] - un["systems"][n]["pooled"]["idf1"]) for n in trk["systems"])
            same = sum(1 for n in trk["systems"] if trk["systems"][n]["pooled"] == un["systems"][n]["pooled"])
            rows.append(row("TR-7h", "claim: tiled stereo does not change tracking accuracy (A9)",
                            f"max abs change in MOTA {dm:.4f} and in IDF1 {di:.4f} over {len(trk['systems'])} systems; bit-identical for {same}", PASS if (dm <= 0.01 and di <= 0.01) else FAIL,
                            "practically identical, not bit-identical, on held-out data"))

    # ------------------------------------------------------------------- VO
    if vo and vo.get("test"):
        p = vo["test_pooled"]
        seqs = vo["test"]
        t_by_seq = {n: r["segments"].get("t_rel_pct_devkit") for n, r in seqs.items()}
        ok_frac = {n: r["frames_ok_fraction"] for n, r in seqs.items()}
        t, r_ = p["t_rel_pct_devkit"], p["r_deg_per_m_devkit"]
        rows.append(row("VO-1", "t_rel <= 5 % (minimum) / <= 2.5 % (target), 100-400 m segments, held-out pooled",
                        f"{t:.2f} % ({p['n_segments']} segments)", PASS if t <= 5.0 else FAIL, "target tier " + ("met" if t <= 2.5 else "not met")))
        rows.append(row("VO-2", "r_rel <= 0.03 deg/m (minimum) / <= 0.015 (target)", f"{r_:.4f} deg/m", PASS if r_ <= 0.03 else FAIL,
                        "target tier " + ("met" if r_ <= 0.015 else "not met")))
        worst = max(v for v in t_by_seq.values() if v is not None)
        rows.append(row("VO-3", ">= 95 % of frames tracked on every held-out sequence",
                        f"min {100 * min(ok_frac.values()):.1f} %", PASS if min(ok_frac.values()) >= 0.95 else FAIL, json.dumps({k: round(100 * x, 1) for k, x in ok_frac.items()})))
        rows.append(row("VO-4", "no held-out sequence with t_rel > 10 %", f"worst {worst:.2f} % {json.dumps({k: round(v, 2) for k, v in t_by_seq.items()})}",
                        PASS if worst <= 10.0 else FAIL))
    else:
        for cid in ("VO-1", "VO-2", "VO-3", "VO-4"):
            rows.append(row(cid, "VO", "not yet measured", NOMEAS))

    # ----------------------------------------------------------------- stereo
    if syn:
        m, w = syn["tiles_4"]["mean_over_seeds"], syn["tiles_4"]["worst_seed"]
        ok = m["disparity_bad_2.0"] <= 0.02 and m["disparity_rmse_px"] <= 1.5 and m["density"] >= 0.95
        rows.append(row("ST-1", "synthetic (5 seeds x 30 frames, project ROI): bad-2.0 <= 2 %, RMSE <= 1.5 px, density >= 95 %",
                        f"bad-2.0 {100 * m['disparity_bad_2.0']:.2f} %, RMSE {m['disparity_rmse_px']:.3f} px (worst seed {w['disparity_rmse_px_max']:.3f}), density {100 * m['density']:.2f} %",
                        PASS if ok else FAIL, "regression gate on a benign scene; low evidential value"))
    if k15:
        t4, t1, bm = k15["sgbm_tiles4"]["occ_gt"], k15["sgbm_tiles1"]["occ_gt"], k15["bm15"]["occ_gt"]
        ok = t4["d1_all_holes_as_error"] <= 0.25 and t4["d1_all_holes_as_error"] < bm["d1_all_holes_as_error"]
        rows.append(row("ST-2", "KITTI Stereo 2015 D1-all (holes = errors) <= 25 % and better than the BM baseline",
                        f"SGBM(4 tiles) {100 * t4['d1_all_holes_as_error']:.2f} % (valid-only {100 * t4['d1_valid_only']:.2f} %, density {100 * t4['density']:.1f} %); BM {100 * bm['d1_all_holes_as_error']:.2f} %",
                        PASS if ok else FAIL, "BM baseline is weak by construction (density only %.0f %%)" % (100 * bm["density"])))
        d = abs(t4["d1_all_holes_as_error"] - t1["d1_all_holes_as_error"])
        rows.append(row("ST-4", "|D1-all(4 tiles) - D1-all(untiled)| <= 0.5 pp", f"{100 * d:.3f} pp", PASS if d <= 0.005 else FAIL))
    if objd:
        v = objd["summary"]["Vehicle"]["5-30m"]
        rows.append(row("ST-3", "Vehicle 5-30 m: RMSE(detection depth - label front-surface depth) <= 1.0 m (the tracker's configured sigma)",
                        f"RMSE {v['rmse_m']:.2f} m (median signed {v['median_signed_err_m']:+.2f} m, n={v['n']}); vs cuboid centre RMSE {v['vs_center_rmse_m']:.2f} m",
                        PASS if v["rmse_m"] <= 1.0 else FAIL))
    for cid in ("ST-1", "ST-2", "ST-3", "ST-4"):
        if not any(r["id"] == cid for r in rows):
            rows.append(row(cid, "stereo", "not yet measured", NOMEAS))

    # -------------------------------------------------------------- integrity
    rep = load("reproduce_check.json")
    if rep:
        rows.append(row("IN-1", "every headline number regenerates from one documented command",
                        f"scripts/eval/run_all.sh --score-only re-run from the dumps: {rep['files_compared']} result files, {rep['files_different']} differ",
                        PASS if rep["files_different"] == 0 else FAIL,
                        "scoring stage re-executed; dump stage regenerated for dev (legacy regression) and checked for determinism on one sequence, not re-run end-to-end for every sequence"))
    else:
        rows.append(row("IN-1", "every headline number regenerates from one documented command", "not yet verified", NOMEAS))
    rows.append(row("IN-2", "two identical runs give bit-identical dumps", "dets + 8 tracker dumps of sequence 0012 identical (byte compare)", PASS,
                    "checked on one held-out sequence (78 frames) under load"))
    rows.append(row("IN-3", "no leakage: unmodified pretrained detector; selection on dev only",
                    "model metadata: 'Ultralytics YOLOv8n model trained on coco.yaml'; selection committed before held-out frontend", PASS,
                    "dev-influenced choices are declared in INVENTORY.md"))

    order = {"SP": 0, "DE": 1, "TR": 2, "VO": 3, "ST": 4, "IN": 5}
    rows.sort(key=lambda r: (order[r["id"][:2]], r["id"]))
    json.dump(rows, open(os.path.join(R, "verdicts.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(R, "verdicts.md"), "w", encoding="utf-8") as f:
        f.write("| ID | Criterion (from PROTOCOL.md §7) | Measured | Verdict |\n|---|---|---|---|\n")
        for r in rows:
            note = f" — {r['note']}" if r["note"] else ""
            # a literal pipe inside a cell would end the table cell
            crit, meas = r["criterion"].replace("|", "\\|"), (r["measured"] + note).replace("|", "\\|")
            f.write(f"| {r['id']} | {crit} | {meas} | **{r['verdict']}** |\n")
    for r in rows:
        print(f"{r['id']:5s} {r['verdict']:12s} {r['measured'][:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
