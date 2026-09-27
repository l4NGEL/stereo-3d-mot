#!/usr/bin/env python3
"""Builds the fcde488-vs-remediated side-by-side table for all 32 acceptance criteria
(the final deliverable table) and embeds it into docs/evaluation/REPORT.md between the
REMEDIATION-TABLE markers. Reads only already-written result files; computes nothing new."""
from __future__ import annotations

import json
import re

R = "docs/evaluation/results"
v = {r["id"]: r for r in json.load(open(f"{R}/verdicts.json"))}  # fcde488, frozen

# id -> (new_measured, new_verdict, note) for the criteria a phase actually touched;
# everything else is reported as unchanged from fcde488.
CHANGED = {
    "DE-1": ("P 0.851 [dev-selected score=0.31, held-out], R 0.716", "GEÇTİ",
            "Phase 3: score_threshold 0.25→0.31 (dev-selected, held-out-verified); both floors now cleared"),
}
INFORMATIONAL = {  # verdict and criterion text unchanged, but new evidence was produced this remediation
    "TR-1": "Phase 1 confirms: the emitted system was already E=0 pre-freeze; root cause (ghost tracks) now fixed in the product itself, see Phase 1",
    "TR-2": "unchanged; Phase 2's range-gated association (dev-selected) scores MOTA/IDF1 in the same range, not enough to cross 0.50/0.60",
    "TR-3": "unchanged",
    "TR-4": "unchanged",
    "TR-5": "unchanged; Phase 2's improved association is also still worse than SORT (MOTA −0.051 [−0.116,−0.027])",
    "DE-3": "Phase 3: same global threshold change improves P 0.709→0.741, R 0.679→0.648 — closer, still short of 0.75",
    "VO-1": "Phase 3 quantifies the cause (rotation-error breakdown) but applies no fix; number unchanged",
    "VO-2": "unchanged; see VO-1",
    "VO-4": "unchanged; see VO-1",
    "SP-1": "Phase 4 breaks down the bottleneck (stereo 52.8%, detect 30.9%) and costs the two candidates that could close the gap; neither was safe to apply within scope",
    "SP-2": "unchanged; see SP-1",
    "SP-3": "unchanged; see SP-1",
    "ST-3": "unchanged; Phase 3's one tested stereo candidate (MODE_HH) did not improve D1-all, so was not applied",
}

ORDER = ["SP-1", "SP-2", "SP-3", "DE-1", "DE-2", "DE-3", "DE-4",
        "TR-1", "TR-2", "TR-3", "TR-4", "TR-5", "TR-6",
        "TR-7a", "TR-7b", "TR-7c", "TR-7d", "TR-7e", "TR-7f", "TR-7g", "TR-7h",
        "VO-1", "VO-2", "VO-3", "VO-4", "ST-1", "ST-2", "ST-3", "ST-4", "IN-1", "IN-2", "IN-3"]
assert set(ORDER) == set(v), (set(ORDER) ^ set(v))


def esc(t: str) -> str:
    return t.replace("|", "\\|")


def main() -> int:
    lines = ["| ID | fcde488 (old) | fcde488 verdict | Remediated (new) | New verdict | Changed? |",
            "|---|---|---|---|---|---|"]
    n_changed = 0
    for cid in ORDER:
        old = v[cid]
        if cid in CHANGED:
            new_measured, new_verdict, note = CHANGED[cid]
            changed = "**YES**"
            n_changed += 1
            new_cell = f"{new_measured} — {note}"
        else:
            new_verdict = old["verdict"]
            changed = "no"
            new_cell = INFORMATIONAL.get(cid, "unchanged (no phase touched this criterion)")
        lines.append(f"| {cid} | {esc(old['measured'])} | **{old['verdict']}** | {esc(new_cell)} | **{new_verdict}** | {changed} |")
    lines.append("")
    lines.append(f"**{n_changed} of 32 criteria changed verdict** (DE-1, KALDI → GEÇTİ). "
                 "The other 31 are unchanged from fcde488 — several have new, independent evidence behind them "
                 "(Phases 1–4 above), which did not move the number past its threshold; that is reported as plainly "
                 "as the one that did.")
    body = "\n".join(lines)
    text = open("docs/evaluation/REPORT.md", encoding="utf-8").read()
    new, n = re.subn(r"<!-- REMEDIATION-TABLE-BEGIN -->.*?<!-- REMEDIATION-TABLE-END -->",
                     lambda _: f"<!-- REMEDIATION-TABLE-BEGIN -->\n{body}\n<!-- REMEDIATION-TABLE-END -->",
                     text, flags=re.S)
    if n != 1:
        raise SystemExit("REMEDIATION-TABLE markers not found exactly once")
    open("docs/evaluation/REPORT.md", "w", encoding="utf-8").write(new)
    print(f"embedded, {n_changed} changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
