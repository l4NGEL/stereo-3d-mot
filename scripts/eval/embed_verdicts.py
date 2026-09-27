#!/usr/bin/env python3
"""Copy results/verdicts.md into docs/evaluation/REPORT.md between the VERDICTS markers (idempotent),
so the acceptance table in the report is always exactly what make_verdicts.py computed."""
from __future__ import annotations

import re

REPORT = "docs/evaluation/REPORT.md"
VERDICTS = "docs/evaluation/results/verdicts.md"


def main() -> int:
    text = open(REPORT, encoding="utf-8").read()
    table = open(VERDICTS, encoding="utf-8").read().strip()
    new, n = re.subn(r"<!-- VERDICTS-BEGIN -->.*?<!-- VERDICTS-END -->",
                     lambda _: f"<!-- VERDICTS-BEGIN -->\n{table}\n<!-- VERDICTS-END -->", text, flags=re.S)
    if n != 1:
        raise SystemExit("VERDICTS markers not found exactly once in REPORT.md")
    open(REPORT, "w", encoding="utf-8").write(new)
    print(f"embedded {table.count(chr(10))} table lines into {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
