"""Compare two harness report JSON files canonically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.report import reports_equal


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare eval reports")
    parser.add_argument("report_a")
    parser.add_argument("report_b")
    args = parser.parse_args()
    a = json.loads(Path(args.report_a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.report_b).read_text(encoding="utf-8"))
    ok, msg = reports_equal(a, b)
    if not ok:
        raise SystemExit(msg or "reports differ")
    print("CANONICAL REPORTS EQUAL")


if __name__ == "__main__":
    main()
