"""CLI: python -m src.eval --ckpt ... --config configs/eval.yaml --out runs/eval"""

from __future__ import annotations

import argparse
import json

from src.eval.harness import run_harness
from src.eval.report import reports_equal


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified eval harness v1")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--out", default="runs/eval")
    parser.add_argument("--stem", default="report")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--compare",
        default=None,
        help="optional second report JSON to compare canonical equality against",
    )
    args = parser.parse_args()

    report = run_harness(
        ckpt=args.ckpt,
        config_path=args.config,
        out_dir=args.out,
        stem=args.stem,
        device_override=args.device,
    )
    summary = {
        "git_sha": report["git_sha"],
        "ckpt_sha256": report["ckpt_sha256"],
        "benchmarks": {
            k: v["summary"] for k, v in report["benchmarks"].items()
        },
    }
    print(json.dumps(summary, indent=2))

    if args.compare:
        other = json.loads(open(args.compare, encoding="utf-8").read())
        ok, msg = reports_equal(report, other)
        if not ok:
            raise SystemExit(msg)
        print("CANONICAL REPORTS EQUAL")


if __name__ == "__main__":
    main()
