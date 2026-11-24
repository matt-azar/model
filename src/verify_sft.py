"""Phase 3 exit checks: CoT format on smoke + GSM8K-style beats Phase 2 base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval_gsm8k_style import evaluate
from src.generate import generate_text
from src.template import has_scratchpad_and_final


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_format_smoke(
    ckpt: Path,
    smoke_path: Path,
    max_new_tokens: int,
    min_format_rate: float,
) -> dict:
    rows = load_jsonl(smoke_path)
    format_cases = [r for r in rows if r.get("expect_scratchpad")]
    if not format_cases:
        raise SystemExit(f"no expect_scratchpad cases in {smoke_path}")

    results = []
    ok = 0
    answer_ok = 0
    for case in format_cases:
        gen = generate_text(
            ckpt,
            prompt=case["prompt"],
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        fmt = has_scratchpad_and_final(gen)
        expected = str(case.get("expected", ""))
        ans = expected.lower() in gen.lower() if expected else True
        if fmt:
            ok += 1
        if ans:
            answer_ok += 1
        results.append(
            {
                "id": case.get("id"),
                "format_ok": fmt,
                "answer_ok": ans,
                "generation": gen,
            }
        )

    rate = ok / len(format_cases)
    report = {
        "n": len(format_cases),
        "format_rate": rate,
        "answer_rate": answer_ok / len(format_cases),
        "min_format_rate": min_format_rate,
        "passed": rate >= min_format_rate,
        "results": results,
    }
    if not report["passed"]:
        raise SystemExit(f"format smoke failed: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Phase 3 SFT exit criteria")
    parser.add_argument("--base-ckpt", required=True, help="Phase 2-style base checkpoint")
    parser.add_argument("--sft-ckpt", required=True, help="SFT checkpoint")
    parser.add_argument("--eval", default="data/sft/gsm8k_style_eval.jsonl")
    parser.add_argument("--smoke", default="data/personal_smoke.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--min-format-rate", type=float, default=0.8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    format_report = check_format_smoke(
        Path(args.sft_ckpt),
        Path(args.smoke),
        max_new_tokens=args.max_new_tokens,
        min_format_rate=args.min_format_rate,
    )

    base_eval = evaluate(
        args.base_ckpt,
        args.eval,
        max_new_tokens=args.max_new_tokens,
    )
    sft_eval = evaluate(
        args.sft_ckpt,
        args.eval,
        max_new_tokens=args.max_new_tokens,
    )

    beat = sft_eval["exact_match"] > base_eval["exact_match"]
    report = {
        "format_smoke": {
            k: v for k, v in format_report.items() if k != "results"
        },
        "format_results": format_report["results"],
        "base_gsm8k_style": {
            k: v for k, v in base_eval.items() if k != "results"
        },
        "sft_gsm8k_style": {
            k: v for k, v in sft_eval.items() if k != "results"
        },
        "sft_beats_base": beat,
        "passed": format_report["passed"] and beat,
    }

    print(
        json.dumps(
            {
                "format_smoke": report["format_smoke"],
                "base_gsm8k_style": report["base_gsm8k_style"],
                "sft_gsm8k_style": report["sft_gsm8k_style"],
                "sft_beats_base": beat,
                "passed": report["passed"],
            },
            indent=2,
        )
    )
    for row in format_report["results"]:
        flag = "PASS" if row["format_ok"] else "FAIL"
        print(f"[{flag}] {row.get('id')}: {row['generation']!r}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")

    if not beat:
        raise SystemExit(
            f"SFT did not beat base: "
            f"base={base_eval['exact_match']:.3f} sft={sft_eval['exact_match']:.3f}"
        )
    print("PHASE3 VERIFY OK")


if __name__ == "__main__":
    main()
