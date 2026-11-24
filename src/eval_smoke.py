"""Personal smoke eval: JSONL prompts with optional exact-match expectations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.generate import generate_text


def load_cases(path: str | Path) -> list[dict]:
    cases: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def normalize(s: str) -> str:
    return " ".join(s.strip().lower().split())


def contains_expected(generation: str, expected: str) -> bool:
    return normalize(expected) in normalize(generation)


def run_eval(
    ckpt: str,
    eval_path: str,
    max_new_tokens: int = 48,
    temperature: float = 0.0,
    device: str = "auto",
    require_any_match: bool = True,
) -> dict:
    cases = load_cases(eval_path)
    results = []
    matched = 0
    scored = 0
    for case in cases:
        prompt = case["prompt"]
        expected = case.get("expected")
        generation = generate_text(
            ckpt,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device_name=device,
        )
        ok = None
        if expected is not None:
            scored += 1
            ok = contains_expected(generation, str(expected))
            if ok:
                matched += 1
        results.append(
            {
                "id": case.get("id"),
                "prompt": prompt,
                "expected": expected,
                "generation": generation,
                "match": ok,
            }
        )

    report = {
        "ckpt": ckpt,
        "n_cases": len(cases),
        "n_scored": scored,
        "n_matched": matched,
        "exact_substring_acc": (matched / scored) if scored else None,
        "results": results,
    }

    if require_any_match and scored > 0 and matched < 1:
        raise SystemExit(
            "smoke eval failed: expected at least one substring match on "
            f"{eval_path} (matched={matched}/{scored})"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run personal smoke eval")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--eval", default="data/personal_smoke.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None, help="optional JSON report path")
    parser.add_argument(
        "--allow-zero-match",
        action="store_true",
        help="do not fail when zero expected substrings match",
    )
    args = parser.parse_args()

    report = run_eval(
        args.ckpt,
        args.eval,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
        require_any_match=not args.allow_zero_match,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    for row in report["results"]:
        flag = {True: "PASS", False: "FAIL", None: "n/a"}[row["match"]]
        print(f"[{flag}] {row.get('id')}: {row['generation']!r}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
