"""Tiny GSM8K-style exact-match eval on curated local JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from src.sft_data import load_sft_rows
from src.template import format_reasoning_prompt, has_scratchpad_and_final
from src.verifier import extract_eval_answer, normalize_answer

if TYPE_CHECKING:
    from src.generate import Generator


def evaluate(
    ckpt: str | Path,
    eval_path: str | Path,
    max_new_tokens: int = 96,
    temperature: float = 0.0,
    device: str = "auto",
    *,
    batch_size: int = 8,
    include_results: bool = True,
    generator: Generator | None = None,
) -> dict:
    """Score completion suffixes, tracking answer, format, and token caps separately."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = load_sft_rows(eval_path)
    if generator is None:
        from src.generate import Generator

        generator = Generator.from_checkpoint(ckpt, device_name=device)

    results = []
    n_correct = n_format_ok = n_truncated = total_generated_tokens = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        completions = generator.generate(
            [format_reasoning_prompt(row["question"], generator.tokenizer) for row in batch],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        for row, completion in zip(batch, completions, strict=True):
            pred = extract_eval_answer(completion.text)
            gold = str(row["answer"])
            truncated = completion.finish_reason == "length"
            # A capped completion can end midway through a number.
            match = (
                not truncated
                and pred is not None
                and normalize_answer(pred) == normalize_answer(gold)
            )
            format_ok = has_scratchpad_and_final(completion.text)
            n_correct += match
            n_format_ok += format_ok
            n_truncated += truncated
            total_generated_tokens += completion.tokens
            if include_results:
                results.append(
                    {
                        "id": row.get("id"),
                        "question": row["question"],
                        "gold": gold,
                        "pred": pred,
                        "match": match,
                        "format_ok": format_ok,
                        "finish_reason": completion.finish_reason,
                        "generated_tokens": completion.tokens,
                        "generation": completion.text,
                    }
                )

    n = len(rows)
    return {
        "ckpt": str(ckpt),
        "eval_path": str(eval_path),
        "n": n,
        "exact_match": n_correct / n,
        "format_rate": n_format_ok / n,
        "truncation_rate": n_truncated / n,
        "n_correct": n_correct,
        "n_format_ok": n_format_ok,
        "n_truncated": n_truncated,
        "total_generated_tokens": total_generated_tokens,
        "mean_generated_tokens": total_generated_tokens / n,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GSM8K-style local eval")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--eval", default="data/sft/gsm8k_style_eval.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = evaluate(
        args.ckpt,
        args.eval,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
        batch_size=args.batch_size,
        include_results=args.out is not None,
    )
    summary = {k: v for k, v in report.items() if k != "results"}
    print(json.dumps(summary, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
