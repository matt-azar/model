"""Tiny GSM8K-style exact-match eval on curated local JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.generate import generate_text
from src.sft_data import load_sft_rows
from src.template import extract_final_answer, format_prompt, has_scratchpad_and_final


def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    # Strip trailing punctuation / common wrappers.
    while s and s[-1] in ".;:":
        s = s[:-1]
    if s.endswith(".0") and s[:-2].replace("-", "").isdigit():
        s = s[:-2]
    return s.strip()


def evaluate(
    ckpt: str | Path,
    eval_path: str | Path,
    max_new_tokens: int = 96,
    temperature: float = 0.0,
    device: str = "auto",
) -> dict:
    rows = load_sft_rows(eval_path)
    results = []
    correct = 0
    format_ok = 0
    for row in rows:
        prompt = format_prompt(row["question"])
        generation = generate_text(
            ckpt,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device_name=device,
        )
        # Generation includes the prompt prefix from decode of full sequence.
        # Score on the suffix after the prompt when present.
        scored_text = generation
        if prompt in generation:
            scored_text = generation[len(prompt) :]
        elif "### Response" in generation:
            scored_text = generation.split("### Response", 1)[-1]

        pred = extract_final_answer(scored_text) or extract_final_answer(generation)
        gold = str(row["answer"])
        match = (
            pred is not None
            and normalize_answer(pred) == normalize_answer(gold)
        )
        fmt = has_scratchpad_and_final(scored_text) or has_scratchpad_and_final(
            generation
        )
        if match:
            correct += 1
        if fmt:
            format_ok += 1
        results.append(
            {
                "id": row.get("id"),
                "question": row["question"],
                "gold": gold,
                "pred": pred,
                "match": match,
                "format_ok": fmt,
                "generation": generation,
            }
        )

    n = len(rows)
    report = {
        "ckpt": str(ckpt),
        "eval_path": str(eval_path),
        "n": n,
        "exact_match": correct / n if n else 0.0,
        "format_rate": format_ok / n if n else 0.0,
        "n_correct": correct,
        "n_format_ok": format_ok,
        "results": results,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="GSM8K-style local eval")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--eval", default="data/sft/gsm8k_style_eval.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = evaluate(
        args.ckpt,
        args.eval,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
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
