"""Score mixed-domain reasoning checkpoints on held-out prompts."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from src.generate import Generator
from src.sft_data import load_sft_rows
from src.template import format_chat_prompt
from src.verifier import extract_eval_answer, normalize_answer


def matches(text: str, answer: str, scorer: str) -> bool:
    text = text.strip()
    # Only strip a wrapper around the entire completion.
    fenced = re.fullmatch(r'```(?:python|json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if fenced:
        text = fenced[1].strip()
    if scorer == "number":
        guess = extract_eval_answer(text)
        return guess is not None and normalize_answer(guess) == normalize_answer(answer)
    if scorer == "choice":
        letter = re.fullmatch(r"(?:\(\s*([A-Z])\s*\)|([A-Z]))\.?", text, re.IGNORECASE)
        return bool(letter and (letter[1] or letter[2]).casefold() == answer.strip("()").casefold())
    if scorer == "boolean":
        return text.rstrip(".").casefold() == answer.casefold()
    if scorer == "comma_list":
        return [item.strip() for item in text.split(",")] == [item.strip() for item in answer.split(",")]
    if scorer == "text":
        return text == answer
    if scorer in {"literal", "json"}:
        # Compare data values without executing any generated code.
        if len(text) > 2048:
            return False
        parse = ast.literal_eval if scorer == "literal" else json.loads
        try:
            actual, expected = parse(text), parse(answer)
        except (ValueError, SyntaxError, TypeError, RecursionError):
            return False
        return type(actual) is type(expected) and actual == expected
    raise ValueError(f"unsupported scorer: {scorer}")


def evaluate(
    model: Path,
    data: Path,
    *,
    device: str,
    batch_size: int,
    max_new_tokens: int,
    per_domain_limit: int | None = None,
) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in load_sft_rows(data):
        groups.setdefault(row["domain"], []).append(row)

    generator = Generator.from_checkpoint(model, device)
    results = []
    metrics = {}
    for domain, rows in sorted(groups.items()):
        rows = rows[:per_domain_limit]
        n_correct = n_truncated = generated_tokens = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            output = generator.generate(
                [format_chat_prompt(row["question"], generator.tokenizer) for row in batch],
                max_new_tokens=max_new_tokens, temperature=0.0,
            )
            for row, completion in zip(batch, output, strict=True):
                truncated = completion.finish_reason == "length"
                correct = not truncated and matches(completion.text, row["answer"], row["scorer"])
                n_correct += correct
                n_truncated += truncated
                generated_tokens += completion.tokens
                results.append({
                    "id": row["id"], "domain": domain, "question": row["question"],
                    "reference": row["answer"], "generation": completion.text, "correct": correct,
                    "finish_reason": completion.finish_reason, "generated_tokens": completion.tokens,
                })
            if (start // batch_size + 1) % 16 == 0:
                print(f"{domain}: evaluated {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
        metrics[domain] = {
            "accuracy": n_correct / len(rows), "correct": n_correct, "examples": len(rows),
            "truncated": n_truncated, "mean_generated_tokens": generated_tokens / len(rows),
        }
        print(f"{domain}: {n_correct}/{len(rows)} correct, {n_truncated} length-capped", flush=True)

    return {
        "n": len(results), "domains": metrics, "results": results,
        "checkpoint": str(model), "data": str(data),
        "batch_size": batch_size, "max_new_tokens": max_new_tokens,
    }
