"""Rejection-sampling RL-lite: keep verifier-passed trajectories, then SFT.

Method (lab-lite, one loop):
1. Score gold SFT rows with the verifier (should pass).
2. Sample K generations per prompt from a policy checkpoint.
3. Keep samples that pass the verifier.
4. Write winners JSONL for prompt-masked SFT.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from src.generate import Generator
from src.sft_data import load_sft_rows
from src.template import format_reasoning_prompt, format_response, parse_response
from src.verifier import score_response


def collect_winners(
    ckpt: str | Path,
    pool_path: str | Path,
    out_path: str | Path,
    k: int = 2,
    temperature: float = 0.8,
    max_new_tokens: int = 96,
    max_prompts: int | None = 128,
    seed: int = 42,
    include_gold: bool = True,
    device: str = "auto",
    batch_size: int = 8,
    allow_empty: bool = False,
) -> dict:
    if k < 1 or batch_size < 1:
        raise ValueError("k and batch_size must be positive")
    if max_prompts is not None and max_prompts < 1:
        raise ValueError("max_prompts must be positive")
    rows = load_sft_rows(pool_path)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_prompts is not None:
        rows = rows[:max_prompts]

    generator = Generator.from_checkpoint(ckpt, device)
    torch.manual_seed(seed)
    prompts = [format_reasoning_prompt(row["question"], generator.tokenizer) for row in rows]
    winners: list[dict] = []
    stats = {
        "prompts": len(rows),
        "gold_kept": 0,
        "samples": 0,
        "sample_passed": 0,
        "generated_kept": 0,
        "length_capped": 0,
    }

    if include_gold:
        for index, row in enumerate(rows):
            question = row["question"]
            gold = str(row["answer"])
            scratch = str(row["scratchpad"])
            prompt_id = row.get("id", f"row-{index}")

            gold_score = score_response(format_response(scratch, gold), gold_answer=gold)
            if gold_score["passed"]:
                winners.append(
                    {
                        "id": f"{prompt_id}-gold",
                        "prompt_id": prompt_id,
                        "question": question,
                        "scratchpad": scratch,
                        "answer": gold,
                        "text": prompts[index] + format_response(scratch, gold),
                        "source": "gold",
                        "reward": gold_score["reward"],
                    }
                )
                stats["gold_kept"] += 1

    seen: set[tuple[int, str, str]] = set()
    chosen: dict[int, str] = {}
    rejected: dict[int, str] = {}
    sample_count = len(rows) * k
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        jobs = [divmod(i, k) for i in range(start, stop)]
        completions = generator.generate(
            [prompts[row_i] for row_i, _ in jobs],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        for (row_i, sample_i), completion in zip(jobs, completions, strict=True):
            row = rows[row_i]
            prompt_id = row.get("id", f"row-{row_i}")
            stats["samples"] += 1
            if completion.finish_reason == "length":
                stats["length_capped"] += 1
                continue
            score = score_response(completion.text, gold_answer=str(row["answer"]))
            parsed = parse_response(completion.text)
            if not score["passed"] or parsed is None:
                if completion.text.strip():
                    rejected.setdefault(row_i, completion.text)
                continue
            stats["sample_passed"] += 1
            scratch, answer = parsed
            dedup_key = (row_i, scratch, answer)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            chosen.setdefault(row_i, completion.text)
            stats["generated_kept"] += 1
            winners.append(
                {
                    "id": f"{prompt_id}-rs{sample_i}",
                    "prompt_id": prompt_id,
                    "question": row["question"],
                    "scratchpad": scratch,
                    "answer": answer,
                    "text": prompts[row_i] + format_response(scratch, answer),
                    "source": "reject_sample",
                    "reward": score["reward"],
                }
            )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in winners:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Write sampled preference pairs for inspection.
    pairs_path = out_path.with_name(out_path.stem + "_pairs.jsonl")
    with pairs_path.open("w", encoding="utf-8") as f:
        for row_i in sorted(chosen.keys() & rejected.keys()):
            row = rows[row_i]
            f.write(
                json.dumps(
                    {
                        "id": row.get("id"),
                        "question": row["question"],
                        "chosen": chosen[row_i],
                        "rejected": rejected[row_i],
                        "method": "rejection_sampling_rllite",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        **stats,
        "winners": len(winners),
        "checkpoint": str(ckpt),
        "pool_path": str(pool_path),
        "out_path": str(out_path),
        "pairs_path": str(pairs_path),
        "seed": seed,
        "batch_size": batch_size,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "samples_per_prompt": k,
    }
    print(json.dumps(summary, indent=2))
    if not winners and not allow_empty:
        raise SystemExit("rejection sampling produced zero winners")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect verifier-passed RS winners")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--pool", default="data/sft/train.jsonl")
    parser.add_argument("--out", default="data/rllite/winners.jsonl")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-prompts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-gold", action="store_true")
    parser.add_argument("--allow-empty", action="store_true", help="allow zero winners")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    collect_winners(
        ckpt=args.ckpt,
        pool_path=args.pool,
        out_path=args.out,
        k=args.k,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_prompts=args.max_prompts,
        seed=args.seed,
        include_gold=not args.no_gold,
        device=args.device,
        batch_size=args.batch_size,
        allow_empty=args.allow_empty,
    )


if __name__ == "__main__":
    main()
