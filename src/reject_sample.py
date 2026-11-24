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

from src.generate import generate_text
from src.sft_data import load_sft_rows
from src.template import format_example, format_prompt, format_response
from src.verifier import score_response


def _scratch_from_generation(text: str, prompt: str) -> str:
    body = text
    if prompt in body:
        body = body[len(prompt) :]
    elif "### Response" in body:
        body = body.split("### Response", 1)[-1]
    # Prefer content between Show your work and Final answer.
    lower = body.lower()
    start = lower.find("show your work:")
    end = lower.rfind("final answer:")
    if start >= 0 and end > start:
        return body[start + len("Show your work:") : end].strip()
    return body.strip()


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
) -> dict:
    rows = load_sft_rows(pool_path)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_prompts is not None:
        rows = rows[:max_prompts]

    winners: list[dict] = []
    stats = {
        "prompts": len(rows),
        "gold_kept": 0,
        "samples": 0,
        "sample_passed": 0,
    }

    for row in rows:
        question = row["question"]
        gold = str(row["answer"])
        scratch = str(row["scratchpad"])

        if include_gold:
            gold_text = format_example(question, scratch, gold)
            gold_score = score_response(gold_text, gold_answer=gold, require_format=True)
            if gold_score["passed"]:
                winners.append(
                    {
                        "id": f"{row.get('id', 'row')}-gold",
                        "question": question,
                        "scratchpad": scratch,
                        "answer": gold,
                        "text": gold_text,
                        "source": "gold",
                        "reward": gold_score["reward"],
                    }
                )
                stats["gold_kept"] += 1

        prompt = format_prompt(question)
        for sample_i in range(k):
            # Vary temperature slightly for diversity while staying reproducible.
            temp = temperature if sample_i > 0 else max(temperature, 0.7)
            gen = generate_text(
                ckpt,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                device_name=device,
            )
            stats["samples"] += 1
            scored_text = gen
            if prompt in gen:
                scored_text = gen[len(prompt) :]
            score = score_response(scored_text, gold_answer=gold, require_format=True)
            if not score["passed"]:
                # Also try full generation text.
                score = score_response(gen, gold_answer=gold, require_format=True)
            if score["passed"]:
                stats["sample_passed"] += 1
                pred = score["pred"] or gold
                gen_scratch = _scratch_from_generation(gen, prompt) or scratch
                winners.append(
                    {
                        "id": f"{row.get('id', 'row')}-rs{sample_i}",
                        "question": question,
                        "scratchpad": gen_scratch,
                        "answer": pred,
                        "text": format_example(question, gen_scratch, pred),
                        "source": "reject_sample",
                        "reward": score["reward"],
                    }
                )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in winners:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Also write preference pairs (chosen=winner gold/response, rejected=bad) for inspection.
    pairs_path = out_path.with_name(out_path.stem + "_pairs.jsonl")
    with pairs_path.open("w", encoding="utf-8") as f:
        for row in rows:
            chosen = format_response(row["scratchpad"], row["answer"])
            rejected = "Final answer: "  # empty CoT / missing scratchpad
            f.write(
                json.dumps(
                    {
                        "id": row.get("id"),
                        "question": row["question"],
                        "chosen": chosen,
                        "rejected": rejected,
                        "method": "rejection_sampling_rllite",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        **stats,
        "winners": len(winners),
        "out_path": str(out_path),
        "pairs_path": str(pairs_path),
    }
    print(json.dumps(summary, indent=2))
    if not winners:
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
    )


if __name__ == "__main__":
    main()
