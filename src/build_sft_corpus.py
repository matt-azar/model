"""Build a curated CC0 GSM8K-style SFT corpus with explicit CoT."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.template import format_example


def _split_rows(
    rows: list[dict], val_ratio: float, seed: int
) -> tuple[list[dict], list[dict]]:
    items = list(rows)
    rng = random.Random(seed)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio))
    if n_val >= len(items):
        n_val = 1
    return items[n_val:], items[:n_val]


def _examples() -> list[dict]:
    """Hand-authored arithmetic word problems with visible scratchpads."""
    rows: list[dict] = []

    # Fixed curated seeds.
    curated = [
        {
            "id": "apples-3-5",
            "question": "Sam has 3 apples and buys 5 more. How many apples does Sam have?",
            "scratchpad": "Start with 3. Add 5. 3 + 5 = 8.",
            "answer": "8",
        },
        {
            "id": "buses-4-2",
            "question": "There are 4 buses with 2 wheels shown on a toy each. How many wheels in total?",
            "scratchpad": "4 buses * 2 wheels = 8.",
            "answer": "8",
        },
        {
            "id": "two-plus-two",
            "question": "What is two plus two?",
            "scratchpad": "2 + 2 = 4.",
            "answer": "4",
        },
        {
            "id": "dozen-eggs",
            "question": "A carton holds 12 eggs. If 3 are used, how many remain?",
            "scratchpad": "12 - 3 = 9.",
            "answer": "9",
        },
        {
            "id": "pages-20-5",
            "question": "Maya read 20 pages yesterday and 5 today. How many pages total?",
            "scratchpad": "20 + 5 = 25.",
            "answer": "25",
        },
        {
            "id": "stickers-15-7",
            "question": "Leo had 15 stickers and gave away 7. How many does he have left?",
            "scratchpad": "15 - 7 = 8.",
            "answer": "8",
        },
        {
            "id": "boxes-6-4",
            "question": "Each box has 4 pencils. How many pencils in 6 boxes?",
            "scratchpad": "6 * 4 = 24.",
            "answer": "24",
        },
        {
            "id": "share-18-3",
            "question": "18 candies are shared equally among 3 kids. How many each?",
            "scratchpad": "18 / 3 = 6.",
            "answer": "6",
        },
    ]
    rows.extend(curated)

    # Synthetic variants for packing / overfitting capacity on a tiny model.
    for a in range(1, 21):
        for b in range(1, 11):
            rows.append(
                {
                    "id": f"add-{a}-{b}",
                    "question": f"What is {a} plus {b}?",
                    "scratchpad": f"{a} + {b} = {a + b}.",
                    "answer": str(a + b),
                }
            )
            if a > b:
                rows.append(
                    {
                        "id": f"sub-{a}-{b}",
                        "question": f"What is {a} minus {b}?",
                        "scratchpad": f"{a} - {b} = {a - b}.",
                        "answer": str(a - b),
                    }
                )
            rows.append(
                {
                    "id": f"word-add-{a}-{b}",
                    "question": (
                        f"A jar has {a} marbles. Someone adds {b} more. "
                        f"How many marbles are in the jar now?"
                    ),
                    "scratchpad": f"Start with {a}. Add {b}. {a} + {b} = {a + b}.",
                    "answer": str(a + b),
                }
            )
    return rows


def write_split(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            text = format_example(row["question"], row["scratchpad"], row["answer"])
            out = {
                "id": row["id"],
                "question": row["question"],
                "scratchpad": row["scratchpad"],
                "answer": row["answer"],
                "text": text,
                "license": "CC0-1.0",
                "stand_in": False,
                "stage": "sft",
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build curated SFT CoT corpus")
    parser.add_argument("--out-dir", default="data/sft")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-n", type=int, default=40, help="held-out GSM8K-style eval size")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(
        "\n".join(
            [
                "# Curated SFT corpus (Phase 3)",
                "",
                "Hand-authored + synthetic arithmetic word problems with explicit",
                "scratchpads and `Final answer:` lines. License: CC0-1.0.",
                "Not the real GSM8K dataset; GSM8K-style for local scoring only.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    all_rows = _examples()
    rng = random.Random(args.seed)
    rng.shuffle(all_rows)

    eval_n = min(args.eval_n, max(1, len(all_rows) // 5))
    eval_rows = all_rows[:eval_n]
    remain = all_rows[eval_n:]
    train_rows, val_rows = _split_rows(remain, val_ratio=args.val_ratio, seed=args.seed)
    write_split(train_rows, out / "train.jsonl")
    write_split(val_rows, out / "val.jsonl")
    write_split(eval_rows, out / "gsm8k_style_eval.jsonl")

    # Prompt-only smoke for format compliance.
    smoke_path = Path("data/personal_smoke.jsonl")
    smoke = [
        {
            "id": "format-apples",
            "prompt": "### Instruction\nSam has 3 apples and buys 5 more. How many apples does Sam have?\n\n### Response\n",
            "expected": "Final answer: 8",
            "expect_scratchpad": True,
        },
        {
            "id": "format-two-plus-two",
            "prompt": "### Instruction\nWhat is two plus two?\n\n### Response\n",
            "expected": "Final answer: 4",
            "expect_scratchpad": True,
        },
        {
            "id": "format-minus",
            "prompt": "### Instruction\nWhat is 15 minus 7?\n\n### Response\n",
            "expected": "Final answer: 8",
            "expect_scratchpad": True,
        },
        {
            "id": "capital-france",
            "prompt": "The capital of France is",
            "expected": "Paris",
            "expect_scratchpad": False,
        },
    ]
    with smoke_path.open("w", encoding="utf-8") as f:
        for row in smoke:
            f.write(json.dumps(row) + "\n")

    print(
        f"wrote SFT corpus to {out}/ "
        f"(train={len(train_rows)} val={len(val_rows)} eval={len(eval_rows)})"
    )
    print(f"updated {smoke_path}")


if __name__ == "__main__":
    main()
