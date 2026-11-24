"""Build a clearly marked Phase 1 toy stand-in pretrain mix.

This is NOT a real web crawl. It is synthetic CC0 text authored for pipeline
tests so we can stream >1B tokens without babysitting downloads.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from src.data import train_val_split, write_jsonl, write_parquet


TEMPLATES = [
    "The capital of France is Paris. Paris is known for the Eiffel Tower.",
    "Two plus two equals four. Show your work: 2 + 2 = 4. Final answer: 4",
    "def add(a, b):\n    return a + b\nassert add(2, 2) == 4",
    "Reasoning: first list the given facts, then derive the next step carefully.",
    "Code comment: prefer clear names, small functions, and verified tests.",
    "Math drill: {a} + {b} = {c}. Keep the scratchpad short and exact.",
    "Science note: water freezes at 0 C and boils at 100 C at sea level.",
    "Logic: if all A are B, and x is A, then x is B.",
]


def synth_docs(n: int, seed: int = 42) -> list[str]:
    docs: list[str] = []
    for i in range(n):
        template = TEMPLATES[i % len(TEMPLATES)]
        a = (i * 3 + seed) % 50
        b = (i * 7 + 11) % 50
        text = template.format(a=a, b=b, c=a + b)
        # Stable unique suffix keeps docs distinct under hashing/splits.
        digest = hashlib.sha1(f"{seed}:{i}:{text}".encode()).hexdigest()[:12]
        docs.append(f"[stand_in doc={i} id={digest}] {text}")
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 1 stand-in corpus")
    parser.add_argument("--out-dir", default="data/standin")
    parser.add_argument("--n-docs", type=int, default=2000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--also-parquet", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "STANDIN.md").write_text(
        "\n".join(
            [
                "# Toy stand-in corpus",
                "",
                "Clearly marked Phase 1 stand-in. Synthetic CC0 text for streaming,",
                "packing, train/val split, and manifest hashing tests.",
                "",
                "Not a substitute for a license-clean real pretrain mix.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    docs = synth_docs(args.n_docs, seed=args.seed)
    train, val = train_val_split(docs, val_ratio=args.val_ratio, seed=args.seed)
    extra = {"stand_in": True, "license": "CC0-1.0"}
    write_jsonl(train, out / "train.jsonl", extra=extra)
    write_jsonl(val, out / "val.jsonl", extra=extra)
    if args.also_parquet:
        write_parquet(train, out / "train.parquet")
        write_parquet(val, out / "val.parquet")

    print(
        f"wrote stand-in corpus to {out}/ "
        f"(train={len(train)} val={len(val)} parquet={args.also_parquet})"
    )


if __name__ == "__main__":
    main()
