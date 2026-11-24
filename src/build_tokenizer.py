"""Train and freeze a BPE tokenizer from seed corpora."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import iter_texts
from src.tokenizer import train_frozen_bpe


def collect_texts(paths: list[str]) -> list[str]:
    texts: list[str] = []
    for path in paths:
        texts.extend(iter_texts(path))
    if not texts:
        raise SystemExit("no training texts found for tokenizer")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen BPE tokenizer")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="jsonl/parquet paths used as tokenizer seed text",
    )
    parser.add_argument("--out", default="tokenizer", help="output directory")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    texts = collect_texts(args.inputs)
    # Repeat small corpora so BPE sees enough surface forms.
    if len(texts) < 200:
        texts = texts * (200 // max(len(texts), 1) + 1)

    tok = train_frozen_bpe(
        texts,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    out = Path(args.out)
    tok.save(out)
    print(f"wrote frozen tokenizer to {out}/ (vocab_size={tok.vocab_size})")


if __name__ == "__main__":
    main()
