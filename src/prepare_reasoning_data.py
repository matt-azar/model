"""Prepare reproducible GSM8K splits for the single-GPU reasoning experiment."""

from __future__ import annotations

import argparse
import json
import random
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path

from src.manifest import load_manifest, refresh_hashes, save_manifest, verify_manifest_hashes


DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
DEFAULT_OUT = Path("data/reasoning")
SPLITS = ("train", "rollout", "dev", "test")
OFFICIAL_TRAIN_ROWS = 7473
OFFICIAL_TEST_ROWS = 1319


def _question_key(question: str) -> str:
    return " ".join(question.casefold().split())


def _convert(rows: Iterable[dict[str, str]], source_split: str) -> list[dict]:
    result = []
    for index, row in enumerate(rows):
        question = row["question"].strip()
        reasoning, separator, final = row["answer"].rpartition("####")
        reasoning = re.sub(r"<<[^<>]*>>", "", reasoning).strip()
        final = final.strip()
        if not (question and separator and reasoning and final):
            raise ValueError(f"malformed GSM8K {source_split} row {index}")
        result.append(
            {
                "id": f"gsm8k-{source_split}-{index}",
                "question": question,
                "scratchpad": reasoning,
                "answer": final,
            }
        )
    return result


def _source(revision: str | None, fixture: bool) -> dict:
    if fixture:
        return {"id": "local-fixture", "config": None, "revision": None, "license": None}
    if not _is_commit(revision):
        raise ValueError("GSM8K revision must be a full Hugging Face commit SHA")
    return {"id": DATASET_ID, "config": DATASET_CONFIG, "revision": revision, "license": "MIT"}


def _is_commit(revision: object) -> bool:
    return isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None


def _load_gsm8k(revision: str) -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, revision=revision)
    if len(dataset["train"]) != OFFICIAL_TRAIN_ROWS or len(dataset["test"]) != OFFICIAL_TEST_ROWS:
        raise ValueError("GSM8K split sizes differ from the official main subset")
    return _convert(dataset["train"], "train"), _convert(dataset["test"], "test")


def _fixture_splits() -> dict[str, list[dict]]:
    sizes = (16, 8, 4, 4)
    splits = {}
    offset = 0
    for split, size in zip(SPLITS, sizes):
        rows = []
        for a in range(offset + 1, offset + size + 1):
            b = a % 5 + 1
            rows.append(
                {
                    "id": f"fixture-{a}",
                    "question": f"Ada has {a} apples and buys {b} more. How many does she have?",
                    "scratchpad": f"Add the apples: {a} + {b} = {a + b}.",
                    "answer": str(a + b),
                }
            )
        splits[split] = rows
        offset += size
    return splits


def _split_training(train: list[dict], test: list[dict], seed: int) -> dict[str, list[dict]]:
    # Keep all official test rows. Remove exact question overlaps from the training pool.
    seen = {_question_key(row["question"]) for row in test}
    unique = []
    for row in train:
        key = _question_key(row["question"])
        if key not in seen:
            unique.append(row)
            seen.add(key)
    random.Random(seed).shuffle(unique)
    train_end = len(unique) * 80 // 100
    rollout_end = len(unique) * 95 // 100
    return {
        "train": unique[:train_end],
        "rollout": unique[train_end:rollout_end],
        "dev": unique[rollout_end:],
        "test": test,
    }


def validate(output: Path, *, fixture: bool = False) -> dict:
    """Open a prepared dataset and check provenance, split metadata and file hashes."""
    output = Path(output)
    manifest = load_manifest(output / "manifest.json")
    if manifest.get("version") != 1 or manifest.get("fixture") is not fixture:
        raise ValueError(f"wrong dataset version or kind in {output / 'manifest.json'}")
    source = manifest.get("source", {})
    if source != _source(source.get("revision"), fixture):
        raise ValueError(f"invalid dataset source in {output / 'manifest.json'}")
    entries = manifest.get("datasets", [])
    if [entry.get("name") for entry in entries] != list(SPLITS):
        raise ValueError(f"missing or invalid dataset splits under {output}")
    for entry in entries:
        if (
            entry.get("path") != f"{entry['name']}.jsonl"
            or not isinstance(entry.get("rows"), int)
            or entry["rows"] <= 0
            or not isinstance(entry.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
        ):
            raise ValueError(f"invalid manifest entry for {entry['name']}")
    if not fixture and entries[-1]["rows"] != OFFICIAL_TEST_ROWS:
        raise ValueError("prepared test split is missing official GSM8K test rows")
    errors = verify_manifest_hashes(manifest, output)
    if errors:
        raise ValueError("; ".join(errors))
    return manifest


def prepare(
    output: Path,
    *,
    fixture: bool = False,
    revision: str = GSM8K_REVISION,
    seed: int = 42,
) -> dict:
    """Publish the split files and manifest together. Existing valid outputs can be reused."""
    output = Path(output)
    if fixture and output.resolve() == DEFAULT_OUT.resolve():
        raise ValueError("fixture data needs a separate --out directory")
    source = _source(revision, fixture)
    if output.exists():
        manifest = validate(output, fixture=fixture)
        if manifest["source"] != source or manifest.get("seed") != seed:
            raise ValueError(f"existing data under {output} uses another source or split seed")
        return manifest

    if fixture:
        splits = _fixture_splits()
    else:
        train, test = _load_gsm8k(revision)
        splits = _split_training(train, test, seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        staging = Path(temp) / "data"
        staging.mkdir()
        entries = []
        for split, rows in splits.items():
            path = staging / f"{split}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as file:
                for row in rows:
                    file.write(json.dumps({**row, "split": split}, ensure_ascii=False) + "\n")
            entries.append({"name": split, "path": path.name, "rows": len(rows)})
        manifest = refresh_hashes(
            {"version": 1, "fixture": fixture, "source": source, "seed": seed, "datasets": entries},
            staging,
        )
        save_manifest(manifest, staging / "manifest.json")
        validate(staging, fixture=fixture)
        staging.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--revision", default=GSM8K_REVISION, help="full GSM8K repository commit SHA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixture", action="store_true", help="generate offline fixtures at a separate path")
    parser.add_argument("--verify", action="store_true", help="validate prepared files without downloading")
    args = parser.parse_args()
    try:
        if args.verify:
            manifest = validate(args.out, fixture=args.fixture)
        else:
            manifest = prepare(args.out, fixture=args.fixture, revision=args.revision, seed=args.seed)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"dataset preparation failed: {exc}\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
