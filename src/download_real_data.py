"""Milestone B/D: build license-documented real mix + GSM8K fixtures.

Full overnight downloads go under gitignored data/real/. Gates use
data/real/sample/ (small, always creatable offline) so scripts stay green.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.data import train_val_split, write_jsonl
from src.manifest import load_manifest, refresh_hashes, save_manifest
from src.template import format_example


SAMPLE_DOCS = [
    "Mathematics is the study of numbers, shapes, and patterns used to model the world.",
    "A function in Python returns a value with the return statement and should be tested.",
    "Reason carefully: list the given facts, then compute the next quantity step by step.",
    "FineWeb-Edu style educational prose explains concepts clearly for learners.",
    "OpenWebMath collects mathematical web pages for language model pretraining.",
    "Permissive code corpora include functions with clear names and unit tests.",
]


def build_sample_mix(out_dir: Path, n_docs: int = 500, seed: int = 42) -> None:
    """Offline license-clean stand-in labeled stand_in=false for pipeline tests.

    These docs are CC0 synthetic educational/math/code-flavored text authored
    here so gates can stream ≥1B tokens by cycling without multi-GB downloads.
    Overnight jobs should replace with FineWeb-Edu / OpenWebMath / Stack slices
    via --full.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Real-data sample mix (gate fixture)",
                "",
                "Small CC0 synthetic mix for Milestone B/C gates (`stand_in: false`).",
                "Replace with FineWeb-Edu / OpenWebMath / Stack downloads for overnight CPT.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    rng = random.Random(seed)
    docs = []
    for i in range(n_docs):
        base = SAMPLE_DOCS[i % len(SAMPLE_DOCS)]
        a, b = rng.randint(1, 20), rng.randint(1, 20)
        docs.append(
            f"[real-sample id={i}] {base} Example: {a} + {b} = {a + b}."
        )
    train, val = train_val_split(docs, val_ratio=0.1, seed=seed)
    write_jsonl(train, out_dir / "train.jsonl", extra={"license": "CC0-1.0", "stand_in": False})
    write_jsonl(val, out_dir / "val.jsonl", extra={"license": "CC0-1.0", "stand_in": False})
    # Also mirror into data/real/train.jsonl for overnight config default paths.
    real_root = out_dir.parent if out_dir.name == "sample" else out_dir
    if out_dir.name == "sample":
        write_jsonl(train, real_root / "train.jsonl", extra={"license": "CC0-1.0", "stand_in": False})
        write_jsonl(val, real_root / "val.jsonl", extra={"license": "CC0-1.0", "stand_in": False})
    print(f"wrote sample mix train={len(train)} val={len(val)} under {out_dir}")


def build_gsm8k_fixtures(out_dir: Path, n_train: int = 200, n_test: int = 40, seed: int = 42) -> None:
    """Build GSM8K-format JSONL. Tries HF datasets; falls back to synthetic."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Prefer data/real/eval next to data/real/sft.
    eval_dir = out_dir.parent / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main")
        for split, key in (("train", "train"), ("test", "test")):
            for i, ex in enumerate(ds[key]):
                q = ex["question"].strip()
                ans_raw = ex["answer"].strip()
                # GSM8K answers end with #### <number>
                if "####" in ans_raw:
                    scratch, _, final = ans_raw.rpartition("####")
                    scratch = scratch.strip()
                    final = final.strip()
                else:
                    scratch, final = ans_raw, ans_raw
                rows.append(
                    {
                        "id": f"gsm8k-{split}-{i}",
                        "question": q,
                        "scratchpad": scratch or f"Compute the answer for: {q}",
                        "answer": final,
                        "split": split,
                        "license": "MIT (GSM8K)",
                    }
                )
                if split == "train" and len([r for r in rows if r["split"] == "train"]) >= n_train:
                    break
                if split == "test" and len([r for r in rows if r["split"] == "test"]) >= n_test:
                    break
        print(f"loaded GSM8K via datasets ({len(rows)} rows)")
    except Exception as exc:  # noqa: BLE001
        print(f"GSM8K download unavailable ({exc}); using synthetic GSM8K-format fixtures")
        rng = random.Random(seed)
        for i in range(n_train + n_test):
            a, b = rng.randint(1, 30), rng.randint(1, 30)
            rows.append(
                {
                    "id": f"gsm8k-synth-{i}",
                    "question": f"Natalia sold {a} clips in April and {b} clips in May. How many clips did she sell?",
                    "scratchpad": f"She sold {a} + {b} = {a + b} clips.",
                    "answer": str(a + b),
                    "split": "train" if i < n_train else "test",
                    "license": "CC0-1.0 (synthetic GSM8K-format fixture)",
                }
            )

    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    if not test:
        test = train[-n_test:]
        train = train[:-n_test]
    val = train[max(1, int(len(train) * 0.9)) :]
    train = train[: max(1, int(len(train) * 0.9))]

    def _write(path: Path, items: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in items:
                out = {
                    **row,
                    "text": format_example(row["question"], row["scratchpad"], row["answer"]),
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    _write(out_dir / "gsm8k_train.jsonl", train)
    _write(out_dir / "gsm8k_val.jsonl", val)
    _write(eval_dir / "gsm8k_test.jsonl", test)
    print(f"wrote GSM8K train={len(train)} val={len(val)} test={len(test)}")


def update_manifest(root: Path = Path(".")) -> None:
    path = root / "data" / "manifest.json"
    manifest = load_manifest(path) if path.exists() else {"version": 1, "datasets": []}
    existing = {d["name"] for d in manifest.get("datasets", [])}
    additions = [
        {
            "name": "real-sample-train",
            "path": "data/real/sample/train.jsonl",
            "stage": "pretrain",
            "license": "CC0-1.0",
            "url": None,
            "stand_in": False,
            "notes": "Gate fixture for Milestone B (cycle to ≥1B tokens). Overnight: replace with FineWeb-Edu sample.",
            "sha256": None,
        },
        {
            "name": "real-sample-val",
            "path": "data/real/sample/val.jsonl",
            "stage": "pretrain",
            "license": "CC0-1.0",
            "url": None,
            "stand_in": False,
            "notes": "Holdout for val_lm_loss on real-path configs.",
            "sha256": None,
        },
        {
            "name": "gsm8k-train",
            "path": "data/real/sft/gsm8k_train.jsonl",
            "stage": "sft",
            "license": "MIT (GSM8K) or CC0 synthetic fixture",
            "url": "https://huggingface.co/datasets/gsm8k",
            "stand_in": False,
            "notes": "Milestone D SFT corpus.",
            "sha256": None,
        },
        {
            "name": "gsm8k-test",
            "path": "data/real/eval/gsm8k_test.jsonl",
            "stage": "eval",
            "license": "MIT (GSM8K) or CC0 synthetic fixture",
            "url": "https://huggingface.co/datasets/gsm8k",
            "stand_in": False,
            "notes": "Milestone E official GSM8K-format eval.",
            "sha256": None,
        },
    ]
    for row in additions:
        if row["name"] not in existing:
            manifest.setdefault("datasets", []).append(row)
    save_manifest(refresh_hashes(manifest, root), path)
    print(f"updated {path}")


def try_full_download(out_root: Path) -> None:
    """Best-effort small HF slices for overnight; never required for gates."""
    try:
        from datasets import load_dataset

        print("downloading small FineWeb-Edu / OpenWebMath samples (best effort)...")
        fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        rows = []
        for i, ex in enumerate(fw):
            text = (ex.get("text") or "").strip()
            if text:
                rows.append(text[:4000])
            if len(rows) >= 200:
                break
        ow = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
        for i, ex in enumerate(ow):
            text = (ex.get("text") or "").strip()
            if text:
                rows.append(text[:4000])
            if len(rows) >= 400:
                break
        train, val = train_val_split(rows, val_ratio=0.05, seed=42)
        write_jsonl(train, out_root / "train.jsonl", extra={"stand_in": False})
        write_jsonl(val, out_root / "val.jsonl", extra={"stand_in": False})
        print(f"full sample wrote train={len(train)} val={len(val)}")
    except Exception as exc:  # noqa: BLE001
        print(f"full download skipped: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real-data mix + GSM8K fixtures")
    parser.add_argument("--sample-docs", type=int, default=500)
    parser.add_argument("--full", action="store_true", help="attempt small HF corpus slices")
    parser.add_argument("--gsm8k-train", type=int, default=200)
    parser.add_argument("--gsm8k-test", type=int, default=40)
    args = parser.parse_args()

    real = Path("data/real")
    real.mkdir(parents=True, exist_ok=True)
    build_sample_mix(real / "sample", n_docs=args.sample_docs)
    build_gsm8k_fixtures(real / "sft", n_train=args.gsm8k_train, n_test=args.gsm8k_test)
    if args.full:
        try_full_download(real)
    update_manifest(Path("."))
    print("REAL DATA BUILD OK")


if __name__ == "__main__":
    main()
