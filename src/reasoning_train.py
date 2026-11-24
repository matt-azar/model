"""Single-GPU reasoning SFT, optional verified self-training, and evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import torch
import yaml
from huggingface_hub import HfApi, snapshot_download

from src.eval_gsm8k_style import evaluate
from src.prepare_reasoning_data import prepare, validate
from src.reject_sample import collect_winners
from src.train import train

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def write_json(path: Path, data: dict) -> None:
    staged = path.with_suffix(".tmp")
    staged.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staged.replace(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_smoke_model(path: Path, seed: int) -> None:
    """Local tokenizer and random causal LM to test the real HF code path offline."""
    if (path / "model.safetensors").is_file():
        return

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    torch.manual_seed(seed)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["### Instruction\nCalculate the answer.\n\n### Response\nShow your work:\n1 + 2 = 3.\nFinal answer: 3"],
        trainers.BpeTrainer(
            vocab_size=280,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        ),
    )
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(hf_tokenizer),
            n_positions=512,
            n_embd=32,
            n_layer=1,
            n_head=2,
            bos_token_id=hf_tokenizer.bos_token_id,
            eos_token_id=hf_tokenizer.eos_token_id,
            pad_token_id=hf_tokenizer.pad_token_id,
        )
    )
    path.mkdir(parents=True, exist_ok=True)
    hf_tokenizer.save_pretrained(path)
    model.save_pretrained(path)


def resolve_base(name: str, revision: str | None) -> tuple[Path, str | None]:
    local = Path(name).expanduser()
    if local.is_dir():
        if revision:
            raise ValueError("--revision is only for a Hugging Face repository")
        return local.resolve(), None

    if revision is None and name == DEFAULT_MODEL:
        revision = DEFAULT_REVISION
    sha = revision
    if not sha or not re.fullmatch(r"[0-9a-f]{40}", sha):
        sha = HfApi().model_info(name, revision=revision).sha
    if not sha:
        raise ValueError(f"could not resolve model revision for {name}")
    snapshot = snapshot_download(
        name, revision=sha, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]
    )
    return Path(snapshot), sha


def train_stage(
    root: Path,
    name: str,
    *,
    base: Path,
    init: Path | None,
    train_path: Path,
    dev_path: Path,
    data_manifest: Path,
    settings: dict,
    batches: int,
    learning_rate: float,
) -> Path:
    directory = root / name
    export = directory / "export"
    if (directory / "train_summary.json").is_file() and (export / "config.json").is_file():
        return export

    directory.mkdir(exist_ok=True)
    accum = settings["grad_accum"]
    config = {
        "seed": settings["seed"],
        "device": settings["device"],
        "dtype": "auto",
        "manifest": str(data_manifest),
        "tokenizer": {"hf_name": str(base)},
        "model": {"hf_name": str(base)},
        "data": {
            "sft": True,
            "train_path": str(train_path),
            "val_path": str(dev_path),
            "seq_len": settings["seq_len"],
            "batch_size": 1,
        },
        "train": {
            "mode": "sft",
            "steps": batches,
            "lr": learning_rate,
            "grad_accum": accum,
            "activation_checkpointing": settings["device"] == "cuda",
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "log_every": accum * 10,
            "eval_every": accum * 100,
            "save_every": accum * 100,
            "eval_batches": 16,
        },
        "checkpoint": {
            "dir": str(directory),
            "init_from": str(init) if init else None,
            "export_hf": True,
            "keep_last": 1,
        },
    }
    config_path = directory / "stage.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    latest = directory / "ckpt_latest.pt"
    train(str(config_path), resume_override=str(latest) if latest.is_file() else None)
    if not (export / "config.json").is_file():
        raise RuntimeError(f"training stage did not produce a Hugging Face export: {export}")
    return export


def evaluate_stage(
    model: Path, dataset: Path, report: Path, settings: dict
) -> dict:
    if report.is_file():
        return load_json(report)
    result = evaluate(
        model,
        dataset,
        device=settings["device"],
        batch_size=settings["eval_batch_size"],
        max_new_tokens=settings["max_new_tokens"],
    )
    result["batch_size"] = settings["eval_batch_size"]
    result["max_new_tokens"] = settings["max_new_tokens"]
    write_json(report, result)
    return result


def run(args: argparse.Namespace) -> dict:
    root = Path(args.out).resolve()
    data = Path(args.data).resolve() if args.data else (
        root / "fixture_data" if args.smoke else Path("data/reasoning").resolve()
    )
    if not args.smoke and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a CUDA build of PyTorch on your GPU machine.")

    metadata = root / "run.json"
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise ValueError(f"{root} already contains files. Use --resume with the same settings or choose a new --out.")
    root.mkdir(parents=True, exist_ok=True)
    if args.smoke and not args.data and not (data / "manifest.json").is_file():
        prepare(data, fixture=True, seed=args.seed)
    manifest = validate(data, fixture=args.smoke)
    paths = {row["name"]: data / row["path"] for row in manifest["datasets"]}
    counts = {row["name"]: row["rows"] for row in manifest["datasets"]}

    previous = load_json(metadata) if metadata.is_file() else None
    saved_model = previous["model"] if previous is not None else None
    model_name = args.model or saved_model or ("local-smoke-model" if args.smoke else DEFAULT_MODEL)
    if args.smoke and model_name == "local-smoke-model":
        base = root / "fixture_model"
        make_smoke_model(base, args.seed)
        revision = None
    else:
        pinned = (
            previous["model_revision"]
            if previous is not None and saved_model == model_name and not args.revision
            else args.revision
        )
        base, revision = resolve_base(model_name, pinned)

    settings = {
        "data": str(data),
        "data_sha256": hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest(),
        "model": model_name,
        "base_path": str(base),
        "model_revision": revision,
        "seed": args.seed,
        "device": "cpu" if args.smoke else "cuda",
        "seq_len": 512 if args.smoke else args.seq_len,
        "grad_accum": 2 if args.smoke else args.grad_accum,
        "sft_batches": 4 if args.smoke else max(1, math.ceil(counts["train"] * args.epochs)),
        "max_new_tokens": 12 if args.smoke else args.max_new_tokens,
        "eval_batch_size": args.eval_batch_size,
        "skip_rollout": args.skip_rollout,
        "rollout_prompts": 2 if args.smoke else args.rollout_prompts,
        "samples": 2 if args.smoke else args.samples,
    }
    if previous is not None:
        previous["eval_batch_size"] = settings["eval_batch_size"]
        if previous != settings:
            raise ValueError("Run settings differ from run.json. Resume with the original settings or start a new run.")
    write_json(metadata, settings)

    baseline = evaluate_stage(base, paths["dev"], root / "base_dev.json", settings)
    sft = train_stage(
        root, "sft", base=base, init=None, train_path=paths["train"], dev_path=paths["dev"],
        data_manifest=data / "manifest.json", settings=settings,
        batches=settings["sft_batches"], learning_rate=2e-5,
    )
    sft_dev = evaluate_stage(sft, paths["dev"], root / "sft_dev.json", settings)
    selected = sft
    rollout_counts = None
    rollout_dev = None
    if not args.skip_rollout:
        winners = root / "winners.jsonl"
        rollout_report = root / "rollouts.json"
        if rollout_report.is_file():
            rollout_counts = load_json(rollout_report)
        else:
            rollout_counts = collect_winners(
                sft,
                paths["rollout"],
                winners,
                k=settings["samples"],
                max_prompts=settings["rollout_prompts"],
                max_new_tokens=settings["max_new_tokens"],
                batch_size=settings["eval_batch_size"],
                seed=args.seed,
                include_gold=False,
                allow_empty=True,
                device=settings["device"],
            )
            write_json(rollout_report, rollout_counts)

        if rollout_counts["winners"] > 0:
            if not winners.is_file():
                raise FileNotFoundError(winners)
            tuned = train_stage(
                root, "rollout_sft", base=base, init=sft, train_path=winners, dev_path=paths["dev"],
                data_manifest=data / "manifest.json", settings=settings,
                batches=rollout_counts["winners"], learning_rate=5e-6,
            )
            rollout_dev = evaluate_stage(tuned, paths["dev"], root / "rollout_dev.json", settings)
            if rollout_dev["n_correct"] > sft_dev["n_correct"]:
                selected = tuned

    summary = {
        "selected": str(selected),
        "dev_examples": sft_dev["n"],
        "base_dev_exact_match": baseline["exact_match"],
        "sft_dev_exact_match": sft_dev["exact_match"],
        "rollout_dev_exact_match": rollout_dev["exact_match"] if rollout_dev else None,
        "verified_generated_solutions": rollout_counts["winners"] if rollout_counts else None,
    }
    test_report = root / "selected_test.json"
    if args.test or test_report.is_file():
        test = evaluate_stage(selected, paths["test"], test_report, settings)
        summary["test_exact_match"] = test["exact_match"]
        summary["test_examples"] = test["n"]

    write_json(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="prepared dataset directory (default: data/reasoning)")
    parser.add_argument("--out", default="runs/reasoning")
    parser.add_argument("--model", help=f"local HF model directory or Hub ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--revision", help="Hub commit, tag, or branch to resolve and pin")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--seq-len", type=int, default=640)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--rollout-prompts", type=int, default=512)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test", action="store_true", help="evaluate the selected model on the sealed test split")
    parser.add_argument("--smoke", action="store_true", help="use an offline CPU fixture and four training batches")
    args = parser.parse_args()
    if args.smoke and args.revision and not args.model:
        parser.error("--revision requires --model in smoke mode")
    if not math.isfinite(args.epochs) or args.epochs <= 0 or min(
        args.seq_len, args.grad_accum, args.max_new_tokens,
        args.eval_batch_size, args.rollout_prompts, args.samples,
    ) <= 0:
        parser.error("epoch count, sequence limits, batch settings, and rollout sizes must be positive")
    if args.smoke:
        torch.set_num_threads(1)
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
