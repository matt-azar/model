"""Single training entrypoint for smoke, continued pretraining, and SFT."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from itertools import islice
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Iterator

import torch
import yaml

from src.data import (
    load_configured_tokenizer,
    load_texts,
    make_dataloader,
    make_streaming_dataloader,
)
from src.hf_adapter import (
    HFCausalLM,
    HFTokenizer,
    cuda_supports_bf16,
    load_hf_checkpoint_model,
    resolve_hf_name,
)
from src.manifest import sha256_file
from src.model import GPT, ModelConfig
from src.sft_data import make_sft_dataloader
from src.tokenizer import TokenizerProto, load_tokenizer_from_checkpoint_state


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    return device


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_run_meta(cfg: dict, config_path: str | Path) -> dict:
    manifest_path = Path(cfg.get("manifest", "data/manifest.json"))
    tok_path = (cfg.get("tokenizer") or {}).get("path")
    train_path, val_path = _data_paths(cfg)
    return {
        "git_sha": git_sha(),
        "config_path": str(config_path),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "tokenizer_path": tok_path,
        "tokenizer_sha256": (
            sha256_file(Path(tok_path) / "tokenizer.json")
            if tok_path and (Path(tok_path) / "tokenizer.json").is_file()
            else None
        ),
        "hf_name": resolve_hf_name(cfg),
        "train_sha256": sha256_file(Path(train_path)),
        "val_sha256": sha256_file(Path(val_path)) if val_path else None,
    }


def _training_dtype(name: str, device: torch.device) -> torch.dtype:
    key = name.lower()
    native_bf16 = cuda_supports_bf16(device)
    if key == "auto":
        if device.type != "cuda":
            return torch.float32
        return torch.bfloat16 if native_bf16 else torch.float16
    choices = {
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp16": torch.float16, "float16": torch.float16,
        "fp32": torch.float32, "float32": torch.float32,
    }
    if key not in choices:
        raise ValueError(f"unsupported training dtype: {name}")
    if device.type != "cuda":
        return torch.float32
    if choices[key] == torch.bfloat16 and not native_bf16:
        raise ValueError("this CUDA device does not support BF16; use dtype: auto, fp16, or fp32")
    return choices[key]


def _save_checkpoint(path: Path, payload: dict) -> None:
    # Replace the numbered checkpoint only after the complete file was written.
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temp_path = Path(temp.name)
    latest = path.with_name("ckpt_latest.pt")
    pending_latest = temp_path.with_suffix(".link")
    try:
        torch.save(payload, temp_path)
        temp_path.replace(path)
        try:
            os.link(path, pending_latest)
        except OSError:
            shutil.copyfile(path, pending_latest)
        pending_latest.replace(latest)
    finally:
        temp_path.unlink(missing_ok=True)
        pending_latest.unlink(missing_ok=True)


def load_checkpoint(path: Path, map_location: str | torch.device) -> dict:
    return torch.load(path, map_location=map_location, weights_only=True)


@torch.no_grad()
def eval_holdout_loss(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
    dtype: torch.dtype = torch.float32,
) -> float:
    model.eval()
    total_loss = 0.0
    total_targets = 0
    try:
        for batch in islice(loader, max_batches):
            targets = int(batch["labels"].ne(-100).sum())
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                _, loss = model(batch["input_ids"].to(device), batch["labels"].to(device))
            total_loss += float(loss) * targets
            total_targets += targets
    finally:
        model.train()
    if not total_targets or not math.isfinite(total_loss):
        raise ValueError("validation requires at least one batch with finite loss")
    return total_loss / total_targets


def _data_paths(cfg: dict) -> tuple[str, str | None]:
    data = cfg["data"]
    train_path = data.get("train_path") or data.get("path")
    if not train_path:
        raise ValueError("data.train_path or data.path is required")
    val_path = data.get("val_path")
    return str(train_path), (str(val_path) if val_path else None)


def _initialize_model(
    cfg: dict,
    sft_mode: bool,
    train_path: str,
    source: str | None,
    checkpoint: dict | None,
) -> tuple[GPT | HFCausalLM, TokenizerProto]:
    grad_ckpt = bool(cfg["train"].get("activation_checkpointing", False))
    if checkpoint is not None:
        tokenizer = load_tokenizer_from_checkpoint_state(checkpoint["tokenizer"])
        kind = checkpoint.get("kind") or checkpoint["model_config"].get("kind")
        if kind == "hf":
            assert source is not None
            model = load_hf_checkpoint_model(checkpoint, source, gradient_checkpointing=grad_ckpt)
        else:
            model = GPT(ModelConfig(**checkpoint["model_config"]))
            model.load_state_dict(checkpoint["model"])
        return model, tokenizer

    hf_name = source or resolve_hf_name(cfg)
    if hf_name:
        if source:
            tokenizer = HFTokenizer.from_pretrained(source)
        else:
            tokenizer = load_configured_tokenizer(cfg)
        model = HFCausalLM.from_pretrained(hf_name, gradient_checkpointing=grad_ckpt)
        return model, tokenizer

    texts = None
    if not cfg.get("tokenizer", {}).get("path"):
        if sft_mode:
            raise ValueError("SFT requires a pretrained model or a configured tokenizer")
        texts = load_texts(train_path)
    tokenizer = load_configured_tokenizer(cfg, texts=texts)
    model_cfg = {**cfg["model"], "vocab_size": tokenizer.vocab_size}
    return GPT(ModelConfig(**model_cfg)), tokenizer


def _training_batches(
    loader: torch.utils.data.DataLoader,
    completed_steps: int,
    seed: int,
    streaming: bool,
) -> Iterator[dict[str, torch.Tensor]]:
    generator = loader.generator or torch.Generator()
    loader.generator = generator
    if streaming:
        generator.manual_seed(seed)
        stream = iter(loader)
        yield from islice(stream, completed_steps, None)
        return
    batches_per_epoch = len(loader)
    if not batches_per_epoch:
        raise ValueError("training dataset is too small for the configured batch size")
    epoch, offset = divmod(completed_steps, batches_per_epoch)
    while True:
        # Use a per-epoch seed so restoring needs to skip at most one partial epoch.
        generator.manual_seed(seed + epoch)
        yield from islice(loader, offset, None)
        epoch += 1
        offset = 0


def _resume_common(checkpoint: dict, cfg: dict, run_meta: dict) -> None:
    old = checkpoint["config"]
    previous = checkpoint["run_meta"]
    for key in ("seed", "model", "tokenizer", "dtype"):
        if old.get(key) != cfg.get(key):
            raise ValueError(f"resume config changed {key}; use checkpoint.init_from for weights only")
    for key in ("batch_size", "seq_len", "sft", "streaming"):
        if old["data"].get(key) != cfg["data"].get(key):
            raise ValueError(f"resume data config changed {key}")
    for key in ("mode", "lr", "weight_decay", "grad_accum", "activation_checkpointing", "grad_clip"):
        if old["train"].get(key) != cfg["train"].get(key):
            raise ValueError(f"resume training config changed {key}")
    for key in ("train_sha256", "val_sha256"):
        if key in previous and previous[key] != run_meta[key]:
            raise ValueError(f"resume data contents changed ({key})")


def _trim_progress(path: Path, completed_step: int) -> None:
    if not path.is_file():
        return
    saved = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        if row["step"] > completed_step:
            break
        saved.append(line if line.endswith("\n") else line + "\n")
    staged = path.with_suffix(".tmp")
    staged.write_text("".join(saved), encoding="utf-8")
    staged.replace(path)


def train(config_path: str, resume_override: str | None = None) -> Path:
    cfg = load_config(config_path)
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    checkpoint_cfg = cfg["checkpoint"]
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    device = resolve_device(str(cfg.get("device", "auto")))
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    dtype = _training_dtype(str(cfg.get("dtype", "float32")), device)
    print(f"training on {device} with {dtype}")
    run_dir = Path(checkpoint_cfg["dir"])
    latest = run_dir / "ckpt_latest.pt"
    train_path, val_path = _data_paths(cfg)
    run_meta = build_run_meta(cfg, config_path)
    run_meta.update(device=str(device), training_dtype=str(dtype))

    mode = str(train_cfg.get("mode", "pretrain")).lower()
    sft_mode = mode == "sft" or bool(data_cfg.get("sft", False))
    streaming = not sft_mode and bool(data_cfg.get("streaming", False))
    resume_path = resume_override or checkpoint_cfg.get("resume")
    source = resume_path or checkpoint_cfg.get("init_from")
    checkpoint = load_checkpoint(Path(source), "cpu") if source and Path(source).is_file() else None
    if resume_path:
        if checkpoint is None:
            raise FileNotFoundError(resume_path)
        _resume_common(checkpoint, cfg, run_meta)
        previous_dtype = checkpoint["run_meta"].get("training_dtype")
        if previous_dtype is not None and previous_dtype != str(dtype):
            raise ValueError(
                f"resume requires the original precision ({previous_dtype}); "
                "use checkpoint.init_from to transfer weights"
            )
    elif source and Path(source).suffix == ".pt" and checkpoint is None:
        raise FileNotFoundError(source)

    model, tokenizer = _initialize_model(cfg, sft_mode, train_path, source, checkpoint)
    model = model.to(device=device, dtype=torch.float32)
    tokenizer_state = tokenizer.to_state()
    if source:
        print(f"{'resuming from' if resume_path else 'initialized from'} {source}")

    lr = float(train_cfg["lr"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    use_scaler = dtype == torch.float16
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler, init_scale=4096.0)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler, init_scale=4096.0)

    step = 0
    optimizer_steps = 0
    tokens_seen = 0
    targets_seen = 0
    first_val = None
    last_val = None
    last_loss = float("inf")

    if resume_path:
        assert checkpoint is not None
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint["step"])
        optimizer_steps = int(checkpoint.get("optimizer_steps", 0))
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        targets_seen = int(checkpoint.get("targets_seen", 0))
        if use_scaler and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        if "cpu_rng" in checkpoint:
            torch.set_rng_state(checkpoint["cpu_rng"])
            if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(checkpoint["cuda_rng"], device=device)
        else:
            print("legacy checkpoint has no RNG state; resume is best effort")
        stored_metrics = checkpoint.get("metrics", {})
        last_loss = float(stored_metrics.get("loss", last_loss))
        first_val = stored_metrics.get("first_val_loss")
        last_val = stored_metrics.get("val_loss")
        print(f"resumed at step {step}, tokens={tokens_seen}")
    del checkpoint

    seq_len = int(data_cfg["seq_len"])
    batch_size = int(data_cfg["batch_size"])
    block_size = model.block_size if isinstance(model, HFCausalLM) else model.cfg.block_size
    if seq_len < 1 or seq_len > block_size:
        raise ValueError(f"seq_len must be in [1, {block_size}] for this model")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if isinstance(model, HFCausalLM) and tokenizer.vocab_size > model.vocab_size:
        raise ValueError("tokenizer vocabulary is larger than the model embedding table")

    if sft_mode:
        loader = make_sft_dataloader(train_path, tokenizer, seq_len, batch_size, seed)
    elif streaming:
        loader = make_streaming_dataloader(
            paths=[train_path],
            tokenizer=tokenizer,
            seq_len=seq_len,
            batch_size=batch_size,
            seed=seed,
            stand_in=bool(data_cfg.get("stand_in", False)),
        )
    else:
        loader = make_dataloader(load_texts(train_path), tokenizer, seq_len, batch_size, seed)
    data_iter = _training_batches(loader, step, seed, streaming)

    val_loader = None
    if val_path:
        if sft_mode:
            val_loader = make_sft_dataloader(val_path, tokenizer, seq_len, batch_size, seed + 1)
        else:
            val_loader = make_dataloader(load_texts(val_path), tokenizer, seq_len, batch_size, seed + 1)

    total_steps = int(train_cfg.get("steps") or 0)
    token_budget = int(train_cfg.get("token_budget") or 0)
    if total_steps < 0 or token_budget < 0 or not (total_steps or token_budget):
        raise ValueError("set a positive train.steps and/or train.token_budget")
    grad_accum = int(train_cfg.get("grad_accum", 1))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 200))
    eval_every = int(train_cfg.get("eval_every", save_every))
    eval_batches = int(train_cfg.get("eval_batches", 20))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    if min(grad_accum, log_every, save_every, eval_every, eval_batches) < 1:
        raise ValueError("accumulation, logging, save, and eval intervals must be positive")
    keep_last = int(checkpoint_cfg.get("keep_last", 0))
    if keep_last < 0:
        raise ValueError("checkpoint.keep_last must be nonnegative")

    def finished() -> bool:
        return bool((total_steps and step >= total_steps) or (token_budget and tokens_seen >= token_budget))

    already_finished = finished()
    if already_finished:
        assert resume_path is not None
        print("checkpoint already meets the requested budget; finalizing artifacts")

    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "train_summary.json"
    summary_path.unlink(missing_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    val_metrics_path = run_dir / "val_metrics.jsonl"
    if resume_path:
        _trim_progress(metrics_path, step)
        _trim_progress(val_metrics_path, step)
    config_snapshot = run_dir / "config.yaml"
    if Path(config_path).resolve() != config_snapshot.resolve():
        shutil.copy2(config_path, config_snapshot)
    (run_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2) + "\n", encoding="utf-8"
    )

    model.train()
    t0 = time.monotonic()
    start_tokens = tokens_seen
    start_optimizer_steps = optimizer_steps
    group_batches = 0
    group_targets = 0
    group_loss = 0.0
    # Bound the intermediate backward scale, then normalize by the actual target
    # tokens accumulated. Padding and prompt tokens never contribute to the loss.
    gradient_denominator = batch_size * seq_len * grad_accum
    last_log_step = step
    last_eval_step = step
    last_save_step = step

    with metrics_path.open("a", encoding="utf-8") as metrics_f, val_metrics_path.open(
        "a", encoding="utf-8"
    ) as val_f:
        while not finished():
            batch = next(data_iter)
            targets = int(batch["labels"].ne(-100).sum())
            if not targets:
                raise ValueError("training batch contains no supervised tokens")
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                _, loss = model(batch["input_ids"].to(device), batch["labels"].to(device))
            loss_value = float(loss.detach())
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"non-finite training loss at step {step + 1}")
            scaler.scale(loss * (targets / gradient_denominator)).backward()
            tokens_seen += int(batch["input_ids"].numel())
            targets_seen += targets
            group_batches += 1
            group_targets += targets
            group_loss += loss_value * targets
            step += 1

            done = finished()
            if group_batches < grad_accum and not done:
                continue
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(gradient_denominator / group_targets)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip if grad_clip > 0 else float("inf"),
                error_if_nonfinite=not use_scaler,
            ))
            if math.isfinite(grad_norm):
                scaler.step(optimizer)
                scaler.update()
                optimizer_steps += 1
            else:
                # Drop an overflowing FP16 update and find a usable loss scale.
                scaler.update(new_scale=scaler.get_scale() * 0.5)
                print(f"step {step}: skipped FP16 update with non-finite gradient norm")
            if done and optimizer_steps == start_optimizer_steps:
                raise FloatingPointError("training finished without a successful optimizer update")
            optimizer.zero_grad(set_to_none=True)
            last_loss = group_loss / group_targets
            group_batches = group_targets = 0
            group_loss = 0.0

            if step // log_every > last_log_step // log_every or done:
                elapsed = max(time.monotonic() - t0, 1e-8)
                tps = (tokens_seen - start_tokens) / elapsed
                row = {
                    "step": step,
                    "optimizer_steps": optimizer_steps,
                    "loss": last_loss,
                    "grad_norm": grad_norm if math.isfinite(grad_norm) else None,
                    "lr": lr,
                    "tokens_seen": tokens_seen,
                    "targets_seen": targets_seen,
                    "tokens_per_sec": tps,
                    "device": str(device),
                    "git_sha": run_meta["git_sha"],
                }
                metrics_f.write(json.dumps(row) + "\n")
                metrics_f.flush()
                print(
                    f"step {step} loss={last_loss:.4f} grad={grad_norm:.3f} "
                    f"tokens={tokens_seen} tok/s={tps:.1f} device={device}"
                )
                last_log_step = step

            if val_loader is not None and (step // eval_every > last_eval_step // eval_every or done):
                # Keep the same validation batches across checkpoints.
                val_loader.generator.manual_seed(seed + 1)
                last_val = eval_holdout_loss(model, val_loader, device, eval_batches, dtype)
                if first_val is None:
                    first_val = last_val
                val_f.write(json.dumps({
                    "step": step, "val_loss": last_val, "tokens_seen": tokens_seen,
                    "git_sha": run_meta["git_sha"],
                }) + "\n")
                val_f.flush()
                print(f"eval step {step} val_loss={last_val:.4f}")
                last_eval_step = step

            if step // save_every > last_save_step // save_every or done:
                payload = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "optimizer_steps": optimizer_steps,
                    "tokens_seen": tokens_seen,
                    "targets_seen": targets_seen,
                    "tokenizer": tokenizer_state,
                    "model_config": vars(model.cfg),
                    "config": cfg,
                    "metrics": {
                        "loss": last_loss,
                        "first_val_loss": first_val,
                        "val_loss": last_val,
                        "tokens_seen": tokens_seen,
                    },
                    "run_meta": run_meta,
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                    "scaler": scaler.state_dict(),
                }
                if isinstance(model, HFCausalLM):
                    payload.update(
                        kind="hf",
                        hf_name=model.hf_name,
                        hf_config=model.model.config.to_dict(),
                        hf_generation_config=model.model.generation_config.to_dict(),
                    )
                path = run_dir / f"ckpt_step_{step:06d}.pt"
                _save_checkpoint(path, payload)
                del payload
                if keep_last:
                    paths = sorted(run_dir.glob("ckpt_step_*.pt"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
                    for old in paths[:-keep_last]:
                        old.unlink()
                print(f"saved {path}")
                last_save_step = step

    overfit_threshold = train_cfg.get("overfit_loss_threshold")
    if overfit_threshold is not None and last_loss > float(overfit_threshold):
        raise SystemExit(
            f"overfit check failed: final loss {last_loss:.4f} > threshold {overfit_threshold}"
        )
    if isinstance(model, HFCausalLM) and checkpoint_cfg.get("export_hf", False):
        assert isinstance(tokenizer, HFTokenizer)
        export_dir = run_dir / "export"
        with TemporaryDirectory(prefix=".export-", dir=run_dir) as folder:
            pending_export = Path(folder) / "model"
            model.save_pretrained(pending_export)
            tokenizer.tok.save_pretrained(pending_export)
            if export_dir.exists():
                shutil.rmtree(export_dir)
            pending_export.replace(export_dir)
        print(f"exported model and tokenizer to {export_dir}")

    summary = {
        "final_loss": last_loss,
        "first_val_loss": first_val,
        "final_val_loss": last_val,
        "steps": step,
        "optimizer_steps": optimizer_steps,
        "tokens_seen": tokens_seen,
        "targets_seen": targets_seen,
        "git_sha": run_meta["git_sha"],
        "config_path": str(config_path),
    }
    pending_summary = summary_path.with_suffix(".tmp")
    pending_summary.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    pending_summary.replace(summary_path)
    print(f"train done final_loss={last_loss:.4f} tokens={tokens_seen} sha={run_meta['git_sha']}")
    return Path(resume_path) if already_finished else latest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train decoder-only LM from config")
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument("--resume", default=None, help="optional checkpoint path")
    args = parser.parse_args()
    train(args.config, resume_override=args.resume)


if __name__ == "__main__":
    main()
