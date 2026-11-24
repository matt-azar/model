"""Single training entrypoint for smoke / pretrain / sft configs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import torch
import yaml

from src.data import (
    load_configured_tokenizer,
    load_texts,
    make_dataloader,
    make_streaming_dataloader,
)
from src.manifest import sha256_file
from src.model import GPT, ModelConfig
from src.tokenizer import load_tokenizer_from_checkpoint_state


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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
    meta = {
        "git_sha": git_sha(),
        "config_path": str(config_path),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "tokenizer_path": tok_path,
        "tokenizer_sha256": (
            sha256_file(Path(tok_path) / "tokenizer.json")
            if tok_path and (Path(tok_path) / "tokenizer.json").is_file()
            else None
        ),
    }
    return meta


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    tokenizer_state: dict,
    config: dict,
    metrics: dict,
    run_meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "tokenizer": tokenizer_state,
            "model_config": model.cfg.__dict__,
            "config": config,
            "metrics": metrics,
            "run_meta": run_meta,
        },
        path,
    )


def load_checkpoint(path: Path, map_location: str | torch.device) -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


@torch.no_grad()
def eval_holdout_loss(
    model: GPT,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        _, loss = model(input_ids, labels)
        total += float(loss.item())
        n += 1
    model.train()
    if n == 0:
        return float("nan")
    return total / n


def _data_paths(cfg: dict) -> tuple[str, str | None]:
    data = cfg["data"]
    train_path = data.get("train_path") or data.get("path")
    if not train_path:
        raise ValueError("data.train_path or data.path is required")
    val_path = data.get("val_path")
    return str(train_path), (str(val_path) if val_path else None)


def train(config_path: str, resume_override: str | None = None) -> Path:
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = resolve_device(str(cfg.get("device", "auto")))
    run_dir = Path(cfg["checkpoint"]["dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    val_metrics_path = run_dir / "val_metrics.jsonl"

    # Snapshot config used for this run (reproducibility).
    shutil.copy2(config_path, run_dir / "config.yaml")
    run_meta = build_run_meta(cfg, config_path)
    (run_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2) + "\n", encoding="utf-8"
    )

    train_path, val_path = _data_paths(cfg)
    train_texts = load_texts(train_path)
    tokenizer = load_configured_tokenizer(cfg, texts=train_texts)

    mcfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=int(cfg["model"]["n_layer"]),
        n_embd=int(cfg["model"]["n_embd"]),
        n_head=int(cfg["model"]["n_head"]),
        block_size=int(cfg["model"]["block_size"]),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        bias=bool(cfg["model"].get("bias", False)),
    )
    model = GPT(mcfg).to(device)
    lr = float(cfg["train"]["lr"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    start_step = 0
    tokens_seen = 0
    resume_path = resume_override or cfg.get("checkpoint", {}).get("resume")
    if resume_path:
        ckpt = load_checkpoint(Path(resume_path), map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])
        tokens_seen = int(ckpt.get("tokens_seen", 0))
        tokenizer = load_tokenizer_from_checkpoint_state(ckpt["tokenizer"])
        print(f"resumed from {resume_path} at step {start_step} tokens={tokens_seen}")

    seq_len = int(cfg["data"]["seq_len"])
    batch_size = int(cfg["data"]["batch_size"])
    use_streaming = bool(cfg["data"].get("streaming", False))

    if use_streaming:
        loader = make_streaming_dataloader(
            paths=[train_path],
            tokenizer=tokenizer,
            seq_len=seq_len,
            batch_size=batch_size,
            seed=seed,
            cycle=True,
            max_tokens=None,
            stand_in=bool(cfg["data"].get("stand_in", False)),
        )
    else:
        loader = make_dataloader(
            train_texts,
            tokenizer,
            seq_len=seq_len,
            batch_size=batch_size,
            seed=seed,
        )
    data_iter = iter(loader)

    val_loader = None
    if val_path:
        val_texts = load_texts(val_path)
        val_loader = make_dataloader(
            val_texts,
            tokenizer,
            seq_len=seq_len,
            batch_size=batch_size,
            seed=seed + 1,
        )

    total_steps = int(cfg["train"].get("steps") or 0)
    token_budget = cfg["train"].get("token_budget")
    token_budget = int(token_budget) if token_budget is not None else None
    if total_steps <= 0 and token_budget is None:
        raise ValueError("set train.steps and/or train.token_budget")

    log_every = int(cfg["train"].get("log_every", 50))
    save_every = int(cfg["train"].get("save_every", 200))
    eval_every = int(cfg["train"].get("eval_every", save_every))
    eval_batches = int(cfg["train"].get("eval_batches", 20))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    overfit_threshold = cfg["train"].get("overfit_loss_threshold")

    model.train()
    last_loss = float("inf")
    last_grad_norm = 0.0
    first_val = None
    last_val = None
    t0 = time.time()
    step = start_step
    tokens_this_run = 0

    with metrics_path.open("a", encoding="utf-8") as metrics_f, val_metrics_path.open(
        "a", encoding="utf-8"
    ) as val_f:
        while True:
            if total_steps > 0 and step >= total_steps:
                break
            if token_budget is not None and tokens_seen >= token_budget:
                break

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            _, loss = model(input_ids, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                last_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                )
            else:
                last_grad_norm = 0.0
            optimizer.step()

            step += 1
            ntok = int(input_ids.numel())
            tokens_seen += ntok
            tokens_this_run += ntok
            last_loss = float(loss.item())

            should_log = step % log_every == 0 or (
                total_steps > 0 and step == total_steps
            ) or (token_budget is not None and tokens_seen >= token_budget)
            if should_log:
                elapsed = max(time.time() - t0, 1e-8)
                tps = tokens_this_run / elapsed
                row = {
                    "step": step,
                    "loss": last_loss,
                    "grad_norm": last_grad_norm,
                    "lr": lr,
                    "tokens_seen": tokens_seen,
                    "tokens_per_sec": tps,
                    "device": str(device),
                    "git_sha": run_meta["git_sha"],
                }
                metrics_f.write(json.dumps(row) + "\n")
                metrics_f.flush()
                print(
                    f"step {step} loss={last_loss:.4f} grad={last_grad_norm:.3f} "
                    f"tokens={tokens_seen} tok/s={tps:.1f} device={device}"
                )

            should_eval = val_loader is not None and (
                step % eval_every == 0
                or (total_steps > 0 and step == total_steps)
                or (token_budget is not None and tokens_seen >= token_budget)
            )
            if should_eval:
                val_loss = eval_holdout_loss(model, val_loader, device, eval_batches)
                if first_val is None:
                    first_val = val_loss
                last_val = val_loss
                val_row = {
                    "step": step,
                    "val_loss": val_loss,
                    "tokens_seen": tokens_seen,
                    "git_sha": run_meta["git_sha"],
                }
                val_f.write(json.dumps(val_row) + "\n")
                val_f.flush()
                print(f"eval step {step} val_loss={val_loss:.4f}")

            should_save = (
                step % save_every == 0
                or (total_steps > 0 and step == total_steps)
                or (token_budget is not None and tokens_seen >= token_budget)
            )
            if should_save:
                metrics = {
                    "loss": last_loss,
                    "val_loss": last_val,
                    "tokens_seen": tokens_seen,
                }
                ckpt_path = run_dir / f"ckpt_step_{step:06d}.pt"
                save_checkpoint(
                    ckpt_path,
                    model,
                    optimizer,
                    step,
                    tokens_seen,
                    tokenizer.to_state(),
                    cfg,
                    metrics,
                    run_meta,
                )
                latest = run_dir / "ckpt_latest.pt"
                save_checkpoint(
                    latest,
                    model,
                    optimizer,
                    step,
                    tokens_seen,
                    tokenizer.to_state(),
                    cfg,
                    metrics,
                    run_meta,
                )
                print(f"saved {ckpt_path}")

    if overfit_threshold is not None and last_loss > float(overfit_threshold):
        raise SystemExit(
            f"overfit check failed: final loss {last_loss:.4f} "
            f"> threshold {overfit_threshold}"
        )

    summary = {
        "final_loss": last_loss,
        "first_val_loss": first_val,
        "final_val_loss": last_val,
        "steps": step,
        "tokens_seen": tokens_seen,
        "git_sha": run_meta["git_sha"],
        "config_path": str(config_path),
    }
    (run_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"train done final_loss={last_loss:.4f} tokens={tokens_seen} sha={run_meta['git_sha']}")
    return run_dir / "ckpt_latest.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train decoder-only LM from config")
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument("--resume", default=None, help="optional checkpoint path")
    args = parser.parse_args()
    train(args.config, resume_override=args.resume)


if __name__ == "__main__":
    main()
