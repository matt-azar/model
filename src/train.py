"""Single training entrypoint for Phase 0 smoke / pretrain / sft configs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from src.data import load_configured_tokenizer, load_texts, make_dataloader
from src.model import GPT, ModelConfig
from src.tokenizer import load_tokenizer_from_checkpoint_state


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokenizer_state: dict,
    config: dict,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tokenizer": tokenizer_state,
            "model_config": model.cfg.__dict__,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: Path, map_location: str | torch.device) -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


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

    texts = load_texts(cfg["data"]["path"])
    tokenizer = load_configured_tokenizer(cfg, texts=texts)

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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    start_step = 0
    resume_path = resume_override or cfg.get("checkpoint", {}).get("resume")
    if resume_path:
        ckpt = load_checkpoint(Path(resume_path), map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])
        tokenizer = load_tokenizer_from_checkpoint_state(ckpt["tokenizer"])
        print(f"resumed from {resume_path} at step {start_step}")

    loader = make_dataloader(
        texts,
        tokenizer,
        seq_len=int(cfg["data"]["seq_len"]),
        batch_size=int(cfg["data"]["batch_size"]),
        seed=seed,
    )
    data_iter = iter(loader)

    total_steps = int(cfg["train"]["steps"])
    log_every = int(cfg["train"].get("log_every", 50))
    save_every = int(cfg["train"].get("save_every", 200))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    overfit_threshold = cfg["train"].get("overfit_loss_threshold")

    model.train()
    last_loss = float("inf")
    t0 = time.time()
    tokens_seen = 0

    with metrics_path.open("a", encoding="utf-8") as metrics_f:
        for step in range(start_step + 1, total_steps + 1):
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            last_loss = float(loss.item())
            tokens_seen += input_ids.numel()

            if step % log_every == 0 or step == total_steps:
                elapsed = max(time.time() - t0, 1e-8)
                tps = tokens_seen / elapsed
                row = {
                    "step": step,
                    "loss": last_loss,
                    "tokens_per_sec": tps,
                    "device": str(device),
                }
                metrics_f.write(json.dumps(row) + "\n")
                metrics_f.flush()
                print(
                    f"step {step}/{total_steps} loss={last_loss:.4f} "
                    f"tok/s={tps:.1f} device={device}"
                )

            if step % save_every == 0 or step == total_steps:
                ckpt_path = run_dir / f"ckpt_step_{step:06d}.pt"
                save_checkpoint(
                    ckpt_path,
                    model,
                    optimizer,
                    step,
                    tokenizer.to_state(),
                    cfg,
                    {"loss": last_loss},
                )
                latest = run_dir / "ckpt_latest.pt"
                save_checkpoint(
                    latest,
                    model,
                    optimizer,
                    step,
                    tokenizer.to_state(),
                    cfg,
                    {"loss": last_loss},
                )
                print(f"saved {ckpt_path}")

    if overfit_threshold is not None and last_loss > float(overfit_threshold):
        raise SystemExit(
            f"overfit check failed: final loss {last_loss:.4f} "
            f"> threshold {overfit_threshold}"
        )

    print(f"train done final_loss={last_loss:.4f}")
    return run_dir / "ckpt_latest.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train decoder-only LM from config")
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument("--resume", default=None, help="optional checkpoint path")
    args = parser.parse_args()
    train(args.config, resume_override=args.resume)


if __name__ == "__main__":
    main()
