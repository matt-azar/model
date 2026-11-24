"""Load a checkpoint and decode text."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.model import GPT, ModelConfig
from src.tokenizer import CharTokenizer
from src.train import load_checkpoint, resolve_device


def generate_text(
    ckpt_path: str | Path,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    device_name: str = "auto",
) -> str:
    device = resolve_device(device_name)
    ckpt = load_checkpoint(Path(ckpt_path), map_location=device)
    tokenizer = CharTokenizer.from_state(ckpt["tokenizer"])
    mcfg = ModelConfig(**ckpt["model_config"])
    model = GPT(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        eos_id=tokenizer.eos_id,
    )
    return tokenizer.decode(out[0].tolist(), skip_special=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate from a checkpoint")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", default=None, help="optional YAML for defaults")
    args = parser.parse_args()

    prompt = args.prompt
    max_new = args.max_new_tokens
    temperature = args.temperature
    if args.config:
        with Path(args.config).open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        gen = cfg.get("generate", {})
        if args.prompt == "" and "prompt" in gen:
            prompt = gen["prompt"]
        max_new = gen.get("max_new_tokens", max_new)
        temperature = gen.get("temperature", temperature)

    text = generate_text(
        args.ckpt,
        prompt=prompt,
        max_new_tokens=max_new,
        temperature=temperature,
        device_name=args.device,
    )
    print(text)


if __name__ == "__main__":
    main()
