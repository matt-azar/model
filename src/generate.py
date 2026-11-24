"""Load a checkpoint and decode text (toy GPT or HF)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.hf_adapter import HFCausalLM, HFTokenizer
from src.model import GPT, ModelConfig
from src.tokenizer import load_tokenizer_from_checkpoint_state
from src.train import load_checkpoint, resolve_device


def _load_model_and_tokenizer(ckpt_path: str | Path, device: torch.device):
    path = Path(ckpt_path)
    # HF directory with adapter meta / config.
    if path.is_dir():
        model = HFCausalLM.from_pretrained(str(path)).to(device)
        tokenizer = HFTokenizer.from_pretrained(str(path))
        return model, tokenizer

    ckpt = load_checkpoint(path, map_location=device)
    kind = ckpt.get("kind") or (ckpt.get("model_config") or {}).get("kind")
    if kind == "hf":
        hf_dir = ckpt.get("hf_dir")
        hf_name = ckpt.get("hf_name") or ckpt["model_config"].get("hf_name")
        load_from = hf_dir if hf_dir and Path(str(hf_dir)).is_dir() else hf_name
        model = HFCausalLM.from_pretrained(str(load_from)).to(device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
        tokenizer = load_tokenizer_from_checkpoint_state(ckpt["tokenizer"])
        if not isinstance(tokenizer, HFTokenizer):
            tokenizer = HFTokenizer.from_pretrained(str(hf_name))
        return model, tokenizer

    tokenizer = load_tokenizer_from_checkpoint_state(ckpt["tokenizer"])
    mcfg = ModelConfig(**ckpt["model_config"])
    model = GPT(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, tokenizer


def generate_text(
    ckpt_path: str | Path,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    device_name: str = "auto",
) -> str:
    device = resolve_device(device_name)
    model, tokenizer = _load_model_and_tokenizer(ckpt_path, device)
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
