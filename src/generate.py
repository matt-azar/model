"""Load a checkpoint and decode text (toy GPT or HF)."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import yaml

from src.hf_adapter import (
    HFCausalLM,
    HFTokenizer,
    cuda_supports_bf16,
    load_hf_checkpoint_model,
)
from src.model import GPT, ModelConfig
from src.tokenizer import TokenizerProto, load_tokenizer_from_checkpoint_state
from src.train import load_checkpoint, resolve_device


def _hf_inference_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if cuda_supports_bf16(device) else torch.float16


def _load_model_and_tokenizer(ckpt_path: str | Path, device: torch.device):
    path = Path(ckpt_path)
    # HF directory with adapter meta / config.
    if path.is_dir():
        model = HFCausalLM.from_pretrained(
            str(path), torch_dtype=_hf_inference_dtype(device)
        ).to(device)
        tokenizer = HFTokenizer.from_pretrained(str(path))
        return model, tokenizer

    ckpt = load_checkpoint(path, map_location="cpu")
    kind = ckpt.get("kind") or (ckpt.get("model_config") or {}).get("kind")
    if kind == "hf":
        model = load_hf_checkpoint_model(ckpt, path).to(
            device=device, dtype=_hf_inference_dtype(device)
        )
        tokenizer = load_tokenizer_from_checkpoint_state(ckpt["tokenizer"])
        if not isinstance(tokenizer, HFTokenizer):
            raise ValueError("HF checkpoint is missing an HF tokenizer")
        return model, tokenizer

    tokenizer = load_tokenizer_from_checkpoint_state(ckpt["tokenizer"])
    mcfg = ModelConfig(**ckpt["model_config"])
    model = GPT(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, tokenizer


@dataclass(frozen=True)
class Completion:
    text: str
    tokens: int
    finish_reason: Literal["eos", "length"]


class Generator:
    """Reuse model weights and tokenize / decode the generated suffix by index."""

    def __init__(
        self,
        model: GPT | HFCausalLM,
        tokenizer: TokenizerProto,
        device: torch.device,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_checkpoint(
        cls, ckpt_path: str | Path, device_name: str = "auto"
    ) -> Generator:
        device = resolve_device(device_name)
        model, tokenizer = _load_model_and_tokenizer(ckpt_path, device)
        return cls(model, tokenizer, device)

    def _eos_ids(self) -> list[int]:
        eos = []
        if isinstance(self.model, HFCausalLM):
            native = self.model.model.generation_config.eos_token_id
            if native is not None:
                eos.extend([native] if isinstance(native, int) else native)
        if self.tokenizer.eos_id >= 0 and self.tokenizer.eos_id not in eos:
            eos.append(self.tokenizer.eos_id)
        return eos

    def _batch_suffixes(
        self,
        prefixes: list[list[int]],
        max_new_tokens: int,
        temperature: float,
        eos_ids: list[int],
    ) -> list[list[int]]:
        if isinstance(self.model, HFCausalLM):
            # Decoder-only HF models predict from the rightmost position.
            width = max(map(len, prefixes))
            pad = self.tokenizer.pad_id
            ids = torch.tensor(
                [[pad] * (width - len(prefix)) + prefix for prefix in prefixes],
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.tensor(
                [[0] * (width - len(prefix)) + [1] * len(prefix) for prefix in prefixes],
                dtype=torch.long,
                device=self.device,
            )
            out = self.model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                eos_id=eos_ids or None,
                pad_id=pad,
                attention_mask=attention_mask,
            )
            return out[:, width:].tolist()

        suffixes = []
        for prefix in prefixes:
            ids = torch.tensor([prefix], dtype=torch.long, device=self.device)
            out = self.model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                eos_id=eos_ids[0] if eos_ids else None,
            )
            suffixes.append(out[0, len(prefix):].tolist())
        return suffixes

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        *,
        include_prompt: bool = False,
    ) -> list[Completion]:
        """Generate continuations and report the number of new tokens, including EOS."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")

        prefixes = []
        groups: dict[int, list[int]] = defaultdict(list)
        for index, prompt in enumerate(prompts):
            prefix = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
            if not prefix:
                if self.tokenizer.eos_id < 0:
                    raise ValueError("empty prompt and tokenizer has no start token")
                prefix = [self.tokenizer.eos_id]
            budget = max_new_tokens
            if isinstance(self.model, HFCausalLM):
                budget = min(budget, self.model.block_size - len(prefix))
                if budget < 1:
                    raise ValueError(
                        f"prompt {index} has {len(prefix)} tokens and leaves no space "
                        f"in the model's {self.model.block_size}-token context"
                    )
            prefixes.append(prefix)
            groups[budget].append(index)

        eos_ids = self._eos_ids()
        results = {}
        for budget, indexes in groups.items():
            suffixes = self._batch_suffixes(
                [prefixes[i] for i in indexes], budget, temperature, eos_ids
            )
            for index, suffix in zip(indexes, suffixes, strict=True):
                eos_index = next((i for i, token in enumerate(suffix) if token in eos_ids), None)
                finish: Literal["eos", "length"] = "length"
                n_tokens = len(suffix)
                if eos_index is not None:
                    finish = "eos"
                    n_tokens = eos_index + 1
                    suffix = suffix[:eos_index]
                decode_ids = prefixes[index] + suffix if include_prompt else suffix
                results[index] = Completion(self.tokenizer.decode(decode_ids), n_tokens, finish)
        return [results[i] for i in range(len(prompts))]

    def completions(
        self, prompts: list[str], max_new_tokens: int = 32, temperature: float = 0.0
    ) -> list[str]:
        return [result.text for result in self.generate(prompts, max_new_tokens, temperature)]

    def complete(
        self, prompt: str, max_new_tokens: int = 32, temperature: float = 0.0
    ) -> str:
        return self.completions([prompt], max_new_tokens, temperature)[0]


def generate_text(
    ckpt_path: str | Path,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    device_name: str = "auto",
) -> str:
    generator = Generator.from_checkpoint(ckpt_path, device_name)
    return generator.generate(
        [prompt], max_new_tokens, temperature, include_prompt=True
    )[0].text


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
