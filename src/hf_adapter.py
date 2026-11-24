"""HuggingFace adapter for functional CPT / SFT / eval (Milestone A).

Wraps AutoModelForCausalLM + AutoTokenizer so the existing train / generate /
eval paths can use SmolLM (or Qwen2.5-0.5B) via YAML without rewriting the lab.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_HF_MODEL = "HuggingFaceTB/SmolLM-135M"
FALLBACK_HF_MODEL = "Qwen/Qwen2.5-0.5B"


@dataclass
class HFTokenizer:
    """TokenizerProto-compatible wrapper around a HF tokenizer."""

    tok: Any
    pad_id: int
    bos_id: int
    eos_id: int
    unk_id: int
    hf_name: str

    @property
    def vocab_size(self) -> int:
        return int(len(self.tok))

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        # HF encode; we control BOS/EOS explicitly for SFT masking consistency.
        ids = self.tok.encode(text, add_special_tokens=False)
        out: list[int] = []
        if add_bos and self.bos_id >= 0:
            out.append(self.bos_id)
        out.extend(ids)
        if add_eos and self.eos_id >= 0:
            out.append(self.eos_id)
        return out

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self.tok.decode(ids, skip_special_tokens=skip_special)

    def to_state(self) -> dict:
        return {
            "kind": "hf",
            "hf_name": self.hf_name,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
        }

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> "HFTokenizer":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
        if tok.pad_token_id is None:
            # Causal LMs often lack pad; fall back to eos.
            tok.pad_token = tok.eos_token
        bos = tok.bos_token_id
        if bos is None:
            bos = tok.eos_token_id if tok.eos_token_id is not None else 0
        eos = tok.eos_token_id if tok.eos_token_id is not None else bos
        unk = tok.unk_token_id if tok.unk_token_id is not None else eos
        pad = tok.pad_token_id if tok.pad_token_id is not None else eos
        return cls(
            tok=tok,
            pad_id=int(pad),
            bos_id=int(bos),
            eos_id=int(eos),
            unk_id=int(unk),
            hf_name=str(name_or_path),
        )

    @classmethod
    def from_state(cls, state: dict) -> "HFTokenizer":
        return cls.from_pretrained(state["hf_name"])


class HFCausalLM(nn.Module):
    """Thin wrapper so train.py can call model(input_ids, targets) -> logits, loss."""

    def __init__(self, model: Any, hf_name: str, block_size: int | None = None) -> None:
        super().__init__()
        self.model = model
        self.hf_name = hf_name
        cfg = model.config
        self.vocab_size = int(getattr(cfg, "vocab_size", 0))
        self.block_size = int(
            block_size
            or getattr(cfg, "max_position_embeddings", None)
            or getattr(cfg, "n_positions", 2048)
        )

        class _Cfg:
            pass

        self.cfg = _Cfg()
        self.cfg.kind = "hf"
        self.cfg.hf_name = hf_name
        self.cfg.vocab_size = self.vocab_size
        self.cfg.block_size = self.block_size

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        torch_dtype: torch.dtype | None = None,
        gradient_checkpointing: bool = False,
    ) -> "HFCausalLM":
        from transformers import AutoModelForCausalLM

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return cls(model=model, hf_name=str(name_or_path))

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self.model(input_ids=idx)
        logits = out.logits
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        self.model.eval()
        do_sample = temperature is not None and temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": eos_id if eos_id is not None else self.model.config.eos_token_id,
        }
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = eos_id
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
        return self.model.generate(idx, **gen_kwargs)

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        meta = {"kind": "hf", "hf_name": self.hf_name, "block_size": self.block_size}
        (directory / "hf_adapter.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )


def resolve_hf_name(cfg: dict) -> str | None:
    model = cfg.get("model") or {}
    tok = cfg.get("tokenizer") or {}
    return model.get("hf_name") or tok.get("hf_name") or tok.get("hf_model")


def load_hf_pair(
    name_or_path: str,
    torch_dtype: torch.dtype | None = None,
    gradient_checkpointing: bool = False,
) -> tuple[HFCausalLM, HFTokenizer]:
    tokenizer = HFTokenizer.from_pretrained(name_or_path)
    model = HFCausalLM.from_pretrained(
        name_or_path,
        torch_dtype=torch_dtype,
        gradient_checkpointing=gradient_checkpointing,
    )
    return model, tokenizer


def try_load_default_base(
    preferred: str = DEFAULT_HF_MODEL,
    fallback: str = FALLBACK_HF_MODEL,
) -> tuple[HFCausalLM, HFTokenizer, str]:
    last_err: Exception | None = None
    for name in (preferred, fallback):
        try:
            model, tok = load_hf_pair(name)
            return model, tok, name
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"failed to load {name}: {exc}")
    raise RuntimeError(f"could not load HF base ({preferred} / {fallback}): {last_err}")
