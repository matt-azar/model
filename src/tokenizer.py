"""Tokenizers for the foundation lab.

Phase 0: CharTokenizer (offline overfit).
Phase 1: FrozenTokenizer wrapping HuggingFace tokenizers BPE, loaded from
a committed tokenizer.json (vocab frozen for the rest of the stack).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


SPECIAL = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class TokenizerProto(Protocol):
    @property
    def vocab_size(self) -> int: ...

    @property
    def pad_id(self) -> int: ...

    @property
    def bos_id(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    @property
    def unk_id(self) -> int: ...

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]: ...

    def decode(self, ids: list[int], skip_special: bool = True) -> str: ...

    def to_state(self) -> dict: ...


@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def pad_id(self) -> int:
        return SPECIAL["<pad>"]

    @property
    def bos_id(self) -> int:
        return SPECIAL["<bos>"]

    @property
    def eos_id(self) -> int:
        return SPECIAL["<eos>"]

    @property
    def unk_id(self) -> int:
        return SPECIAL["<unk>"]

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi.get(ch, self.unk_id) for ch in text)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        special_ids = set(SPECIAL.values())
        chars: list[str] = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            chars.append(self.itos.get(i, ""))
        return "".join(chars)

    def to_state(self) -> dict:
        return {
            "kind": "char",
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
        }

    @classmethod
    def from_state(cls, state: dict) -> "CharTokenizer":
        stoi = {str(k): int(v) for k, v in state["stoi"].items()}
        itos = {int(k): str(v) for k, v in state["itos"].items()}
        return cls(stoi=stoi, itos=itos)

    @classmethod
    def from_texts(cls, texts: list[str]) -> "CharTokenizer":
        chars = sorted({ch for text in texts for ch in text})
        stoi = dict(SPECIAL)
        for ch in chars:
            if ch not in stoi:
                stoi[ch] = len(stoi)
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)


class FrozenTokenizer:
    """Frozen HF tokenizers BPE loaded from disk."""

    def __init__(self, tokenizer: Tokenizer, config: dict) -> None:
        self._tok = tokenizer
        self.config = config
        self.pad_id = int(config["pad_id"])
        self.bos_id = int(config["bos_id"])
        self.eos_id = int(config["eos_id"])
        self.unk_id = int(config["unk_id"])

    @property
    def vocab_size(self) -> int:
        return int(self._tok.get_vocab_size())

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        # Disable built-in template; we control BOS/EOS explicitly.
        ids = self._tok.encode(text, add_special_tokens=False).ids
        out: list[int] = []
        if add_bos:
            out.append(self.bos_id)
        out.extend(ids)
        if add_eos:
            out.append(self.eos_id)
        return out

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        if skip_special:
            special = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
            ids = [i for i in ids if i not in special]
        return self._tok.decode(ids, skip_special_tokens=False)

    def to_state(self) -> dict:
        return {
            "kind": "frozen",
            "tokenizer_json": self._tok.to_str(),
            "config": self.config,
        }

    @classmethod
    def from_state(cls, state: dict) -> "FrozenTokenizer":
        tok = Tokenizer.from_str(state["tokenizer_json"])
        return cls(tok, state["config"])

    @classmethod
    def load(cls, directory: str | Path) -> "FrozenTokenizer":
        directory = Path(directory)
        tok_path = directory / "tokenizer.json"
        cfg_path = directory / "tokenizer_config.json"
        if not tok_path.exists() or not cfg_path.exists():
            raise FileNotFoundError(
                f"frozen tokenizer missing under {directory} "
                "(need tokenizer.json and tokenizer_config.json)"
            )
        tok = Tokenizer.from_file(str(tok_path))
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        return cls(tok, config)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(directory / "tokenizer.json"))
        (directory / "tokenizer_config.json").write_text(
            json.dumps(self.config, indent=2) + "\n",
            encoding="utf-8",
        )


def train_frozen_bpe(
    texts: list[str],
    vocab_size: int = 1024,
    min_frequency: int = 2,
) -> FrozenTokenizer:
    """Train a small BPE and wrap it as a frozen tokenizer."""
    if not texts:
        raise ValueError("need texts to train tokenizer")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    # Keep post-processor light; encode() adds bos/eos itself.
    tokenizer.post_processor = TemplateProcessing(
        single="$A",
        pair="$A $B",
        special_tokens=[],
    )
    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "pad_id": tokenizer.token_to_id("<pad>"),
        "bos_id": tokenizer.token_to_id("<bos>"),
        "eos_id": tokenizer.token_to_id("<eos>"),
        "unk_id": tokenizer.token_to_id("<unk>"),
        "model_type": "BPE",
        "byte_level": True,
        "frozen": True,
    }
    return FrozenTokenizer(tokenizer, config)


def load_tokenizer_from_checkpoint_state(state: dict) -> TokenizerProto:
    kind = state.get("kind", "char")
    if kind == "frozen":
        return FrozenTokenizer.from_state(state)
    return CharTokenizer.from_state(state)
