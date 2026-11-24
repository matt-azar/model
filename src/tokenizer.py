"""Minimal character-level tokenizer for Phase 0 toy runs.

Adopt a real open tokenizer in a later phase; this stays offline and
deterministic so smoke trains without downloads.
"""

from __future__ import annotations

from dataclasses import dataclass


SPECIAL = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}


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
        return {"stoi": self.stoi, "itos": {str(k): v for k, v in self.itos.items()}}

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
