"""Toy JSONL dataset with simple packing for Phase 0."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from src.tokenizer import CharTokenizer


def load_texts(path: str | Path) -> list[str]:
    path = Path(path)
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" not in obj:
                raise ValueError(f"missing 'text' field in {path}")
            texts.append(str(obj["text"]))
    if not texts:
        raise ValueError(f"no documents in {path}")
    return texts


def build_tokenizer(texts: list[str]) -> CharTokenizer:
    return CharTokenizer.from_texts(texts)


class PackedDataset(Dataset):
    """Pack tokenized documents into fixed-length causal LM windows."""

    def __init__(
        self,
        texts: list[str],
        tokenizer: CharTokenizer,
        seq_len: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        tokens: list[int] = []
        for text in texts:
            tokens.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
        # Repeat until we have at least a few windows (helps toy overfit).
        min_tokens = seq_len * 8 + 1
        while len(tokens) < min_tokens:
            tokens = tokens + tokens
        self.tokens = torch.tensor(tokens, dtype=torch.long)

    def __len__(self) -> int:
        return max(1, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = (idx * self.seq_len) % (len(self.tokens) - self.seq_len - 1)
        chunk = self.tokens[start : start + self.seq_len + 1]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


def make_dataloader(
    texts: list[str],
    tokenizer: CharTokenizer,
    seq_len: int,
    batch_size: int,
    seed: int,
) -> DataLoader:
    ds = PackedDataset(texts, tokenizer, seq_len=seq_len)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=g,
        num_workers=0,
    )
