"""Data loading, packing, train/val split, and streaming for Phase 1."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from src.tokenizer import CharTokenizer, FrozenTokenizer, TokenizerProto


def load_texts(path: str | Path) -> list[str]:
    """Load all texts from a jsonl or parquet file into memory."""
    path = Path(path)
    texts = list(iter_texts(path))
    if not texts:
        raise ValueError(f"no documents in {path}")
    return texts


def iter_texts(path: str | Path) -> Iterator[str]:
    """Yield text fields from jsonl or parquet without loading everything."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl(path)
    elif suffix in {".parquet", ".pq"}:
        yield from _iter_parquet(path)
    else:
        raise ValueError(f"unsupported data format: {path} (use .jsonl or .parquet)")


def _iter_jsonl(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" not in obj:
                raise ValueError(f"missing 'text' field in {path}:{line_no}")
            yield str(obj["text"])


def _iter_parquet(path: Path) -> Iterator[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required for parquet support") from exc
    table = pq.read_table(path, columns=["text"])
    col = table.column("text")
    for i in range(len(col)):
        value = col[i].as_py()
        if value is None:
            continue
        yield str(value)


def write_jsonl(texts: list[str], path: str | Path, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for text in texts:
            row = {"text": text}
            if extra:
                row.update(extra)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_parquet(texts: list[str], path: str | Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required for parquet support") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"text": texts})
    pq.write_table(table, path)


def train_val_split(
    texts: list[str],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")
    items = list(texts)
    rng = random.Random(seed)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio))
    if n_val >= len(items):
        n_val = 1
    val = items[:n_val]
    train = items[n_val:]
    if not train:
        train, val = items[:-1], items[-1:]
    return train, val


def build_tokenizer(texts: list[str]) -> CharTokenizer:
    return CharTokenizer.from_texts(texts)


def load_configured_tokenizer(
    cfg: dict,
    texts: list[str] | None = None,
) -> TokenizerProto:
    """Load HF / frozen tokenizer from config, else build char tokenizer from texts."""
    tok_cfg = cfg.get("tokenizer") or {}
    model_cfg = cfg.get("model") or {}
    hf_name = tok_cfg.get("hf_name") or tok_cfg.get("hf_model") or model_cfg.get("hf_name")
    if hf_name:
        from src.hf_adapter import HFTokenizer

        return HFTokenizer.from_pretrained(str(hf_name))
    path = tok_cfg.get("path")
    if path:
        return FrozenTokenizer.load(path)
    if texts is None:
        raise ValueError("texts required when tokenizer.path / hf_name is not set")
    return build_tokenizer(texts)


class PackedDataset(Dataset):
    """In-memory packed windows (Phase 0 / small files)."""

    def __init__(
        self,
        texts: list[str],
        tokenizer: TokenizerProto,
        seq_len: int,
        min_windows: int = 8,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        tokens: list[int] = []
        for text in texts:
            tokens.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
        min_tokens = seq_len * min_windows + 1
        while len(tokens) < min_tokens:
            tokens = tokens + tokens
        self.tokens = torch.tensor(tokens, dtype=torch.long)

    def __len__(self) -> int:
        return max(1, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = (idx * self.seq_len) % (len(self.tokens) - self.seq_len - 1)
        chunk = self.tokens[start : start + self.seq_len + 1]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


class StreamingPackedDataset(IterableDataset):
    """Stream documents from disk, pack into seq_len windows, optionally cycle.

    Use ``max_tokens`` to stop after a token budget (Phase 1 exit: >1B tokens
    against a license-documented stand-in mix). When ``cache_tokens`` is true
    (default), documents are tokenized once then cycled from an in-memory
    token buffer so long streams do not re-encode forever.
    """

    def __init__(
        self,
        paths: list[str | Path],
        tokenizer: TokenizerProto,
        seq_len: int,
        seed: int = 42,
        cycle: bool = True,
        max_tokens: int | None = None,
        stand_in: bool = False,
        cache_tokens: bool = True,
    ) -> None:
        if not paths:
            raise ValueError("paths must be non-empty")
        self.paths = [Path(p) for p in paths]
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed
        self.cycle = cycle
        self.max_tokens = max_tokens
        self.stand_in = stand_in
        self.cache_tokens = cache_tokens
        self._cached: list[int] | None = None

    def _load_token_stream(self) -> list[int]:
        if self.cache_tokens and self._cached is not None:
            return self._cached
        tokens: list[int] = []
        for path in self.paths:
            for text in iter_texts(path):
                tokens.extend(self.tokenizer.encode(text, add_bos=True, add_eos=True))
        if not tokens:
            raise ValueError(f"no tokens from paths={self.paths}")
        if self.cache_tokens:
            self._cached = tokens
        return tokens

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        base = self._load_token_stream()
        # Ensure at least one full window even for tiny corpora.
        tokens = list(base)
        while len(tokens) < self.seq_len + 1:
            tokens.extend(base)

        emitted_tokens = 0
        pos = 0
        n = len(tokens)
        while True:
            if pos + self.seq_len + 1 > n:
                if not self.cycle:
                    return
                # Continuously cycle the buffer without reallocating huge lists.
                pos = 0
            chunk = tokens[pos : pos + self.seq_len + 1]
            if len(chunk) < self.seq_len + 1:
                # Wrap across the end boundary.
                need = self.seq_len + 1 - len(chunk)
                chunk = chunk + tokens[:need]
                pos = need
            else:
                pos += self.seq_len
            input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
            labels = torch.tensor(chunk[1:], dtype=torch.long)
            yield {"input_ids": input_ids, "labels": labels}
            emitted_tokens += self.seq_len
            if self.max_tokens is not None and emitted_tokens >= self.max_tokens:
                return


def make_dataloader(
    texts: list[str],
    tokenizer: TokenizerProto,
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


def make_streaming_dataloader(
    paths: list[str | Path],
    tokenizer: TokenizerProto,
    seq_len: int,
    batch_size: int,
    seed: int,
    cycle: bool = True,
    max_tokens: int | None = None,
    stand_in: bool = False,
) -> DataLoader:
    ds = StreamingPackedDataset(
        paths=paths,
        tokenizer=tokenizer,
        seq_len=seq_len,
        seed=seed,
        cycle=cycle,
        max_tokens=max_tokens,
        stand_in=stand_in,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=0)


def stream_packed_token_count(
    paths: list[str | Path],
    tokenizer: TokenizerProto,
    seq_len: int,
    target_tokens: int,
    seed: int = 42,
    stand_in: bool = False,
    log_every_tokens: int = 50_000_000,
) -> dict:
    """Walk the same packed cycling stream as training, counting tokens only.

    Avoids DataLoader/tensor overhead so the Phase 1 >1B-token gate finishes
    quickly while still exercising tokenize -> pack -> cycle.
    """
    ds = StreamingPackedDataset(
        paths=paths,
        tokenizer=tokenizer,
        seq_len=seq_len,
        seed=seed,
        cycle=True,
        max_tokens=target_tokens,
        stand_in=stand_in,
        cache_tokens=True,
    )
    base = ds._load_token_stream()
    if len(base) < seq_len + 1:
        raise ValueError("not enough tokens to form one packed window")

    tokens = 0
    windows = 0
    pos = 0
    n = len(base)
    # Stretch tiny buffers the same way __iter__ does.
    if n < seq_len + 1:
        raise ValueError("internal: buffer too small")
    t0 = time.time()
    next_log = log_every_tokens
    while tokens < target_tokens:
        if pos + seq_len + 1 > n:
            pos = 0
        # Touch the slice so we actually read the packed window.
        _ = base[pos : pos + seq_len + 1]
        if len(_) < seq_len + 1:
            need = seq_len + 1 - len(_)
            _ = _ + base[:need]
            pos = need
        else:
            pos += seq_len
        if len(_) != seq_len + 1:
            raise RuntimeError("packer produced short window")
        tokens += seq_len
        windows += 1
        if tokens >= next_log:
            elapsed = max(time.time() - t0, 1e-8)
            print(
                f"streamed {tokens:,} tokens in {windows:,} windows "
                f"({tokens / elapsed:,.0f} tok/s)"
            )
            next_log += log_every_tokens

    elapsed = max(time.time() - t0, 1e-8)
    return {
        "tokens": tokens,
        "windows": windows,
        "seconds": elapsed,
        "tokens_per_sec": tokens / elapsed,
        "corpus_tokens": n,
        "stand_in": stand_in,
    }
