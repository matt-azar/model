"""SFT dataset with prompt-token masking."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.template import format_reasoning_prompt, format_response
from src.tokenizer import TokenizerProto


def load_sft_rows(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "question" not in obj or "answer" not in obj:
                raise ValueError(f"SFT row missing question/answer at {path}:{line_no}")
            if "scratchpad" not in obj:
                raise ValueError(f"SFT row missing scratchpad at {path}:{line_no}")
            rows.append(obj)
    if not rows:
        raise ValueError(f"no SFT rows in {path}")
    return rows


class SFTPromptMaskedDataset(Dataset):
    """Each item is one example; prompt tokens are masked with label -100."""

    def __init__(
        self,
        rows: list[dict],
        tokenizer: TokenizerProto,
        seq_len: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.examples: list[dict[str, torch.Tensor]] = []
        self.skipped_long = 0
        for row in rows:
            prompt = format_reasoning_prompt(row["question"], tokenizer)
            response = format_response(row["scratchpad"], row["answer"])
            prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
            resp_ids = tokenizer.encode(response, add_bos=False, add_eos=True)
            input_ids = prompt_ids + resp_ids
            # Causal LM labels: predict token t from prefix; mask prompt positions.
            labels = [-100] * len(prompt_ids) + resp_ids
            # Align labels with next-token prediction used by the model:
            # model gets input_ids[:-1], labels[1:] conceptually — we store full
            # sequences and slice in __getitem__ like PackedDataset.
            if len(input_ids) < 2:
                continue
            if len(input_ids) > seq_len + 1:
                # A clipped completion can silently drop the final answer or EOS.
                self.skipped_long += 1
                continue
            self.examples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )
        if not self.examples:
            raise ValueError(
                f"no complete SFT examples fit seq_len={seq_len} "
                f"({self.skipped_long} over length)"
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        ids = ex["input_ids"]
        labels = ex["labels"]
        # Teacher forcing: inputs are all but last; labels are all but first
        # so prompt mask on labels[0] (BOS under prompt) stays ignored.
        return {
            "input_ids": ids[:-1],
            "labels": labels[1:],
        }


def _pad_collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].numel() for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["input_ids"].numel()
        input_ids[i, :n] = item["input_ids"]
        labels[i, :n] = item["labels"]
    return {"input_ids": input_ids, "labels": labels}


def make_sft_dataloader(
    path: str | Path,
    tokenizer: TokenizerProto,
    seq_len: int,
    batch_size: int,
    seed: int,
) -> DataLoader:
    rows = load_sft_rows(path)
    ds = SFTPromptMaskedDataset(rows, tokenizer, seq_len=seq_len)
    if ds.skipped_long:
        print(f"{path}: skipped {ds.skipped_long}/{len(rows)} examples over seq_len={seq_len}")
    g = torch.Generator()
    g.manual_seed(seed)

    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return _pad_collate(batch, pad_id=tokenizer.pad_id)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=g,
        num_workers=0,
        collate_fn=collate,
    )
