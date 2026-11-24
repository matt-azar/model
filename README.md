# Reasoning LLM lab

Train a small model to solve GSM8K math problems and show its work. The main workflow starts from **Qwen2.5-0.5B-Instruct** at a pinned Hugging Face revision, fine-tunes it on worked solutions, and optionally trains it again on its own verified solutions.

## Train on a GPU

Use Python 3.11 or 3.12 and one NVIDIA GPU. Install a CUDA build of PyTorch for your system using the [PyTorch installer](https://pytorch.org/get-started/locally/), then run these commands from the repository root.

```bash
python -m pip install -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
python -m src.prepare_reasoning_data
python -m src.reasoning_train --out runs/reasoning
```

The workflow uses full-weight fine-tuning, a microbatch of one, gradient accumulation, and activation checkpointing. It uses BF16 on supported GPUs and mixed FP16 otherwise. Plan for at least 20 GB of free disk space for the default model, temporary checkpoint writes, and exports. GPU memory use depends on the model and sequence lengths. If evaluation or rollout sampling runs out of GPU memory, resume with a smaller `--eval-batch-size`, down to 1. Use `--skip-rollout` to run supervised fine-tuning alone.

To continue an interrupted run, repeat its command with `--resume` and the same training options. After choosing your run using validation results, add `--test` to evaluate the selected trained model once on the full official test split.

```bash
python -m src.reasoning_train --out runs/reasoning --resume --test
```

`runs/reasoning/summary.json` records the selected trained checkpoint and validation accuracy for the original base, supervised fine-tuning, and verified self-training when available. The rollout stage trains only if it collects verified generated solutions, and its checkpoint is selected only if it answers more validation questions correctly than the supervised checkpoint. Compare the trained model with the base to measure whether fine-tuning helped.

To try the trained model in the terminal, pass the `selected` path from that summary. The example below uses the supervised checkpoint.

```bash
python -m src.chat --ckpt runs/reasoning/sft/export --prompt "A box holds 12 packs of 6 pencils. If 17 pencils are used, how many remain?"
```

## Data and evaluation

The preparation command pins the official GSM8K dataset, separates the official training examples into supervised training, rollout, and validation sets, and preserves all 1,319 official test examples. Each split is recorded in a manifest with its content hash. The GPU workflow requires that prepared dataset.

Validation reports retain generated completions, answer accuracy, format compliance, generation length, and truncation. Supervised fine-tuning predicts the worked solution and final answer with the prompt masked out of the loss. The optional rollout stage samples multiple responses per question and keeps distinct completions with the required structure and the correct final number. The verifier checks the answer and response structure.

Use `python -m src.reasoning_train --help` for settings such as the base model, number of epochs, rollout budget, and generation length.

## CPU smoke test

```bash
python -m src.reasoning_train --smoke --out runs/reasoning-smoke --test
python -m unittest discover -s tests
```

The smoke command creates a tiny random model and local fixture data. It exercises training, export, evaluation, and the rollout gate without downloading a model or dataset. Its scores measure the fixture and are not evidence of model quality.

## Earlier experiments

The repo also contains a custom decoder-only Transformer and the original Phase 0–5 and Milestone A–G scripts. Use `scripts/smoke.sh` for the custom model and `scripts/functional_path.sh` for the earlier Hugging Face smoke pipeline.
