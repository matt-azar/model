# Reasoning LLM lab

Train a local assistant for math, programming, general reasoning, and ordinary instructions. New runs start from the pretrained **Qwen2.5-1.5B** base checkpoint at a pinned Hugging Face revision and fine-tune it on a mixed set of natural responses.

## Train on Lambda

Use one **40 GB A100** or a larger NVIDIA GPU and the default Lambda Stack Ubuntu image. This workflow updates all model weights. Plan for at least 80 GB of free disk space for the model download, optimizer checkpoints, temporary writes, and the export.

Open a terminal on the instance and start a `tmux` session to keep training alive through disconnects. From the repo root, set up Python and verify that PyTorch sees the GPU. These commands use the CUDA 12.4 build of PyTorch 2.6.

```bash
nvidia-smi
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv venv --python 3.12
source .venv/bin/activate

~/.local/bin/uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
~/.local/bin/uv pip install -r requirements.txt
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
python -u -m src.prepare_general_data
```

Run a short GPU pilot with 200 training examples from the mixed set and eight validation cases per domain.

```bash
python -u -m src.reasoning_train --out runs/general-pilot \
  --epochs 0.02 --dev-limit 8 --eval-batch-size 4
```

For a full one-epoch run and full validation, start a new output directory. The default output is `runs/reasoning`.

```bash
python -u -m src.reasoning_train --out runs/reasoning --eval-batch-size 4
```

Repeat the same command with `--resume` to continue an interrupted run. If only generation runs out of memory, resume with a smaller `--eval-batch-size`. The final evaluation is optional and scores the selected model on the separate test split.

```bash
python -u -m src.reasoning_train --out runs/reasoning --eval-batch-size 4 --resume --test
```

## Training data and evaluation

| Area | Training examples | Held-out evaluation |
|---|---|---|
| Math | GSM8K worked solutions | GSM8K development questions and the full official test split |
| Programming | SmolTalk Self-OSS-Instruct Python solutions and generated code-tracing examples | CRUXEval-O return values |
| General reasoning | QASC evidence-based questions and generated deduction problems with derived explanations | BIG-Bench Hard logical deduction, Boolean expressions, and date understanding |
| Instruction following | SmolTalk conversations, rewriting, summarization, and constrained responses | Exact checks for list ordering, capitalization, and JSON output |

The prepared training set has 2,500 complete examples per area, shuffled together. Dataset revisions, source attribution, example counts, and content hashes are saved with the data. Training masks prompt tokens and predicts the reference response as written, including its end token. It uses a single microbatch, gradient accumulation, activation checkpointing, and mixed precision.

The reports give separate before-and-after scores, completions, and truncation counts. The programming score measures code tracing. The instruction score covers the listed verifiable constraints. These scores measure a defined set of tasks, and the answer checks can give credit to a correct result with weak reasoning. The final benchmark is an evaluation of this training run, with possible overlap between its public questions and the base model's pretraining corpus.

## Use or back up the export

`runs/reasoning/summary.json` records the selected path. Copy `runs/reasoning/sft/export` and the JSON reports off Lambda before terminating it. Keep the run directory and `sft/ckpt_latest.pt` if you need the optimizer state.

```bash
python -m src.chat --ckpt runs/reasoning/sft/export --device cpu
```

Chat sends the question as written. Ask for steps when you want them. `--worked-solution` requests the older GSM8K answer format. The CPU chat CLI uses FP32, so 1.5B needs about 6.2 GB for the weights plus working memory.

### Local browser chat

After you extract the export on your laptop, run the GUI from the repo root using the same Python environment as the terminal CLI.

```powershell
.\.venv\Scripts\python.exe -m src.gui --ckpt runs/reasoning/sft/export --device cpu
```

On Linux or macOS, use `python -m src.gui` with the same options. The launcher opens a browser at `http://127.0.0.1:7860`. Keep its terminal open while you chat. Use `--port 7861` if the default port is occupied. Press Ctrl+C in the launcher to shut it down.

Enter sends a message and Shift+Enter adds a line. Follow-ups include recent turns from the open tab. The tab keeps its conversation until you choose **New chat** or reload it.

## CPU smoke and earlier experiments

```bash
python -m src.reasoning_train --smoke --out runs/reasoning-smoke --test
python -m unittest discover -s tests
```

The offline smoke run uses a tiny random model and local fixture data to exercise training, resume, export, and all four evaluation paths.

An existing math-only run can still resume with its recorded data and checkpoint. To start that experiment separately, prepare it with `python -m src.prepare_reasoning_data`, then pass `--data data/reasoning` and a separate `--out` to the trainer. The original custom Transformer and milestone smoke scripts remain in the repo.
