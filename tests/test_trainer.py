import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from src.hf_adapter import HFTokenizer, load_hf_checkpoint_model
from src.sft_data import SFTPromptMaskedDataset
from src.template import format_reasoning_prompt
from src.tokenizer import CharTokenizer
from src.train import load_checkpoint, train


class TrainerTest(TestCase):
    def test_sft_keeps_complete_answer_and_masks_prompt(self):
        short = {"question": "2+2?", "scratchpad": "2+2=4", "answer": "4"}
        long = {**short, "scratchpad": short["scratchpad"] * 100}
        tokenizer = CharTokenizer.from_texts(
            ["### Instruction\n### Response\nShow your work: Final answer: 2+2?=4"]
        )
        dataset = SFTPromptMaskedDataset([short, long], tokenizer, seq_len=256)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.skipped_long, 1)
        labels = dataset[0]["labels"]
        targets = labels[labels.ne(-100)].tolist()
        self.assertEqual(targets[-1], tokenizer.eos_id)
        self.assertTrue(tokenizer.decode(targets).endswith("Final answer: 4"))

    def test_resume_and_export_hf_checkpoint_without_original_model(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            tokens = ["<unk>", "<eos>", "4", "2", "Show", "Final", "0", "1", "3", "5", "6"]
            vocab = {token: i for i, token in enumerate(tokens)}
            tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
            tokenizer.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(
                tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<eos>",
                chat_template="User: {{ messages[0]['content'] }}\nAssistant:",
            ).save_pretrained(base)
            torch.manual_seed(123)
            Qwen2ForCausalLM(Qwen2Config(
                vocab_size=len(vocab), num_hidden_layers=1, hidden_size=16, intermediate_size=32,
                num_attention_heads=2, num_key_value_heads=1, attention_dropout=0.1,
                max_position_embeddings=128, eos_token_id=vocab["<eos>"], bos_token_id=None,
            )).save_pretrained(base)

            data = root / "rows.jsonl"
            rows = [
                {"question": f"{n} plus 2?", "scratchpad": f"{n} + 2 = {n + 2}.", "answer": str(n + 2)}
                for n in range(5)
            ]
            data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            cfg = {
                "device": "cpu",
                "dtype": "auto",
                "model": {"hf_name": str(base)},
                "data": {"train_path": str(data), "sft": True, "batch_size": 2, "seq_len": 128},
                "train": {
                    "mode": "sft", "steps": 8, "lr": 0.003, "grad_accum": 2,
                    "save_every": 3, "activation_checkpointing": True,
                },
                "checkpoint": {"dir": str(root / "reference"), "export_hf": True},
            }
            config = root / "run.yaml"
            config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            latest = train(str(config))
            expected = load_checkpoint(latest, "cpu")
            self.assertEqual(expected["optimizer_steps"], 4)
            self.assertFalse((latest.parent / "ckpt_step_000003.pt").exists())

            shutil.rmtree(base)
            restored_tokenizer = HFTokenizer.from_state(expected["tokenizer"])
            self.assertEqual(restored_tokenizer.bos_id, -1)
            self.assertTrue(format_reasoning_prompt("2+2?", restored_tokenizer).endswith("Assistant:"))
            cfg["checkpoint"]["dir"] = str(root / "resumed")
            config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            resumed = train(str(config), str(latest.parent / "ckpt_step_000004.pt"))
            actual = load_checkpoint(resumed, "cpu")
            model = load_hf_checkpoint_model(actual, resumed)
            self.assertEqual(model.block_size, 128)
            for name, tensor in expected["model"].items():
                self.assertTrue(torch.equal(tensor, actual["model"][name]), name)

            # A process can exit after its final checkpoint but before export.
            shutil.rmtree(resumed.parent / "export")
            (resumed.parent / "train_summary.json").unlink()
            self.assertEqual(train(str(config), str(resumed)), resumed)
            self.assertTrue((resumed.parent / "export" / "model.safetensors").is_file())
            self.assertTrue((resumed.parent / "train_summary.json").is_file())
