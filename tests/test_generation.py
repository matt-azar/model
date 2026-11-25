import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import torch

from src.chat import main as chat_main, resolve_best_ckpt
from src.generate import Completion, Generator
from src.hf_adapter import HFCausalLM
from src.reject_sample import collect_winners
from src.tokenizer import CharTokenizer


class GenerationTest(TestCase):
    def test_chat_requests_worked_solution_format_only_on_opt_in(self):
        question = "What is a prime number?"
        argv = ["chat", "--ckpt", "unused.pt", "--prompt", question]
        tokenizer = Mock()
        tokenizer.chat_template = "configured"
        tokenizer.apply_chat_template.return_value = "native chat prompt"
        with patch.object(Generator, "from_checkpoint") as load, patch("builtins.print"):
            load.return_value.tokenizer = SimpleNamespace(tok=tokenizer)
            with patch("sys.argv", argv):
                chat_main()
            tokenizer.apply_chat_template.assert_called_once_with(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
            self.assertEqual(load.return_value.complete.call_args.args[0], "native chat prompt")

            with patch("sys.argv", [*argv, "--worked-solution"]):
                chat_main()
            content = tokenizer.apply_chat_template.call_args.args[0][0]["content"]
            self.assertIn("Show your work:", content)
            self.assertIn("Final answer:", content)

    def test_chat_uses_selected_model_from_default_run(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "sft" / "export"
            model.mkdir(parents=True)
            summary = root / "summary.json"
            summary.write_text(json.dumps({"selected": str(model)}), encoding="utf-8")
            with patch("src.chat.DEFAULT_RUN_SUMMARY", summary):
                self.assertEqual(resolve_best_ckpt(None), str(model))
                self.assertEqual(resolve_best_ckpt("explicit.pt"), "explicit.pt")

    def test_hf_batch_masks_prompt_and_counts_until_native_eos(self):
        tokenizer = CharTokenizer.from_texts(["abcxyz"])
        x, y, z = (tokenizer.stoi[c] for c in "xyz")
        native_eos = 99

        class Native(torch.nn.Module):
            config = SimpleNamespace(vocab_size=100, max_position_embeddings=8)
            generation_config = SimpleNamespace(eos_token_id=native_eos)

            def generate(self, ids, **options):
                self.ids = ids.tolist()
                self.options = options
                suffixes = torch.tensor([[x, native_eos, 0], [y, z, tokenizer.eos_id]])
                return torch.cat((ids, suffixes), dim=1)

        native = Native()
        generator = Generator(HFCausalLM(native, "test"), tokenizer, torch.device("cpu"))
        results = generator.generate(["a", "bc"], max_new_tokens=3)

        self.assertEqual(native.ids[0], [tokenizer.pad_id, tokenizer.bos_id, tokenizer.stoi["a"]])
        self.assertEqual(native.options["attention_mask"].tolist(), [[0, 1, 1], [1, 1, 1]])
        self.assertEqual(native.options["eos_token_id"], [native_eos, tokenizer.eos_id])
        self.assertEqual(native.options["pad_token_id"], tokenizer.pad_id)
        self.assertEqual(results, [Completion("x", 2, "eos"), Completion("yz", 3, "eos")])

    def test_rollouts_keep_generated_unique_completed_solutions(self):
        with TemporaryDirectory() as folder:
            pool, out = (Path(folder) / name for name in ("pool.jsonl", "winners.jsonl"))
            row = {"id": "prompt", "question": "Two plus two?", "scratchpad": "Gold", "answer": "4"}
            pool.write_text(json.dumps(row) + "\n", encoding="utf-8")
            correct = "Show your work:\n2 + 2 = 4.\nFinal answer: 4"
            wrong = correct.replace("answer: 4", "answer: 5")
            with patch.object(Generator, "from_checkpoint") as load:
                load.return_value.tokenizer = CharTokenizer.from_texts([""])
                load.return_value.generate.side_effect = [
                    [Completion(correct, 20, "eos"), Completion(correct, 20, "eos")],
                    [Completion(wrong, 20, "eos"), Completion(correct, 20, "length")],
                ]
                report = collect_winners(
                    "unused.pt", pool, out, k=4, batch_size=2, include_gold=False
                )
                load.assert_called_once()
                self.assertEqual(load.return_value.generate.call_count, 2)

            winner = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["sample_passed"], 2)
            self.assertEqual(report["generated_kept"], 1)
            self.assertEqual(report["length_capped"], 1)
            self.assertEqual(winner["prompt_id"], "prompt")
            self.assertEqual(winner["scratchpad"], "2 + 2 = 4.")
