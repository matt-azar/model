from argparse import Namespace
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src import reasoning_train


class ReasoningPipelineTests(unittest.TestCase):
    def test_new_run_uses_base_and_resume_keeps_recorded_checkpoint(self):
        manifest = {"datasets": [
            {"name": name, "path": f"{name}.jsonl", "rows": 1}
            for name in ("train", "rollout", "dev", "test")
        ]}
        old_model = "Qwen/Qwen2.5-0.5B-Instruct"
        old_revision = "7ae557604adf67be50417f59c2c2f167def9a775"
        scores = {"n": 1, "exact_match": 0.0}
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.json").write_text(json.dumps(manifest))
            args = Namespace(
                out=root / "new", data=root, model=None, revision=None, resume=False,
                smoke=False, test=False, seed=42, seq_len=640, grad_accum=8, epochs=1,
                max_new_tokens=384, eval_batch_size=8, skip_rollout=True,
                rollout_prompts=512, samples=4,
            )
            with (
                patch.object(reasoning_train.torch.cuda, "is_available", return_value=True),
                patch.object(reasoning_train, "validate", return_value=manifest),
                patch.object(reasoning_train, "snapshot_download", return_value=str(root)) as snapshot,
                patch.object(reasoning_train, "evaluate_stage", return_value=scores),
                patch.object(reasoning_train, "train_stage", return_value=root / "export"),
            ):
                reasoning_train.run(args)
                self.assertEqual(snapshot.call_args.args[0], "Qwen/Qwen2.5-0.5B")
                self.assertEqual(snapshot.call_args.kwargs["revision"], reasoning_train.DEFAULT_REVISION)

                args.out = root / "existing"
                args.model, args.revision = old_model, old_revision
                reasoning_train.run(args)
                args.model = args.revision = None
                args.resume = True
                reasoning_train.run(args)
                self.assertEqual(snapshot.call_args.args[0], old_model)
                self.assertEqual(snapshot.call_args.kwargs["revision"], old_revision)

    def test_offline_cli_trains_and_evaluates_a_portable_model(self):
        repository = Path(__file__).resolve().parents[1]
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}

        with TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            command = [
                sys.executable, "-m", "src.reasoning_train",
                "--smoke", "--out", str(run),
            ]
            subprocess.run(command, cwd=repository, env=environment, check=True)
            result = json.loads((run / "summary.json").read_text())
            training = json.loads((run / "sft" / "train_summary.json").read_text())
            self.assertEqual(training["optimizer_steps"], 2)
            self.assertEqual(result["dev_examples"], 4)
            self.assertNotIn("test_exact_match", result)
            self.assertEqual(result["verified_generated_solutions"], 0)
            self.assertEqual(Path(result["selected"]), run / "sft" / "export")
            self.assertTrue((run / "sft" / "export" / "tokenizer.json").is_file())

            subprocess.run(
                [*command, "--resume", "--test", "--eval-batch-size", "2"],
                cwd=repository, env=environment, check=True,
            )
            resumed = json.loads((run / "summary.json").read_text())
            self.assertEqual(resumed["selected"], result["selected"])
            self.assertEqual(resumed["test_examples"], 4)
