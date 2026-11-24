import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class ReasoningPipelineTests(unittest.TestCase):
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
