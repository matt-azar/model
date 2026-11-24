"""Regression tests for benchmark separation and artifact provenance."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import prepare_reasoning_data as data


class PrepareReasoningDataTests(unittest.TestCase):
    def test_fixture_is_disjoint_reusable_and_hash_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "data"
            manifest = data.prepare(output, fixture=True)
            self.assertEqual(manifest, data.prepare(output, fixture=True))
            self.assertEqual(manifest, data.prepare(root / "recreated", fixture=True))

            questions = set()
            for entry in manifest["datasets"]:
                rows = [json.loads(line) for line in (output / entry["path"]).read_text().splitlines()]
                self.assertEqual(len(rows), entry["rows"])
                for row in rows:
                    self.assertEqual(row["split"], entry["name"])
                    self.assertNotIn(row["question"], questions)
                    questions.add(row["question"])
            self.assertEqual(len(questions), 32)
            with self.assertRaisesRegex(ValueError, "wrong dataset version or kind"):
                data.validate(output)
            (output / "test.jsonl").write_text("different contents")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                data.validate(output, fixture=True)

    def test_real_split_preserves_official_test_and_excludes_overlaps(self) -> None:
        train = [{"question": f"train {i}", "answer": f"x <<{i}=0>>0\n#### 0"} for i in range(20)]
        train.extend([{"question": " TRAIN  1 ", "answer": "x\n#### 0"}, {"question": "TEST", "answer": "x\n#### 0"}])
        test = [{"question": "test", "answer": "x\n#### 0"}]
        test_rows = data._convert(test, "test")
        splits = data._split_training(data._convert(train, "train"), test_rows, seed=42)
        self.assertIs(splits["test"], test_rows)
        self.assertEqual([len(splits[key]) for key in data.SPLITS], [16, 3, 1, 1])
        keys = [data._question_key(row["question"]) for rows in splits.values() for row in rows]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(row["scratchpad"] == "x 0" for row in splits["train"]))

    def test_real_download_failure_leaves_no_fixture_behind(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "data"
            with patch.object(data, "_load_gsm8k", side_effect=ConnectionError("unavailable")) as load:
                with self.assertRaisesRegex(ConnectionError, "unavailable"):
                    data.prepare(output)
                load.assert_called_once_with(data.GSM8K_REVISION)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
