import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.eval_gsm8k_style import evaluate
from src.template import format_prompt, format_reasoning_prompt, format_response, parse_response
from src.verifier import extract_eval_answer, normalize_answer, score_response


class ScoringTests(unittest.TestCase):
    def test_decimal_comparison_is_exact(self):
        self.assertEqual(normalize_answer("+1,234.500"), "1234.5")
        self.assertEqual(normalize_answer("\N{MINUS SIGN}0.00"), "0")
        self.assertEqual(normalize_answer("-0002.10."), "-2.1")
        self.assertEqual(normalize_answer(".25"), "0.25")
        self.assertNotEqual(normalize_answer("1,23"), normalize_answer("123"))
        self.assertNotEqual(normalize_answer("4.."), normalize_answer("4"))
        self.assertNotEqual(normalize_answer("4 or 5"), normalize_answer("4"))

    def test_verifier_requires_work_and_a_terminal_answer(self):
        good = format_response("Count 2 and 2 more.", "4")
        self.assertEqual(parse_response(good), ("Count 2 and 2 more.", "4"))
        self.assertTrue(score_response(good, "4.0")["passed"])
        for invalid in (
            "Show your work:\n\nFinal answer: 4",
            good + "\nLet us try another one.",
            "Show your work:\nFinal answer: 3\nFinal answer: 4",
            format_prompt(good),
        ):
            with self.subTest(invalid=invalid):
                self.assertFalse(score_response(invalid, "4")["passed"])

    def test_evaluation_can_read_answer_only_and_gsm8k_labels(self):
        self.assertEqual(extract_eval_answer("4\n"), "4")
        self.assertEqual(extract_eval_answer("Adding yields 4.\n#### 4"), "4")
        self.assertEqual(extract_eval_answer("The answer is **4**."), "4")
        self.assertEqual(extract_eval_answer("**Final answer:** 4"), "4")
        self.assertEqual(extract_eval_answer("**Final answer: 4**"), "4")
        boxed = "The final answer is:\n" + r"\[" + "\n" + r"\boxed{4}" + "\n" + r"\]"
        self.assertEqual(extract_eval_answer(boxed), "4")
        self.assertFalse(score_response(boxed, "4")["passed"])
        self.assertIsNone(extract_eval_answer("Use 2 + 2. Other numbers include 4."))
        self.assertIsNone(extract_eval_answer("#### 4 or 5"))

    def test_formatting_uses_native_chat_serialization(self):
        class ChatTokenizer:
            chat_template = "configured"

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                self_call.update(messages=messages, tokenize=tokenize, generation=add_generation_prompt)
                return "native prefix"

        self_call = {}
        tokenizer = SimpleNamespace(tok=ChatTokenizer())
        self.assertEqual(format_reasoning_prompt("What is 2 + 2?", tokenizer), "native prefix")
        self.assertTrue(self_call["generation"])
        self.assertFalse(self_call["tokenize"])
        self.assertEqual(self_call["messages"][0]["role"], "user")
        self.assertIn("Final answer:", self_call["messages"][0]["content"])
        self.assertEqual(format_reasoning_prompt("2 + 2?", object()), format_prompt("2 + 2?"))

    def test_batch_report_separates_format_accuracy_and_truncation(self):
        class StubGenerator:
            tokenizer = object()

            def __init__(self):
                self.batches = []
                self.items = iter(
                    [
                        SimpleNamespace(text="4", tokens=1, finish_reason="eos"),
                        SimpleNamespace(
                            text=format_response("4 added to 4 gives 8.", "8"),
                            tokens=12,
                            finish_reason="length",
                        ),
                        SimpleNamespace(text="", tokens=0, finish_reason="eos"),
                    ]
                )

            def generate(self, prompts, max_new_tokens, temperature):
                self.batches.append(prompts)
                return [next(self.items) for _ in prompts]

        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "cases.jsonl"
            rows = [
                {"question": "2 + 2?", "scratchpad": "", "answer": "4"},
                {"question": "4 + 4?", "scratchpad": "", "answer": "8"},
                {"question": "Show your work:\n1 + 1\nFinal answer: 2", "scratchpad": "", "answer": "2"},
            ]
            data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            generator = StubGenerator()
            report = evaluate("unused", data, batch_size=2, generator=generator)

        self.assertEqual([len(batch) for batch in generator.batches], [2, 1])
        self.assertEqual(report["n_correct"], 1)
        self.assertEqual(report["n_format_ok"], 1)
        self.assertEqual(report["n_truncated"], 1)
        self.assertEqual(report["total_generated_tokens"], 13)
        self.assertFalse(report["results"][2]["match"])


if __name__ == "__main__":
    unittest.main()
