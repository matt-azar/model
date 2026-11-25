import unittest

from src.eval_general import matches
from src.prepare_general_data import _instruction_checks
from src.sft_data import SFTPromptMaskedDataset, encode_sft_pair
from src.template import format_prompt
from src.tokenizer import CharTokenizer


class GeneralReasoningTests(unittest.TestCase):
    def test_instruction_check_prompts_are_unique(self):
        rows = _instruction_checks(seed=42)
        self.assertEqual(len(rows), 96)
        self.assertEqual(len({row["question"] for row in rows}), len(rows))

    def test_natural_responses_keep_prompt_masking_and_eos(self):
        question = "What is a prime number?"
        response = "An integer greater than one with two positive divisors."
        tokenizer = CharTokenizer.from_texts([format_prompt(question), response])
        row = {"question": question, "response": response}
        prompt_ids, response_ids = encode_sft_pair(row, tokenizer)
        self.assertEqual(tokenizer.decode(prompt_ids), format_prompt(question))
        self.assertEqual(tokenizer.decode(response_ids), response)

        example = SFTPromptMaskedDataset([row], tokenizer, seq_len=256)[0]
        self.assertEqual(example["labels"][:len(prompt_ids) - 1].tolist(), [-100] * (len(prompt_ids) - 1))
        self.assertEqual(example["labels"][len(prompt_ids) - 1:].tolist(), response_ids)
        self.assertEqual(response_ids[-1], tokenizer.eos_id)

    def test_domain_metrics_score_requested_output(self):
        self.assertTrue(matches("We count 3 packs.\n\nThe answer is 12.", "12", "number"))
        self.assertTrue(matches("```python\n[1, 2]\n```", "[1, 2]", "literal"))
        self.assertFalse(matches("list(range(1, 3))", "[1, 2]", "literal"))
        self.assertFalse(matches("True", "1", "literal"))
        self.assertTrue(matches("(b).", "(B)", "choice"))
        self.assertFalse(matches("A or B", "(B)", "choice"))
        self.assertTrue(matches("True.", "True", "boolean"))
        self.assertTrue(matches("a,b, c", "a, b, c", "comma_list"))
        self.assertFalse(matches("hello", "HELLO", "text"))
        self.assertTrue(matches('{"right": "b", "left": "a"}', '{"left": "a", "right": "b"}', "json"))
