"""Build the pinned training mix and separate reasoning benchmarks."""

from __future__ import annotations

import argparse
import ast
import json
import random
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from urllib.request import urlopen

from src.manifest import load_manifest, refresh_hashes, save_manifest, verify_manifest_hashes
from src.prepare_reasoning_data import GSM8K_REVISION, _load_gsm8k
from src.reasoning_defaults import DEFAULT_MODEL, DEFAULT_REVISION, GENERAL_DATA_DIR, GENERAL_SEQUENCE_LENGTH

DOMAINS = ("math", "programming", "reasoning", "instruction")
SPLITS = ("train", "dev", "test")
SMOLTALK_REVISION = "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"
QASC_REVISION = "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070"
CRUX_REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"
BBH_REVISION = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"
BBH_TASKS = (
    "logical_deduction_three_objects", "logical_deduction_five_objects",
    "boolean_expressions", "date_understanding",
)
INSTRUCTION_SOURCES = ("everyday-conversations", "smol-rewrite", "smol-summarize", "smol-constraints")
MATH_EVAL_SUFFIX = "\n\nSolve the problem and give your final answer as a number."


def _key(question: str) -> str:
    return " ".join(question.removesuffix(MATH_EVAL_SUFFIX).casefold().split())


def _row(source: str, index: int, domain: str, question: str, response: str, **metadata) -> dict:
    return {
        "id": f"{source}/{index}", "domain": domain,
        "question": question.strip(), "response": response.strip(), **metadata,
    }


def _smoltalk(subset: str, domain: str, seed: int) -> Iterable[dict]:
    from datasets import load_dataset

    dataset = load_dataset(
        "HuggingFaceTB/smoltalk", subset, revision=SMOLTALK_REVISION, split="train",
    ).shuffle(seed=seed)
    for index, example in enumerate(dataset):
        messages = example["messages"]
        instruction = messages[0]["content"] if messages[0]["role"] == "system" else ""
        for user, assistant in zip(messages, messages[1:]):
            if user["role"] == "user" and assistant["role"] == "assistant":
                question = user["content"].strip()
                answer = assistant["content"].strip()
                if len(question) >= 24 and answer:
                    prompt = f"{instruction}\n\n{question}" if instruction else question
                    yield _row(subset, index, domain, prompt, answer)
                    break


def _math(row: dict, index: int, *, evaluation: bool) -> dict:
    question = row["question"]
    if evaluation:
        question += MATH_EVAL_SUFFIX
    response = f"{row['scratchpad']}\n\nThe answer is {row['answer']}."
    return _row("gsm8k", index, "math", question, response, answer=row["answer"], scorer="number")


def _trace_examples(seed: int) -> Iterable[dict]:
    rng = random.Random(seed)
    for index in range(12_000):
        numbers = rng.sample(range(-20, 30), rng.randrange(4, 9))
        threshold = rng.randrange(-8, 15)
        operation = index % 3
        if operation == 0:
            expr = f"sum(x for x in values if x > {threshold})"
            answer = sum(x for x in numbers if x > threshold)
            steps = f"The values above {threshold} are {[x for x in numbers if x > threshold]}."
        elif operation == 1:
            expr = "values[1::2][::-1]"
            answer = numbers[1::2][::-1]
            steps = f"The slice selects {numbers[1::2]}, then reverses it."
        else:
            expr = f"[x * x for x in values if x < {threshold}]"
            answer = [x * x for x in numbers if x < threshold]
            steps = f"It squares the selected values {[x for x in numbers if x < threshold]} in order."
        prompt = f"What does this Python code print?\n\n```python\nvalues = {numbers}\nprint({expr})\n```"
        yield _row("generated-python-traces", index, "programming", prompt, f"{steps}\n\nIt prints `{answer!r}`.")


def _science_inference(seed: int) -> Iterable[dict]:
    from datasets import load_dataset

    dataset = load_dataset("allenai/qasc", revision=QASC_REVISION, split="train").shuffle(seed=seed)
    for index, item in enumerate(dataset):
        options = dict(zip(item["choices"]["label"], item["choices"]["text"], strict=True))
        facts = f"{item['fact1']}\n{item['fact2']}"
        prompt = f"Use these facts to answer the question.\n\n{facts}\n\n{item['formatted_question']}"
        key = item["answerKey"]
        response = f"{item['combinedfact']} Therefore, the best answer is ({key}) {options[key]}."
        yield _row("qasc", index, "reasoning", prompt, response)


def _logic_examples(seed: int) -> Iterable[dict]:
    rng = random.Random(seed)
    objects = ["atlas", "binder", "compass", "drum", "envelope", "flag", "globe", "hammer"]
    categories = ["artists", "bakers", "cyclists", "divers", "editors", "farmers", "gardeners", "historians"]
    for index in range(16_000):
        if index % 2:
            order = rng.sample(objects, rng.randrange(3, 7))
            clues = [f"The {left} is somewhere left of the {right}." for left, right in zip(order, order[1:])]
            rng.shuffle(clues)
            position = rng.randrange(len(order))
            question = f"{' '.join(clues)} The shelf contains only these objects. Which is in position {position + 1} from the left?"
            response = f"The clues give this order from left to right: {', '.join(order)}. Position {position + 1} is the {order[position]}."
        else:
            chain = rng.sample(categories, rng.randrange(3, 7))
            name = rng.choice(("Alex", "Blair", "Casey", "Drew", "Elliot", "Frankie", "Gale", "Harper"))
            rules = [f"All {left} are {right}." for left, right in zip(chain, chain[1:])]
            rng.shuffle(rules)
            direction = index % 4 == 0
            start, target = (chain[0], chain[-1]) if direction else (chain[-1], chain[0])
            question = f"{' '.join(rules)} {name} is in the group of {start}. Must {name} also be in the group of {target}?"
            response = (
                f"Yes. Follow the rules in order: {' → '.join(chain)}."
                if direction else
                f"No. The rules imply membership from {target} toward {start}. {name} could be among the {start} without being among the {target}."
            )
        yield _row("generated-logic", index, "reasoning", question, response)


def _instruction_checks(seed: int) -> list[dict]:
    rng = random.Random(seed)
    terms = ["amber", "birch", "coral", "dune", "fern", "grove", "lunar", "maple", "opal", "pearl",
             "reed", "slate", "tide", "willow", "zinc"]
    rows = []
    seen = set()
    while len(rows) < 96:
        index = len(rows)
        words = rng.sample(terms, 4)
        if index % 3 == 0:
            question = f"Put these words in alphabetical order: {', '.join(words)}. Reply with just the words separated by commas."
            answer = ", ".join(sorted(words))
            scorer = "comma_list"
        elif index % 3 == 1:
            question = f"Reply with only the following phrase in uppercase: {' '.join(words)}."
            answer = " ".join(words).upper()
            scorer = "text"
        else:
            question = f"Return only a JSON object with keys left and right. Give left the value {words[0]} and right the value {words[1]}."
            answer = json.dumps({"left": words[0], "right": words[1]})
            scorer = "json"
        if question in seen:
            continue
        seen.add(question)
        rows.append(_row("instruction-checks", index, "instruction", question, answer, answer=answer, scorer=scorer))
    return rows


def _eval_splits(seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    from datasets import load_dataset

    rng = random.Random(seed)
    math_train, math_test = _load_gsm8k(GSM8K_REVISION)
    rng.shuffle(math_train)
    dev = [_math(row, index, evaluation=True) for index, row in enumerate(math_train[:96])]
    test = [_math(row, index + 20_000, evaluation=True) for index, row in enumerate(math_test)]
    train = [_math(row, index + 96, evaluation=False) for index, row in enumerate(math_train[96:])]

    code = []
    for index, row in enumerate(load_dataset("cruxeval-org/cruxeval", revision=CRUX_REVISION, split="test")):
        # CRUXEval-O has literal outputs. Exclude references outside that scoring protocol.
        try:
            ast.literal_eval(row["output"])
        except (ValueError, SyntaxError):
            continue
        question = (
            f"```python\n{row['code']}\n```\n\nWhat does `f({row['input']})` return? "
            "Reply with only the Python literal for its return value."
        )
        code.append(_row("cruxeval-o", index, "programming", question, row["output"], answer=row["output"], scorer="literal"))
    if len(code) < 750:
        raise ValueError("CRUXEval-O has fewer supported cases than expected")
    rng.shuffle(code)
    dev.extend(code[:96])
    test.extend(code[96:])

    for task in BBH_TASKS:
        url = f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{BBH_REVISION}/bbh/{task}.json"
        print(f"Loading BBH {task}", flush=True)
        with urlopen(url, timeout=30) as file:
            examples = json.load(file)["examples"]
        rows = [
            _row(f"bbh/{task}", index, "reasoning",
                 f"{row['input']}\n\nGive only the answer, using the option letter if choices are listed.",
                 row["target"], answer=row["target"],
                 scorer="choice" if row["target"].startswith("(") else "boolean")
            for index, row in enumerate(examples)
        ]
        if len(rows) <= 24:
            raise ValueError(f"BBH subset too small: {task}")
        rng.shuffle(rows)
        dev.extend(rows[:24])
        test.extend(rows[24:])

    checks = _instruction_checks(seed)
    dev.extend(checks[:48])
    test.extend(checks[48:])
    rng.shuffle(dev)
    return train, dev, test


def _take_rows(
    source: Iterable[dict], count: int, seen: set[str], tokenizer, seq_len: int,
) -> list[dict]:
    from src.sft_data import encode_sft_pair

    result = []
    for row in source:
        key = _key(row["question"])
        if key in seen:
            continue
        src, tgt = encode_sft_pair(row, tokenizer)
        if len(src) + len(tgt) > seq_len + 1:
            continue
        seen.add(key)
        result.append(row)
        if len(result) == count:
            return result
    raise ValueError(f"only {len(result)} of {count} complete training examples available")


def _fixture() -> dict[str, list[dict]]:
    cases = {
        "math": ("How much is {n} plus 3?", "{answer}", "number"),
        "programming": ("What does Python return for len([{n}])?", "1", "literal"),
        "reasoning": ("Statement {n}: Can a named item be in the listed group? Every named item is in it.", "True", "boolean"),
        "instruction": ("Echo the capitalized label ITEM{n}.", "ITEM{n}", "text"),
    }
    return {
        split: [
            _row(f"fixture-{split}-{domain}", n, domain,
                 prompt.format(n=n), response.format(n=n, answer=n + 3),
                 answer=response.format(n=n, answer=n + 3), scorer=scorer)
            for domain, (prompt, response, scorer) in cases.items()
            for n in range(offset, offset + (4 if split == "train" else 1))
        ]
        for offset, split in zip((0, 10, 20), SPLITS)
    }


def validate(output: Path, *, fixture: bool = False) -> dict:
    manifest = load_manifest(output / "manifest.json")
    if manifest.get("profile") != "general" or manifest.get("fixture") is not fixture or manifest.get("version") != 1:
        raise ValueError(f"invalid general reasoning manifest in {output}")
    entries = manifest.get("datasets", [])
    if [entry.get("name") for entry in entries] != list(SPLITS):
        raise ValueError("general reasoning data requires train, dev and test splits")
    for entry in entries:
        if (
            entry.get("path") != f"{entry['name']}.jsonl"
            or not entry.get("sha256")
            or set(entry.get("domains", {})) != set(DOMAINS)
            or any(n < 1 for n in entry["domains"].values())
            or entry["rows"] != sum(entry["domains"].values())
        ):
            raise ValueError(f"invalid general reasoning split {entry['name']}")
    errors = verify_manifest_hashes(manifest, output)
    if errors:
        raise ValueError("; ".join(errors))
    return manifest


def prepare(output: Path, *, fixture: bool = False, seed: int = 42) -> dict:
    if output.exists():
        manifest = validate(output, fixture=fixture)
        if manifest["seed"] != seed:
            raise ValueError("existing data uses a different seed")
        return manifest

    sources = {}
    if fixture:
        splits = _fixture()
    else:
        from huggingface_hub import snapshot_download
        from src.hf_adapter import HFTokenizer

        candidates, dev, test = _eval_splits(seed)
        # Hold out exact question text even where an evaluation instruction was appended.
        seen = {_key(r["question"]) for r in dev + test}
        if len(seen) != len(dev) + len(test):
            raise ValueError("duplicate questions in the held-out benchmarks")
        snapshot = snapshot_download(
            DEFAULT_MODEL, revision=DEFAULT_REVISION,
            allow_patterns=["tokenizer*", "config.json", "merges.txt", "vocab.json", "chat_template*"],
        )
        tokenizer = HFTokenizer.from_pretrained(snapshot)

        def take(rows: Iterable[dict], count: int) -> list[dict]:
            return _take_rows(rows, count, seen, tokenizer, GENERAL_SEQUENCE_LENGTH)

        train = take(candidates, 2500)
        print("Preparing programming examples", flush=True)
        train += take(_smoltalk("self-oss-instruct", "programming", seed), 2000)
        train += take(_trace_examples(seed), 500)
        print("Preparing general reasoning examples", flush=True)
        train += take(_logic_examples(seed), 1250)
        train += take(_science_inference(seed), 1250)
        for subset in INSTRUCTION_SOURCES:
            print(f"Preparing {subset}", flush=True)
            train += take(_smoltalk(subset, "instruction", seed), 625)
        random.Random(seed).shuffle(train)
        splits = {"train": train, "dev": dev, "test": test}
        sources = {
            "openai/gsm8k": {"revision": GSM8K_REVISION, "license": "MIT"},
            "HuggingFaceTB/smoltalk": {
                "revision": SMOLTALK_REVISION,
                "subsets": ["self-oss-instruct", *INSTRUCTION_SOURCES],
                "license": "See source dataset card and the license for each component",
            },
            "allenai/qasc": {"revision": QASC_REVISION, "license": "CC-BY-4.0"},
            "cruxeval-org/cruxeval": {"revision": CRUX_REVISION, "license": "MIT"},
            "suzgunmirac/BIG-Bench-Hard": {"revision": BBH_REVISION, "tasks": BBH_TASKS, "license": "MIT"},
            "generated": ["Python traces", "logic exercises", "instruction checks"],
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        staging = Path(temp) / "data"
        staging.mkdir()
        entries = []
        for split, rows in splits.items():
            path = staging / f"{split}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as file:
                for row in rows:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
            entries.append({"name": split, "path": path.name, "rows": len(rows), "domains": dict(Counter(r["domain"] for r in rows))})
        manifest = refresh_hashes({
            "version": 1, "profile": "general", "fixture": fixture, "seed": seed,
            "sources": sources, "sequence_length": GENERAL_SEQUENCE_LENGTH,
            "tokenizer": {"model": DEFAULT_MODEL, "revision": DEFAULT_REVISION},
            "datasets": entries,
        }, staging)
        save_manifest(manifest, staging / "manifest.json")
        validate(staging, fixture=fixture)
        staging.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(GENERAL_DATA_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true", help="check an existing data build")
    args = parser.parse_args()
    data = validate(args.out) if args.verify else prepare(args.out, seed=args.seed)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
