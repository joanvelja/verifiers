"""GPQA as single-agent RLVR — verifier-graded multiple-choice.

The dataset row format mirrors ``gpqa_debate``: question + 4 shuffled
choices, ground truth as a single letter A/B/C/D. Reward is exact-match
on the parsed letter — no LLM judge involved.

Compare against ``gpqa_consultancy`` (same dataset, same model, judge
grades instead of verifier) and ``gpqa_debate`` (same dataset, two
debaters, judge picks winner).

Launch
------
    export HF_TOKEN=...                  # GPQA is gated
    vf-install gpqa_rlvr
    vf-eval gpqa_rlvr -n 198 -r 4 --model openai/gpt-4.1-mini

env_args
--------
    subset             str    gpqa_diamond | gpqa_main   "gpqa_diamond"
    num_train_examples int    -1 = all                   -1
    num_eval_examples  int    -1 = all                   -1
    seed               int    choice-shuffling seed      0
"""

from __future__ import annotations

import random
from typing import Any

from datasets import Dataset, load_dataset

import verifiers as vf
from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.rubric import Rubric

LETTERS = ("A", "B", "C", "D")


def _format_row(row: dict, rng: random.Random) -> dict:
    """Same shape as gpqa_debate._format_row — letter answer, shuffled choices."""
    correct = row["Correct Answer"].strip()
    choices = [
        correct,
        row["Incorrect Answer 1"].strip(),
        row["Incorrect Answer 2"].strip(),
        row["Incorrect Answer 3"].strip(),
    ]
    rng.shuffle(choices)
    truth_letter = LETTERS[choices.index(correct)]
    body = "\n".join(f"{L}) {c}" for L, c in zip(LETTERS, choices))
    return {
        "prompt": [
            {
                "role": "user",
                "content": (
                    f"{row['Question'].strip()}\n\n{body}\n\n"
                    "Think step by step, then reply with exactly one of A, B, C, "
                    "or D inside <answer>...</answer> tags."
                ),
            }
        ],
        "answer": truth_letter,
        "example_id": str(row.get("Record ID") or f"gpqa_{rng.random():.10f}"),
        "task": "gpqa_rlvr",
    }


def _build_dataset(subset: str, n: int, seed: int) -> Dataset:
    vf.ensure_keys(["HF_TOKEN"])
    raw = list(load_dataset("Idavidrein/gpqa", subset, split="train"))
    rng = random.Random(seed)
    rows = raw if n == -1 else raw[:n]
    return Dataset.from_list([_format_row(r, rng) for r in rows])


class _LetterMatchRubric(Rubric):
    """Reward = 1 if parsed letter matches ground truth, 0 otherwise.

    Strict exact-match on the first letter inside <answer>...</answer>.
    No fuzzy matching — multiple-choice grading is a categorical decision,
    a parser failure should drop reward to 0 (signal: the model didn't
    produce a parsable answer), not silently soft-score a partial match.
    """

    def __init__(self) -> None:
        super().__init__(parser=XMLParser(["answer"], answer_field="answer"))
        self.add_reward_func(self.letter_match)

    async def letter_match(self, parser, completion, answer, **kwargs) -> float:
        parsed = parser.parse_answer(completion) or ""
        # Take the first uppercase letter A-D in the parsed string.
        for ch in parsed.upper():
            if ch in LETTERS:
                return 1.0 if ch == answer else 0.0
        return 0.0


def load_environment(
    subset: str = "gpqa_diamond",
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    seed: int = 0,
    **extra: Any,
) -> vf.Environment:
    """vf-eval entry point."""

    def build_dataset() -> Dataset:
        return _build_dataset(subset, num_train_examples, seed)

    def build_eval_dataset() -> Dataset:
        # Same source; eval slice is independent so train/eval sizes can diverge.
        return _build_dataset(subset, num_eval_examples, seed + 1)

    rubric = _LetterMatchRubric()
    return vf.SingleTurnEnv(
        dataset=build_dataset,
        eval_dataset=build_eval_dataset,
        parser=rubric.parser,
        rubric=rubric,
        **extra,
    )
