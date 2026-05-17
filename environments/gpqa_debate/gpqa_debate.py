"""GPQA under a two-debater plus judge protocol."""

import random

from datasets import Dataset, load_dataset

import verifiers as vf
from verifiers.protocols.debate.env import (
    DebateEnv,
    load_environment as load_debate_env,
)

LETTERS = ("A", "B", "C", "D")

DEFAULT_SCHEDULE = [
    {"slot_id": 0, "agents": ["debater_a"], "phase": "propose"},
    {"slot_id": 1, "agents": ["debater_b"], "phase": "propose"},
    {"slot_id": 2, "agents": ["debater_a"], "phase": "critique"},
    {"slot_id": 3, "agents": ["debater_b"], "phase": "critique"},
    {"slot_id": 4, "agents": ["judge"], "phase": "final"},
]


def load_environment(
    prompts_ref: str = "selfplay",
    subset: str = "gpqa_diamond",
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    seed: int = 0,
    schedule: list[dict] | None = None,
    truth_member: str = "debater_a",
) -> DebateEnv:
    def build_split(n: int, split_seed: int) -> Dataset:
        vf.ensure_keys(["HF_TOKEN"])
        raw = list(load_dataset("Idavidrein/gpqa", subset, split="train"))
        rows = raw if n == -1 else raw[:n]
        rng = random.Random(split_seed)

        def format_row(row: dict, example_idx: int) -> dict:
            correct = row["Correct Answer"].strip()
            choices = [
                correct,
                row["Incorrect Answer 1"].strip(),
                row["Incorrect Answer 2"].strip(),
                row["Incorrect Answer 3"].strip(),
            ]
            rng.shuffle(choices)
            truth_letter = LETTERS[choices.index(correct)]
            body = "\n".join(
                f"{letter}) {choice}" for letter, choice in zip(LETTERS, choices)
            )
            example_id = str(row.get("Record ID") or f"gpqa_{example_idx}")
            return {
                "prompt": [
                    {
                        "role": "user",
                        "content": (
                            f"{row['Question'].strip()}\n\n{body}\n\n"
                            "Reply with exactly one of A, B, C, or D as your <answer>."
                        ),
                    }
                ],
                "answer": truth_letter,
                "example_id": example_id,
                "info": {"env_id": "gpqa_debate"},
            }

        return Dataset.from_list(
            [format_row(row, example_idx=idx) for idx, row in enumerate(rows)]
        )

    def build_dataset() -> Dataset:
        return build_split(num_train_examples, seed)

    def build_eval_dataset() -> Dataset:
        return build_split(num_eval_examples, seed + 1)

    return load_debate_env(
        schedule_slots=schedule or DEFAULT_SCHEDULE,
        members=["debater_a", "debater_b", "judge"],
        truth_member=truth_member,
        prompts_ref=prompts_ref,
        dataset=build_dataset,
        eval_dataset=build_eval_dataset,
    )
