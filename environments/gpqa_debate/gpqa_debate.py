"""GPQA under a two-debater plus judge protocol."""

import random

from datasets import Dataset, load_dataset

import verifiers as vf
from verifiers.protocols.debate.env import (
    DebateEnv,
    load_environment as load_debate_env,
)
from verifiers.utils.hf_tasks import split_train_for_eval

LETTERS = ("A", "B", "C", "D")

DEFAULT_SCHEDULE = [
    {"slot_id": 0, "agents": ["debater_a", "debater_b"], "phase": "propose"},
    {"slot_id": 1, "agents": ["debater_a", "debater_b"], "phase": "critique"},
    {"slot_id": 2, "agents": ["judge"], "phase": "final"},
]


def load_environment(
    prompts_ref: str = "selfplay",
    subset: str = "gpqa_diamond",
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    seed: int = 0,
    schedule: list[dict] | None = None,
    truth_member: str | None = None,
) -> DebateEnv:
    def build_split(split: str) -> Dataset:
        vf.ensure_keys(["HF_TOKEN"])
        raw = load_dataset("Idavidrein/gpqa", subset, split="train")
        if (
            num_train_examples >= 0
            and num_eval_examples >= 0
            and num_train_examples + num_eval_examples > len(raw)
        ):
            raise ValueError(
                f"num_train_examples + num_eval_examples = "
                f"{num_train_examples + num_eval_examples} exceeds the "
                f"{len(raw)} rows of Idavidrein/gpqa:{subset} — train and "
                "eval are disjoint holdouts of the same source split."
            )
        # GPQA ships only a train split, so carve a disjoint eval holdout the
        # same way the HF debate loader derives one (seeded shuffle once, eval
        # rows first, train from the remainder) — eval never overlaps train.
        train_raw, eval_raw = split_train_for_eval(
            raw,
            seed=seed,
            num_train_examples=num_train_examples,
            num_eval_examples=num_eval_examples,
        )
        rows = train_raw if split == "train" else eval_raw
        rng = random.Random(seed if split == "train" else seed + 1)

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
        return build_split("train")

    def build_eval_dataset() -> Dataset:
        return build_split("eval")

    return load_debate_env(
        schedule_slots=schedule or DEFAULT_SCHEDULE,
        members=["debater_a", "debater_b", "judge"],
        truth_member=truth_member,
        prompts_ref=prompts_ref,
        dataset=build_dataset,
        eval_dataset=build_eval_dataset,
    )
