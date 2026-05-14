from __future__ import annotations

import json
from typing import Any, Sequence

from datasets import Dataset

import verifiers as vf
from verifiers.envs.debate.fields import EnumScoring
from verifiers.envs.debate_env import load_environment as load_debate_environment
from verifiers.envs.debate.prompts import resolve_prompts
from verifiers.utils.hf_tasks import (
    AnswerFormat,
    TaskType,
    load_hf_dataset,
    normalize_hf_dataset,
)

DEFAULT_SCHEDULE = [
    {"slot_id": 0, "agents": ["debater_a"], "phase": "propose"},
    {"slot_id": 1, "agents": ["debater_b"], "phase": "propose"},
    {"slot_id": 2, "agents": ["debater_a"], "phase": "critique"},
    {"slot_id": 3, "agents": ["debater_b"], "phase": "critique"},
    {"slot_id": 4, "agents": ["judge"], "phase": "final"},
]


def load_environment(
    dataset_name: str | None = None,
    dataset_subset: str | None = None,
    dataset_split: str = "train",
    eval_dataset_split: str | None = None,
    data_files: str | list[str] | dict[str, str | list[str]] | None = None,
    dataset_streaming: bool = False,
    dataset_columns: Sequence[str] | None = None,
    dataset_streaming_shuffle_buffer_size: int | None = None,
    dataset: Dataset | None = None,
    eval_dataset: Dataset | None = None,
    task_type: TaskType = "mcq",
    question_key: str = "question",
    answer_key: str = "answer",
    choices_key: str | None = None,
    choice_keys: Sequence[str] | None = None,
    answer_format: AnswerFormat = "auto",
    example_id_key: str | None = None,
    task_name: str | None = None,
    info_keys: Sequence[str] | None = None,
    include_raw_info: bool = False,
    prompt_template: str | None = None,
    system_prompt: str | None = None,
    shuffle_choices: bool = False,
    seed: int = 0,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    schedule: list[dict[str, Any]] | None = None,
    prompts_ref: str | None = None,
    truth_member: str = "debater_a",
    judge_model: str = "gpt-5.4-nano",
    **extra: Any,
) -> vf.Environment:
    if system_prompt is not None:
        raise ValueError(
            "hf_debate does not consume row-level system_prompt. Put debate "
            "role system prompts in the selected prompt pack instead."
        )

    resolved_dataset = dataset
    if resolved_dataset is None and dataset_name is None:
        resolved_dataset = _default_dataset(task_type)
        if task_type == "mcq" and choices_key is None and choice_keys is None:
            choices_key = "choices"
            answer_format = "index"

    def build_dataset() -> Dataset:
        if resolved_dataset is not None:
            raw = resolved_dataset
        else:
            raw = _load_required_dataset(
                dataset_name=dataset_name,
                dataset_subset=dataset_subset,
                dataset_split=dataset_split,
                data_files=data_files,
                streaming=dataset_streaming,
                columns=dataset_columns,
                streaming_limit=num_train_examples,
                streaming_shuffle_buffer_size=dataset_streaming_shuffle_buffer_size,
                streaming_seed=seed,
            )
        normalized = normalize_hf_dataset(
            raw,
            task_type=task_type,
            question_key=question_key,
            answer_key=answer_key,
            choices_key=choices_key,
            choice_keys=choice_keys,
            answer_format=answer_format,
            example_id_key=example_id_key,
            task_name=task_name,
            info_keys=info_keys,
            include_raw_info=include_raw_info,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            shuffle_choices=shuffle_choices,
            seed=seed,
            num_examples=num_train_examples,
        )
        if task_type == "mcq":
            _validate_mcq_labels_for_prompt_pack(normalized, resolved_prompts_ref)
        return normalized

    def build_eval_dataset() -> Dataset:
        if eval_dataset is not None:
            raw = eval_dataset
        elif resolved_dataset is not None:
            raw = resolved_dataset
        else:
            raw = _load_required_dataset(
                dataset_name=dataset_name,
                dataset_subset=dataset_subset,
                dataset_split=eval_dataset_split or dataset_split,
                data_files=data_files,
                streaming=dataset_streaming,
                columns=dataset_columns,
                streaming_limit=num_eval_examples,
                streaming_shuffle_buffer_size=dataset_streaming_shuffle_buffer_size,
                streaming_seed=seed + 1,
            )
        normalized = normalize_hf_dataset(
            raw,
            task_type=task_type,
            question_key=question_key,
            answer_key=answer_key,
            choices_key=choices_key,
            choice_keys=choice_keys,
            answer_format=answer_format,
            example_id_key=example_id_key,
            task_name=task_name,
            info_keys=info_keys,
            include_raw_info=include_raw_info,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            shuffle_choices=shuffle_choices,
            seed=seed + 1,
            num_examples=num_eval_examples,
        )
        if task_type == "mcq":
            _validate_mcq_labels_for_prompt_pack(normalized, resolved_prompts_ref)
        return normalized

    resolved_prompts_ref = prompts_ref or (
        "selfplay" if task_type == "mcq" else "default"
    )
    return load_debate_environment(
        schedule_slots=schedule or DEFAULT_SCHEDULE,
        members=["debater_a", "debater_b", "judge"],
        truth_member=truth_member,
        prompts_ref=resolved_prompts_ref,
        judge_model=judge_model,
        dataset=build_dataset,
        eval_dataset=build_eval_dataset,
        **extra,
    )


def _load_required_dataset(
    *,
    dataset_name: str | None,
    dataset_subset: str | None,
    dataset_split: str,
    data_files: str | list[str] | dict[str, str | list[str]] | None,
    streaming: bool,
    columns: Sequence[str] | None,
    streaming_limit: int,
    streaming_shuffle_buffer_size: int | None,
    streaming_seed: int,
) -> Dataset:
    if dataset_name is None:
        raise ValueError("dataset_name is required when dataset is not provided")
    return load_hf_dataset(
        dataset_name,
        dataset_subset=dataset_subset,
        dataset_split=dataset_split,
        data_files=data_files,
        streaming=streaming,
        columns=columns,
        streaming_limit=streaming_limit,
        streaming_shuffle_buffer_size=streaming_shuffle_buffer_size,
        streaming_seed=streaming_seed,
    )


def _validate_mcq_labels_for_prompt_pack(dataset: Dataset, prompts_ref: str) -> None:
    allowed = _debater_answer_enum_values(prompts_ref)
    if not allowed:
        raise ValueError(
            f"Prompt pack {prompts_ref!r} has no enum-scored debater answer field"
        )
    for row in dataset:
        info = json.loads(row.get("info") or "{}")
        labels = {str(label).upper() for label in info.get("choice_labels", [])}
        if not labels.issubset(allowed):
            raise ValueError(
                f"Prompt pack {prompts_ref!r} accepts MCQ labels "
                f"{sorted(allowed)}, but dataset row uses {sorted(labels)}"
            )


def _debater_answer_enum_values(prompts_ref: str) -> set[str]:
    prompts = resolve_prompts(prompts_ref)
    values: set[str] = set()
    for role, triggers in prompts.fields.items():
        if not role.startswith("debater_"):
            continue
        for fields in triggers.values():
            answer = fields.get("answer")
            scoring = answer.scoring if answer is not None else None
            if isinstance(scoring, EnumScoring):
                values.update(str(value).upper() for value in scoring.values)
    return values


def _default_dataset(task_type: TaskType) -> Dataset:
    if task_type == "mcq":
        return Dataset.from_list(
            [
                {
                    "question": "What is 2 + 2?",
                    "choices": ["3", "4", "5", "6"],
                    "answer": 1,
                }
            ]
        )
    return Dataset.from_list([{"question": "What is 2 + 2?", "answer": "4"}])
