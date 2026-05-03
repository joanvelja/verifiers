from __future__ import annotations

from typing import Any, Sequence

from datasets import Dataset

import verifiers as vf
from verifiers.utils.hf_tasks import (
    AnswerFormat,
    DEFAULT_JUDGE_PROMPT_PACK,
    JudgePromptKind,
    TaskType,
    load_hf_dataset,
    make_rubric,
    normalize_hf_dataset,
)


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
    task_type: TaskType = "open_ended",
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
    prompts_ref: str | None = None,
    prompt_role: str = "debater_a",
    prompt_phase: str = "propose",
    shuffle_choices: bool = False,
    seed: int = 0,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    judge_client: Any | None = None,
    judge_model: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key_var: str = "OPENAI_API_KEY",
    judge_sampling_args: dict[str, Any] | None = None,
    judge_prompt_pack: str | None = DEFAULT_JUDGE_PROMPT_PACK,
    judge_prompt_kind: JudgePromptKind = "grader",
    judge_system_prompt: str | None = None,
    judge_prompt: str | None = None,
    judge_positive_label: str | None = None,
    judge_negative_label: str | None = None,
    judge_cache_enabled: bool = True,
    judge_cache_size: int = 10000,
    judge_max_retries: int = 2,
    judge_retry_delay_s: float = 0.5,
    judge_persistent_cache_path: str | None = None,
    judge_persistent_cache_min_samples: int = 1,
    judge_rubric_family: str | None = None,
    judge_variant_id: str | None = None,
    judge_reward_mode: str | None = None,
    judge_panel_threshold: float | None = None,
    judge_panel_prior_alpha: float | None = None,
    judge_panel_prior_beta: float | None = None,
    judge_repeated_call_correlation: float | None = None,
    judge_calibration_mode: str | None = None,
    judge_correctness_prior: float | None = None,
    judge_sensitivity: float | None = None,
    judge_false_positive_rate: float | None = None,
    **extra: Any,
) -> vf.Environment:
    resolved_dataset = dataset
    if resolved_dataset is None and dataset_name is None:
        resolved_dataset = _default_dataset(task_type)
        if task_type == "mcq" and choices_key is None and choice_keys is None:
            choices_key = "choices"
            answer_format = "index"

    resolved_judge_prompt_pack = (
        prompts_ref
        if judge_prompt_pack == DEFAULT_JUDGE_PROMPT_PACK and prompts_ref is not None
        else judge_prompt_pack
    )

    rubric = make_rubric(
        task_type=task_type,
        judge_client=judge_client,
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        judge_api_key_var=judge_api_key_var,
        judge_sampling_args=judge_sampling_args,
        judge_prompt_pack=resolved_judge_prompt_pack,
        judge_prompt_kind=judge_prompt_kind,
        **(
            {"judge_system_prompt": judge_system_prompt}
            if judge_system_prompt is not None
            else {}
        ),
        **({"judge_prompt": judge_prompt} if judge_prompt is not None else {}),
        **(
            {"judge_positive_label": judge_positive_label}
            if judge_positive_label is not None
            else {}
        ),
        **(
            {"judge_negative_label": judge_negative_label}
            if judge_negative_label is not None
            else {}
        ),
        judge_cache_enabled=judge_cache_enabled,
        judge_cache_size=judge_cache_size,
        judge_max_retries=judge_max_retries,
        judge_retry_delay_s=judge_retry_delay_s,
        judge_persistent_cache_path=judge_persistent_cache_path,
        judge_persistent_cache_min_samples=judge_persistent_cache_min_samples,
        judge_rubric_family=judge_rubric_family,
        judge_variant_id=judge_variant_id,
        judge_reward_mode=judge_reward_mode,
        judge_panel_threshold=judge_panel_threshold,
        judge_panel_prior_alpha=judge_panel_prior_alpha,
        judge_panel_prior_beta=judge_panel_prior_beta,
        judge_repeated_call_correlation=judge_repeated_call_correlation,
        judge_calibration_mode=judge_calibration_mode,
        judge_correctness_prior=judge_correctness_prior,
        judge_sensitivity=judge_sensitivity,
        judge_false_positive_rate=judge_false_positive_rate,
    )

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
        return normalize_hf_dataset(
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
            prompts_ref=prompts_ref,
            prompt_role=prompt_role,
            prompt_phase=prompt_phase,
            shuffle_choices=shuffle_choices,
            seed=seed,
            num_examples=num_train_examples,
        )

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
        return normalize_hf_dataset(
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
            prompts_ref=prompts_ref,
            prompt_role=prompt_role,
            prompt_phase=prompt_phase,
            shuffle_choices=shuffle_choices,
            seed=seed + 1,
            num_examples=num_eval_examples,
        )

    return vf.SingleTurnEnv(
        dataset=build_dataset,
        eval_dataset=build_eval_dataset,
        parser=rubric.parser,
        rubric=rubric,
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
