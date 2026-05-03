from __future__ import annotations

import json
import random
import re
import string
from collections.abc import Sequence
from typing import Any, Literal, cast

import fsspec
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset

import verifiers as vf
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.envs.debate.prompts import build_context, resolve_prompts
from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.judge_rubric import JudgeRubric
from verifiers.rubrics.rubric import Rubric
from verifiers.types import ClientConfig, Messages, State
from verifiers.utils.judge_prompts import normalize_verdict_token, resolve_judge_prompts
from verifiers.utils.mcq import normalize_mcq

TaskType = Literal["mcq", "math", "open_ended", "exact"]
AnswerFormat = Literal["auto", "index", "letter", "text"]
JudgePromptKind = Literal["grader", "matcher"]

LETTERS = tuple(string.ascii_uppercase)
DEFAULT_JUDGE_PROMPT_PACK = "default"

MCQ_INSTRUCTION = (
    "Think step by step, then put your final answer letter inside "
    "<answer>...</answer> tags."
)
MATH_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
OPEN_ENDED_INSTRUCTION = (
    "Think step by step, then put your final answer inside <answer>...</answer> tags."
)
EXACT_INSTRUCTION = (
    "Think step by step, then put your final answer inside <answer>...</answer> tags."
)
FINAL_ANSWER_LABEL_RE = re.compile(r"\bfinal\s+answer\s*:\s*(.+)\s*$", re.IGNORECASE)
ANSWER_LABEL_RE = re.compile(r"(?:^|\n)\s*answer\s*:\s*(.+)\s*$", re.IGNORECASE)
ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def load_hf_dataset(
    dataset_name: str,
    *,
    dataset_subset: str | None = None,
    dataset_split: str = "train",
    data_files: str | list[str] | dict[str, str | list[str]] | None = None,
    streaming: bool = False,
    columns: Sequence[str] | None = None,
    streaming_limit: int = -1,
    streaming_shuffle_buffer_size: int | None = None,
    streaming_seed: int = 0,
) -> Dataset | IterableDataset:
    if streaming and dataset_name == "parquet" and data_files is not None:
        return _load_streaming_parquet_dataset(
            data_files=data_files,
            columns=columns,
            limit=streaming_limit,
            shuffle_buffer_size=streaming_shuffle_buffer_size,
            seed=streaming_seed,
        )

    kwargs: dict[str, Any] = {"split": dataset_split, "streaming": streaming}
    if data_files is not None:
        kwargs["data_files"] = data_files
    if columns is not None:
        kwargs["columns"] = list(columns)
    if dataset_subset is None:
        loaded = load_dataset(dataset_name, **kwargs)
    else:
        loaded = load_dataset(dataset_name, dataset_subset, **kwargs)
    if isinstance(loaded, DatasetDict):
        raise TypeError(
            "load_hf_dataset expected a split-specific Dataset, got DatasetDict. "
            "Pass dataset_split explicitly."
        )
    if not isinstance(loaded, Dataset | IterableDataset):
        raise TypeError(f"Expected datasets.Dataset, got {type(loaded).__name__}")
    if isinstance(loaded, IterableDataset) and streaming_shuffle_buffer_size:
        loaded = loaded.shuffle(
            seed=streaming_seed,
            buffer_size=streaming_shuffle_buffer_size,
        )
    return loaded


def _load_streaming_parquet_dataset(
    *,
    data_files: str | list[str] | dict[str, str | list[str]],
    columns: Sequence[str] | None,
    limit: int,
    shuffle_buffer_size: int | None,
    seed: int,
) -> Dataset:
    if limit < 0:
        raise ValueError("streaming_limit must be set when streaming parquet data")

    rows: list[dict[str, Any]] = []
    read_limit = max(limit, shuffle_buffer_size or 0)
    for path in _iter_data_file_paths(data_files):
        with fsspec.open(path, "rb") as f:
            parquet_file = pq.ParquetFile(f)
            for batch in parquet_file.iter_batches(
                batch_size=min(max(read_limit - len(rows), 1), 1024),
                columns=list(columns) if columns is not None else None,
            ):
                rows.extend(batch.to_pylist())
                if len(rows) >= read_limit:
                    return _dataset_from_streaming_rows(
                        rows,
                        limit=limit,
                        shuffle_buffer_size=shuffle_buffer_size,
                        seed=seed,
                    )
    return _dataset_from_streaming_rows(
        rows,
        limit=limit,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
    )


def _dataset_from_streaming_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    shuffle_buffer_size: int | None,
    seed: int,
) -> Dataset:
    if shuffle_buffer_size:
        random.Random(seed).shuffle(rows)
    return Dataset.from_list(rows[:limit])


def _iter_data_file_paths(
    data_files: str | list[str] | dict[str, str | list[str]],
) -> list[str]:
    if isinstance(data_files, str):
        if any(char in data_files for char in "*?["):
            return [
                getattr(file, "full_name", file.path)
                for file in fsspec.open_files(data_files)
            ]
        return [data_files]
    if isinstance(data_files, list):
        paths: list[str] = []
        for item in data_files:
            paths.extend(_iter_data_file_paths(item))
        return paths

    paths = []
    for value in data_files.values():
        paths.extend(_iter_data_file_paths(value))
    return paths


def _materialize_hf_examples(
    dataset: Dataset | IterableDataset,
    *,
    num_examples: int,
) -> Dataset:
    if isinstance(dataset, IterableDataset):
        if num_examples < 0:
            raise ValueError(
                "num_examples must be set when normalizing a streaming dataset"
            )
        rows = list(dataset.take(num_examples))
        features = getattr(dataset, "features", None)
        if features is not None:
            return Dataset.from_list(rows, features=features)
        return Dataset.from_list(rows)

    if num_examples > -1:
        return dataset.select(range(min(num_examples, len(dataset))))
    return dataset


def normalize_hf_dataset(
    dataset: Dataset | IterableDataset,
    *,
    task_type: TaskType,
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
    num_examples: int = -1,
) -> Dataset:
    dataset = _materialize_hf_examples(dataset, num_examples=num_examples)
    if choices_key is not None and choice_keys is not None:
        raise ValueError("Use either choices_key or choice_keys, not both")
    if question_key not in dataset.column_names:
        raise ValueError(
            f"question_key {question_key!r} not in columns {dataset.column_names}"
        )
    if answer_key not in dataset.column_names:
        raise ValueError(
            f"answer_key {answer_key!r} not in columns {dataset.column_names}"
        )
    if choices_key is not None and choices_key not in dataset.column_names:
        raise ValueError(
            f"choices_key {choices_key!r} not in columns {dataset.column_names}"
        )
    for key in choice_keys or ():
        if key not in dataset.column_names:
            raise ValueError(
                f"choice key {key!r} not in columns {dataset.column_names}"
            )
    if example_id_key is not None and example_id_key not in dataset.column_names:
        raise ValueError(
            f"example_id_key {example_id_key!r} not in columns {dataset.column_names}"
        )

    def normalize(row: dict[str, Any], row_index: int) -> dict[str, Any]:
        return normalize_hf_row(
            row,
            row_index=row_index,
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
            rng=random.Random(seed + row_index),
        )

    return dataset.map(
        normalize,
        with_indices=True,
        remove_columns=dataset.column_names,
    )


def normalize_hf_row(
    row: dict[str, Any],
    *,
    row_index: int,
    task_type: TaskType,
    question_key: str,
    answer_key: str,
    choices_key: str | None,
    choice_keys: Sequence[str] | None,
    answer_format: AnswerFormat,
    example_id_key: str | None,
    task_name: str | None,
    info_keys: Sequence[str] | None,
    include_raw_info: bool,
    prompt_template: str | None,
    system_prompt: str | None,
    prompts_ref: str | None,
    prompt_role: str,
    prompt_phase: str,
    shuffle_choices: bool,
    rng: random.Random,
) -> dict[str, Any]:
    question = str(row[question_key]).strip()
    raw_answer = row[answer_key]
    choices = extract_choices(row, choices_key=choices_key, choice_keys=choice_keys)
    if shuffle_choices and choices is not None:
        original_choices = choices
        paired = list(enumerate(original_choices))
        rng.shuffle(paired)
        choices = [choice for _, choice in paired]
        old_answer_index = answer_index_before_shuffle(
            raw_answer,
            choices=original_choices,
            answer_format=answer_format,
        )
        if old_answer_index is not None:
            raw_answer = next(
                i for i, (old_i, _) in enumerate(paired) if old_i == old_answer_index
            )
            answer_format = "index"

    answer = normalize_answer(
        raw_answer,
        task_type=task_type,
        answer_format=answer_format,
        choices=choices,
    )
    prompt = render_messages(
        question,
        task_type=task_type,
        choices=choices,
        prompt_template=prompt_template,
        system_prompt=system_prompt,
        prompts_ref=prompts_ref,
        prompt_role=prompt_role,
        prompt_phase=prompt_phase,
    )

    info = build_info(
        row,
        task_type=task_type,
        answer=answer,
        choices=choices,
        info_keys=info_keys,
        include_raw_info=include_raw_info,
    )
    example_id = row.get(example_id_key) if example_id_key else row.get("example_id")
    if example_id is None:
        example_id = row.get("id", row_index)

    return {
        "prompt": prompt,
        "answer": answer,
        "example_id": str(example_id)
        if not isinstance(example_id, int)
        else example_id,
        "task": task_name or task_type,
        "info": json.dumps(info, sort_keys=True, separators=(",", ":")),
    }


def extract_choices(
    row: dict[str, Any],
    *,
    choices_key: str | None,
    choice_keys: Sequence[str] | None,
) -> list[str] | None:
    if choices_key is not None:
        raw = row[choices_key]
        if isinstance(raw, dict):
            if "text" in raw:
                raw = raw["text"]
            else:
                raw = list(raw.values())
        if not isinstance(raw, Sequence) or isinstance(raw, str):
            raise TypeError(
                f"choices_key must point to a sequence, got {type(raw).__name__}"
            )
        return [str(choice).strip() for choice in raw]
    if choice_keys is not None:
        return [str(row[key]).strip() for key in choice_keys]
    return None


def normalize_answer(
    raw_answer: Any,
    *,
    task_type: TaskType,
    answer_format: AnswerFormat,
    choices: list[str] | None,
) -> str:
    if task_type != "mcq":
        return str(raw_answer).strip()
    if choices is None:
        raise ValueError("MCQ tasks require choices_key or choice_keys")
    labels = labels_for_choices(choices)
    if answer_format == "index" or (
        answer_format == "auto" and isinstance(raw_answer, int)
    ):
        index = int(raw_answer)
        if not 0 <= index < len(labels):
            raise ValueError(
                f"answer index {index} outside choices range 0..{len(labels) - 1}"
            )
        return labels[index]
    answer_text = str(raw_answer).strip()
    upper = answer_text.upper()
    if answer_format == "letter" or (answer_format == "auto" and upper in labels):
        if upper not in labels:
            raise ValueError(f"answer letter {answer_text!r} not in labels {labels}")
        return upper
    if answer_format in ("text", "auto"):
        matches = [i for i, choice in enumerate(choices) if choice == answer_text]
        if len(matches) != 1:
            raise ValueError(
                f"answer text {answer_text!r} must match exactly one choice, matched {len(matches)}"
            )
        return labels[matches[0]]
    raise ValueError(f"Unsupported answer_format: {answer_format}")


def answer_index_before_shuffle(
    raw_answer: Any,
    *,
    choices: list[str],
    answer_format: AnswerFormat,
) -> int | None:
    labels = labels_for_choices(choices)
    if answer_format == "index" or (
        answer_format == "auto" and isinstance(raw_answer, int)
    ):
        return int(raw_answer)
    answer_text = str(raw_answer).strip()
    upper = answer_text.upper()
    if answer_format == "letter" or (answer_format == "auto" and upper in labels):
        if upper not in labels:
            raise ValueError(f"answer letter {answer_text!r} not in labels {labels}")
        return labels.index(upper)
    return None


def labels_for_choices(choices: Sequence[str]) -> list[str]:
    if len(choices) > len(LETTERS):
        raise ValueError(
            f"At most {len(LETTERS)} choices are supported, got {len(choices)}"
        )
    return list(LETTERS[: len(choices)])


def render_messages(
    question: str,
    *,
    task_type: TaskType,
    choices: list[str] | None,
    prompt_template: str | None,
    system_prompt: str | None,
    prompts_ref: str | None,
    prompt_role: str,
    prompt_phase: str,
) -> list[dict[str, str]]:
    if prompts_ref is None:
        prompt_text = render_prompt(
            question,
            task_type=task_type,
            choices=choices,
            prompt_template=prompt_template,
        )
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_text})
        return messages

    if system_prompt is not None:
        raise ValueError(
            "system_prompt is ambiguous when prompts_ref is set. Put the "
            "system prompt in the prompt pack instead."
        )

    task_prompt = render_task_prompt(
        question,
        choices=choices,
        prompt_template=prompt_template,
    )
    prompts = resolve_prompts(prompts_ref)
    ctx = build_context(
        task_prompt=task_prompt,
        viewer_id=prompt_role,
        phase=prompt_phase,
        round_index=0,
        num_rounds=1,
    )
    rendered_system = prompts.render_system(prompt_role, ctx).strip()
    rendered_question = prompts.render_question(prompt_role, ctx)
    rendered_instruction = prompts.render_instruction(prompt_role, prompt_phase, ctx)
    user_parts = [
        part.strip()
        for part in (rendered_question, rendered_instruction)
        if part is not None and part.strip()
    ]
    if not user_parts:
        raise ValueError(
            f"Prompt pack {prompts_ref!r} rendered an empty user prompt for "
            f"{prompt_role}.{prompt_phase}"
        )

    messages = []
    if rendered_system:
        messages.append({"role": "system", "content": rendered_system})
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})
    return messages


def render_prompt(
    question: str,
    *,
    task_type: TaskType,
    choices: list[str] | None,
    prompt_template: str | None,
) -> str:
    if prompt_template is not None:
        choices_text = format_choices(choices) if choices is not None else ""
        return prompt_template.format(
            question=question,
            choices=choices_text,
            field_instructions="",
        ).strip()
    if task_type == "mcq":
        if choices is None:
            raise ValueError("MCQ tasks require choices")
        return f"{question}\n\n{format_choices(choices)}\n\n{MCQ_INSTRUCTION}"
    if task_type == "math":
        return f"{question}\n\n{MATH_INSTRUCTION}"
    if task_type == "exact":
        return f"{question}\n\n{EXACT_INSTRUCTION}"
    return f"{question}\n\n{OPEN_ENDED_INSTRUCTION}"


def render_task_prompt(
    question: str,
    *,
    choices: list[str] | None,
    prompt_template: str | None,
) -> str:
    if prompt_template is not None:
        choices_text = format_choices(choices) if choices is not None else ""
        return prompt_template.format(
            question=question,
            choices=choices_text,
            field_instructions="",
        ).strip()
    if choices is None:
        return question
    return f"{question}\n\n{format_choices(choices)}"


def format_choices(choices: Sequence[str] | None) -> str:
    if choices is None:
        return ""
    return "\n".join(
        f"{label}. {choice}"
        for label, choice in zip(labels_for_choices(choices), choices)
    )


def build_info(
    row: dict[str, Any],
    *,
    task_type: TaskType,
    answer: str,
    choices: list[str] | None,
    info_keys: Sequence[str] | None,
    include_raw_info: bool,
) -> dict[str, Any]:
    info: dict[str, Any] = {"task_type": task_type}
    if choices is not None:
        choice_labels = labels_for_choices(choices)
        info["choice_labels"] = choice_labels
        info["choices"] = choices
        if task_type == "mcq":
            info["answer_label"] = answer
            if answer in choice_labels:
                info["answer_text"] = choices[choice_labels.index(answer)]
    for key in info_keys or ():
        if key not in row:
            raise ValueError(f"info key {key!r} missing from row")
        info[key] = row[key]
    if include_raw_info:
        info["raw_row"] = row
    return info


class MultipleChoiceRubric(Rubric):
    def __init__(self) -> None:
        super().__init__(parser=XMLParser(["answer"], answer_field="answer"))
        self.add_reward_func(self.correct_answer)

    async def correct_answer(
        self, parser: XMLParser, completion: Messages, answer: str, info: dict, **kwargs
    ) -> float:
        parsed = parser.parse_answer(completion)
        response = parsed if parsed is not None else completion_text(completion)
        labels = [str(label).upper() for label in info.get("choice_labels", LETTERS)]
        normalized = normalize_choice_response(response, labels)
        return 1.0 if normalized == str(answer).upper() else 0.0


class ExactMatchRubric(Rubric):
    def __init__(self, *, case_sensitive: bool = False) -> None:
        super().__init__(parser=XMLParser(["answer"], answer_field="answer"))
        self.case_sensitive = case_sensitive
        self.add_reward_func(self.correct_answer)

    async def correct_answer(
        self, parser: XMLParser, completion: Messages, answer: str, **kwargs
    ) -> float:
        parsed = parser.parse_answer(completion)
        if parsed is None:
            return 0.0
        lhs = parsed.strip()
        rhs = str(answer).strip()
        if not self.case_sensitive:
            lhs = lhs.casefold()
            rhs = rhs.casefold()
        return 1.0 if lhs == rhs else 0.0


def completion_text(completion: Messages) -> str:
    parts: list[str] = []
    for message in completion:
        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = getattr(message, "content", "")
        if isinstance(content, str) and content:
            parts.append(content)
    return "\n".join(parts)


def extract_final_answer(response: str) -> str | None:
    answer_tags = ANSWER_TAG_RE.findall(response)
    if answer_tags:
        return _strip_answer_markup(answer_tags[-1])

    for label_re in (FINAL_ANSWER_LABEL_RE, ANSWER_LABEL_RE):
        label_match = label_re.search(response)
        if label_match is not None:
            return _strip_answer_markup(label_match.group(1))

    return None


def _strip_answer_markup(answer: str) -> str:
    stripped = answer.strip().rstrip(".").strip()
    if stripped.startswith("**") and stripped.endswith("**") and len(stripped) >= 4:
        stripped = stripped[2:-2].strip()
    return stripped


def _make_open_ended_judge_rubric(
    *,
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
) -> JudgeRubric:
    client = judge_client
    if client is None and judge_base_url is not None:
        client = OpenAIChatCompletionsClient(
            ClientConfig(
                api_base_url=judge_base_url,
                api_key_var=judge_api_key_var,
            )
        )

    overrides_prompt = judge_system_prompt is not None or judge_prompt is not None
    if overrides_prompt and judge_positive_label is None:
        raise ValueError(
            "judge_positive_label is required when overriding judge prompt text"
        )

    judge_kwargs: dict[str, Any] = {}
    template = None
    if judge_prompt_pack is not None:
        prompts = resolve_judge_prompts(judge_prompt_pack)
        template = prompts.get(judge_prompt_kind)
        if template is None:
            raise ValueError(
                f"Prompt pack {judge_prompt_pack!r} has no {judge_prompt_kind!r} judge"
            )
        judge_kwargs["judge_system_prompt"] = template.system
        judge_kwargs["judge_prompt"] = template.user
        positive_label = template.positive
        negative_label = template.negative
    else:
        positive_label = "YES" if judge_prompt is None else "CORRECT"
        negative_label = "NO" if judge_prompt is None else "INCORRECT"

    if judge_positive_label is not None:
        normalized_positive = normalize_verdict_token(judge_positive_label)
        if normalized_positive is None:
            raise ValueError("judge_positive_label must be a non-empty verdict token")
        positive_label = normalized_positive
    if judge_negative_label is not None:
        normalized_negative = normalize_verdict_token(judge_negative_label)
        if normalized_negative is None:
            raise ValueError("judge_negative_label must be a non-empty verdict token")
        negative_label = normalized_negative

    if judge_system_prompt is not None:
        judge_kwargs["judge_system_prompt"] = judge_system_prompt
    if judge_prompt is not None:
        judge_kwargs["judge_prompt"] = judge_prompt

    if template is not None:
        judge_model = judge_model or template.model
        judge_sampling_args = judge_sampling_args or template.sampling_args
        judge_rubric_family = judge_rubric_family or template.rubric_family
        judge_variant_id = judge_variant_id or template.variant_id
        judge_reward_mode = judge_reward_mode or template.reward_mode
        judge_panel_threshold = (
            judge_panel_threshold
            if judge_panel_threshold is not None
            else template.threshold
        )
        judge_calibration_mode = judge_calibration_mode or template.calibration_mode
        judge_correctness_prior = (
            judge_correctness_prior
            if judge_correctness_prior is not None
            else template.correctness_prior
        )
        judge_sensitivity = (
            judge_sensitivity if judge_sensitivity is not None else template.sensitivity
        )
        judge_false_positive_rate = (
            judge_false_positive_rate
            if judge_false_positive_rate is not None
            else template.false_positive_rate
        )
        judge_repeated_call_correlation = (
            judge_repeated_call_correlation
            if judge_repeated_call_correlation is not None
            else template.repeated_call_correlation
        )

    rubric = JudgeRubric(
        parser=vf.Parser(extract_fn=extract_final_answer),
        judge_client=client,
        judge_model=judge_model or "gpt-5.4-nano",
        judge_sampling_args=judge_sampling_args
        or {"temperature": 0.0, "max_completion_tokens": 64},
        judge_cache_enabled=judge_cache_enabled,
        judge_cache_size=judge_cache_size,
        judge_max_retries=judge_max_retries,
        judge_retry_delay_s=judge_retry_delay_s,
        judge_persistent_cache_path=judge_persistent_cache_path,
        judge_persistent_cache_min_samples=judge_persistent_cache_min_samples,
        judge_rubric_family=judge_rubric_family or "default",
        judge_variant_id=judge_variant_id or "default",
        judge_positive_label=positive_label,
        judge_negative_label=negative_label,
        judge_reward_mode=judge_reward_mode or "hard",
        judge_panel_threshold=0.5
        if judge_panel_threshold is None
        else judge_panel_threshold,
        judge_panel_prior_alpha=1.0
        if judge_panel_prior_alpha is None
        else judge_panel_prior_alpha,
        judge_panel_prior_beta=1.0
        if judge_panel_prior_beta is None
        else judge_panel_prior_beta,
        judge_repeated_call_correlation=0.0
        if judge_repeated_call_correlation is None
        else judge_repeated_call_correlation,
        judge_calibration_mode=judge_calibration_mode or "vote_fraction",
        judge_correctness_prior=0.5
        if judge_correctness_prior is None
        else judge_correctness_prior,
        judge_sensitivity=judge_sensitivity,
        judge_false_positive_rate=judge_false_positive_rate,
        **judge_kwargs,
    )

    async def correct_answer(
        judge,
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State,
        **kwargs,
    ) -> float:
        if rubric.parser.parse_answer(completion) is None:
            return 0.0
        raw_grade = await judge(prompt, completion, answer, state)
        decision = state.get("judge_decision_last")
        if isinstance(decision, dict) and isinstance(
            decision.get("reward"), int | float
        ):
            return float(decision["reward"])
        return 1.0 if normalize_verdict_token(raw_grade) == positive_label else 0.0

    rubric.add_reward_func(correct_answer)
    return rubric


def make_rubric(
    *,
    task_type: TaskType,
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
) -> Rubric:
    if task_type == "mcq":
        return MultipleChoiceRubric()
    if task_type == "math":
        return vf.MathRubric()
    if task_type == "exact":
        return ExactMatchRubric()
    if task_type == "open_ended":
        return _make_open_ended_judge_rubric(
            judge_client=judge_client,
            judge_model=judge_model,
            judge_base_url=judge_base_url,
            judge_api_key_var=judge_api_key_var,
            judge_sampling_args=judge_sampling_args,
            judge_prompt_pack=judge_prompt_pack,
            judge_prompt_kind=judge_prompt_kind,
            judge_system_prompt=judge_system_prompt,
            judge_prompt=judge_prompt,
            judge_positive_label=judge_positive_label,
            judge_negative_label=judge_negative_label,
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
    raise ValueError(f"Unsupported task_type: {task_type}")


def normalize_choice_response(response: str, labels: Sequence[str]) -> str | None:
    if has_ambiguous_choice_pair(response, labels):
        return None
    if set(labels).issubset(set("ABCDE")):
        extracted = normalize_mcq(response)
        if extracted in labels:
            return extracted
        return None

    cleaned = response.strip().upper()
    if cleaned in labels:
        return cleaned
    for token in cleaned.replace(")", " ").replace(".", " ").replace(":", " ").split():
        if token in labels:
            return token
    return None


def has_ambiguous_choice_pair(response: str, labels: Sequence[str]) -> bool:
    escaped = [re.escape(str(label)) for label in labels]
    if not escaped:
        return False
    choice = "|".join(escaped)
    return (
        re.search(
            rf"(?i)(?<!\w)(?:{choice})(?!\w)\s*(?:/|\bor\b|\band\b)\s*(?<!\w)(?:{choice})(?!\w)",
            response,
        )
        is not None
    )


def coerce_dataset(value: Any) -> Dataset:
    if not isinstance(value, Dataset):
        raise TypeError(f"Expected datasets.Dataset, got {type(value).__name__}")
    return cast(Dataset, value)
