from __future__ import annotations

import asyncio
import json
import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset

import verifiers as vf
from environments.hf_debate.hf_debate import load_environment as load_hf_debate
from environments.hf_singleturn.hf_singleturn import (
    load_environment as load_hf_singleturn,
)
from verifiers.utils.hf_tasks import (
    MultipleChoiceRubric,
    extract_final_answer,
    make_rubric,
    load_hf_dataset,
    normalize_choice_response,
    normalize_hf_dataset,
)
from verifiers.utils.judge_prompts import normalize_verdict_token, resolve_judge_prompts


def test_normalize_gpqa_style_mcq_row() -> None:
    raw = Dataset.from_list(
        [
            {
                "Question": "Which option is correct?",
                "Correct Answer": "alpha",
                "Incorrect Answer 1": "beta",
                "Incorrect Answer 2": "gamma",
                "Incorrect Answer 3": "delta",
                "Record ID": "rec-1",
            }
        ]
    )

    dataset = normalize_hf_dataset(
        raw,
        task_type="mcq",
        question_key="Question",
        answer_key="Correct Answer",
        choice_keys=[
            "Correct Answer",
            "Incorrect Answer 1",
            "Incorrect Answer 2",
            "Incorrect Answer 3",
        ],
        answer_format="text",
        example_id_key="Record ID",
        task_name="gpqa",
    )

    row = dataset[0]
    assert row["example_id"] == "rec-1"
    assert row["answer"] == "A"
    assert row["task"] == "gpqa"
    info = json.loads(row["info"])
    assert info["answer_label"] == "A"
    assert info["answer_text"] == "alpha"
    assert "A. alpha" in row["prompt"][0]["content"]
    assert "D. delta" in row["prompt"][0]["content"]


def test_normalize_mcq_shuffle_preserves_letter_answer() -> None:
    raw = Dataset.from_list(
        [
            {
                "question": "Pick the right option.",
                "choices": ["wrong", "right", "also wrong"],
                "answer": "B",
            }
        ]
    )

    dataset = normalize_hf_dataset(
        raw,
        task_type="mcq",
        choices_key="choices",
        answer_format="letter",
        shuffle_choices=True,
        seed=1,
    )

    info = json.loads(dataset[0]["info"])
    assert info["answer_text"] == "right"


def test_normalize_hf_dataset_accepts_streaming_iterable_with_num_examples() -> None:
    raw = Dataset.from_list(
        [
            {"question": "First?", "answer": "A", "id": "row-1"},
            {"question": "Second?", "answer": "B", "id": "row-2"},
            {"question": "Third?", "answer": "C", "id": "row-3"},
        ]
    ).to_iterable_dataset()

    dataset = normalize_hf_dataset(raw, task_type="exact", num_examples=2)

    assert len(dataset) == 2
    assert dataset[0]["example_id"] == "row-1"
    assert dataset[1]["example_id"] == "row-2"


def test_normalize_hf_dataset_requires_num_examples_for_streaming() -> None:
    raw = Dataset.from_list(
        [{"question": "First?", "answer": "A"}]
    ).to_iterable_dataset()

    with pytest.raises(ValueError, match="num_examples must be set"):
        normalize_hf_dataset(raw, task_type="exact")


def test_load_hf_dataset_streaming_parquet_reads_requested_columns(tmp_path) -> None:
    path = tmp_path / "rows.parquet"
    pq.write_table(
        pa.table(
            {
                "question": ["First?", "Second?", "Third?"],
                "answer": ["A", "B", "C"],
                "drop_me": ["x", "y", "z"],
            }
        ),
        path,
    )

    dataset = load_hf_dataset(
        "parquet",
        data_files=str(path),
        streaming=True,
        columns=["question", "answer"],
        streaming_limit=2,
    )

    assert isinstance(dataset, Dataset)
    assert dataset.column_names == ["question", "answer"]
    assert len(dataset) == 2
    assert dataset[1]["answer"] == "B"


def test_load_hf_dataset_streaming_parquet_supports_bounded_shuffle(tmp_path) -> None:
    path = tmp_path / "rows.parquet"
    rows = [
        {"question": "First?", "answer": "A"},
        {"question": "Second?", "answer": "B"},
        {"question": "Third?", "answer": "C"},
    ]
    pq.write_table(
        pa.table(
            {
                "question": [row["question"] for row in rows],
                "answer": [row["answer"] for row in rows],
            }
        ),
        path,
    )

    dataset = load_hf_dataset(
        "parquet",
        data_files=str(path),
        streaming=True,
        columns=["question", "answer"],
        streaming_limit=2,
        streaming_shuffle_buffer_size=3,
        streaming_seed=7,
    )

    random.Random(7).shuffle(rows)
    assert dataset.to_list() == rows[:2]


def test_mcq_rubric_scores_xml_answer_with_dataset_labels() -> None:
    rubric = MultipleChoiceRubric()
    state = vf.State(
        prompt=[],
        completion=[{"role": "assistant", "content": "<answer>B</answer>"}],
        answer="B",
        task="toy",
        info={"choice_labels": ["A", "B", "C", "D"]},
        timing={"start_time": 0.0, "scoring_ms": 0.0, "total_ms": 0.0},
    )

    asyncio.run(rubric.score_rollout(state))

    assert state["reward"] == 1.0


def test_mcq_rubric_falls_back_to_visible_answer_frame() -> None:
    rubric = MultipleChoiceRubric()
    state = vf.State(
        prompt=[],
        completion=[{"role": "assistant", "content": "The correct answer is B."}],
        answer="B",
        task="toy",
        info={"choice_labels": ["A", "B", "C", "D"]},
        timing={"start_time": 0.0, "scoring_ms": 0.0, "total_ms": 0.0},
    )

    asyncio.run(rubric.score_rollout(state))

    assert state["reward"] == 1.0


def test_mcq_response_normalizer_rejects_ambiguous_answers() -> None:
    assert normalize_choice_response("A or B", ["A", "B", "C", "D"]) is None
    assert normalize_choice_response("The answer is B.", ["A", "B", "C", "D"]) == "B"


def test_open_ended_judge_rubric_uses_nano_and_one_word_verdict(mock_client) -> None:
    mock_client.set_default_response("CORRECT")
    rubric = make_rubric(task_type="open_ended", judge_client=mock_client)
    state = vf.State(
        prompt=[{"role": "user", "content": "What principle applies?"}],
        completion=[
            {
                "role": "assistant",
                "content": "<answer>The energy-time uncertainty principle applies.</answer>",
            }
        ],
        answer="Energy-time uncertainty principle.",
        task="open",
        timing={"start_time": 0.0, "scoring_ms": 0.0, "total_ms": 0.0},
    )

    asyncio.run(rubric.score_rollout(state))

    assert state["reward"] == 1.0
    assert mock_client.last_call_kwargs["model"] == "gpt-5.4-nano"
    routed_prompt = mock_client.last_call_kwargs["prompt"]
    assert [message.role for message in routed_prompt] == ["system", "user"]
    assert routed_prompt[1].content == (
        "Target: Energy-time uncertainty principle.\n"
        "Response: The energy-time uncertainty principle applies\n"
    )


def test_open_ended_judge_rubric_grades_extracted_final_answer(mock_client) -> None:
    mock_client.set_default_response("CORRECT")
    rubric = make_rubric(task_type="open_ended", judge_client=mock_client)
    state = vf.State(
        prompt=[{"role": "user", "content": "How many compounds are active?"}],
        completion=[
            {
                "role": "assistant",
                "content": (
                    "Only two are active -- correction: the count is three.\n"
                    "Final answer: **Three**."
                ),
            }
        ],
        answer="3",
        task="open",
        timing={"start_time": 0.0, "scoring_ms": 0.0, "total_ms": 0.0},
    )

    asyncio.run(rubric.score_rollout(state))

    assert state["reward"] == 1.0
    assert mock_client.last_call_kwargs["prompt"][1].content == (
        "Target: 3\nResponse: Three\n"
    )


def test_extract_final_answer_handles_inline_final_answer_label() -> None:
    response = (
        "Some noisy reasoning says maybe two, then corrects itself. "
        "Final answer: **Three**."
    )

    assert extract_final_answer(response) == "Three"


def test_hf_prompt_pack_renders_question_system_and_field_instructions(
    tmp_path,
) -> None:
    raw = Dataset.from_list([{"question": "What is the threshold?", "answer": "x"}])
    pack = tmp_path / "singleturn_prompt.yaml"
    pack.write_text(
        """
version: 2
system:
  debater_a: "Solve carefully."
  debater_b: "Solve carefully."
  judge: "Judge carefully."
question:
  debater_a: "{{ task_prompt }}"
  debater_b: "{{ task_prompt }}"
  judge: "{{ task_prompt }}"
user:
  debater_a:
    propose: "Work through the problem before giving the answer."
  debater_b:
    propose: "Work through the problem before giving the answer."
  judge:
    final: "Choose the better supported answer."
fields:
  debater_a:
    propose:
      answer:
        type: str
        description: shortest standalone final answer
  debater_b:
    propose:
      answer:
        type: str
        description: shortest standalone final answer
  judge:
    final:
      decision:
        type: str
        scoring:
          mode: enum
          values: [debater_a, debater_b, tie]
_grader:
  system: "Grade: CORRECT or INCORRECT. One word."
  user: "Target: {answer}\\nResponse: {response}\\n"
  positive: CORRECT
  negative: INCORRECT
_matcher:
  system: "Compare two answers. One word: SAME or DIFFERENT."
  user: "Answer A: \\"{answer}\\"\\nAnswer B: \\"{response}\\"\\n"
  positive: SAME
  negative: DIFFERENT
""".strip()
    )

    dataset = normalize_hf_dataset(
        raw,
        task_type="open_ended",
        prompt_template="Solve fully.\n\n{question}",
        prompts_ref=str(pack),
    )

    messages = dataset[0]["prompt"]
    assert messages[0] == {"role": "system", "content": "Solve carefully."}
    prompt = messages[1]["content"]
    assert "Solve fully.\n\nWhat is the threshold?" in prompt
    assert "Work through the problem before giving the answer." in prompt
    assert "After your reasoning" in prompt
    assert "<answer>shortest standalone final answer</answer>" in prompt


def test_hf_singleturn_accepts_prompt_pack(
    tmp_path,
) -> None:
    raw = Dataset.from_list([{"question": "What is the threshold?", "answer": "x"}])
    pack = tmp_path / "singleturn_prompt.yaml"
    pack.write_text(
        """
version: 2
system:
  debater_a: "Single-turn solver."
  debater_b: "Single-turn solver."
question:
  debater_a: "{{ task_prompt }}"
  debater_b: "{{ task_prompt }}"
user:
  debater_a:
    propose: "Reason first."
  debater_b:
    propose: "Reason first."
fields:
  debater_a:
    propose:
      answer:
        type: str
        description: final answer
_grader:
  model: gpt-test-mini
  sampling_args:
    reasoning_effort: medium
    max_completion_tokens: 1024
  cache:
    rubric_family: test_family
    variant_id: test_variant_v1
  reward:
    mode: soft
    threshold: 0.5
  calibration:
    mode: confusion_matrix
    correctness_prior: 0.25
    sensitivity: 0.96
    false_positive_rate: 0.01
    repeated_call_correlation: 0.95
  system: "PACK GRADER"
  user: "Target: {answer}\\nResponse: {response}\\n"
  positive: CORRECT
  negative: INCORRECT
_matcher:
  system: "Compare two answers. One word: SAME or DIFFERENT."
  user: "Answer A: \\"{answer}\\"\\nAnswer B: \\"{response}\\"\\n"
  positive: SAME
  negative: DIFFERENT
""".strip()
    )

    env = load_hf_singleturn(
        dataset=raw,
        task_type="open_ended",
        prompts_ref=str(pack),
    )

    prompt = env.get_dataset()[0]["prompt"][1]["content"]
    assert "Reason first." in prompt
    assert prompt.count("After your reasoning") == 1
    assert "<answer>final answer</answer>" in prompt
    rubric = env.rubric.rubrics[0]
    assert rubric.judge_system_prompt == "PACK GRADER"
    assert rubric.judge_model == "gpt-test-mini"
    assert rubric.judge_sampling_args == {
        "reasoning_effort": "medium",
        "max_completion_tokens": 1024,
    }
    assert rubric.judge_rubric_family == "test_family"
    assert rubric.judge_variant_id == "test_variant_v1"
    assert rubric.judge_reward_mode == "soft"
    assert rubric.judge_panel.calibration_mode == "confusion_matrix"
    assert rubric.judge_panel.correctness_prior == 0.25
    assert rubric.judge_panel.judge_sensitivity == 0.96
    assert rubric.judge_panel.judge_false_positive_rate == 0.01
    assert rubric.judge_panel.repeated_call_correlation == 0.95


def test_hf_singleturn_threads_persistent_judge_cache_args(tmp_path) -> None:
    raw = Dataset.from_list([{"question": "What is the threshold?", "answer": "x"}])
    cache_path = tmp_path / "judge.sqlite3"

    env = load_hf_singleturn(
        dataset=raw,
        task_type="open_ended",
        judge_persistent_cache_path=str(cache_path),
        judge_persistent_cache_min_samples=3,
        judge_rubric_family="strict_equivalence",
        judge_variant_id="numeric_guard",
        judge_reward_mode="soft",
        judge_repeated_call_correlation=0.5,
        judge_calibration_mode="confusion_matrix",
        judge_correctness_prior=0.25,
        judge_sensitivity=0.96,
        judge_false_positive_rate=0.01,
    )

    rubric = env.rubric.rubrics[0]
    assert rubric.judge_persistent_cache_path == str(cache_path)
    assert rubric.judge_persistent_cache_min_samples == 3
    assert rubric.judge_rubric_family == "strict_equivalence"
    assert rubric.judge_variant_id == "numeric_guard"
    assert rubric.judge_reward_mode == "soft"
    assert rubric.judge_panel.repeated_call_correlation == 0.5
    assert rubric.judge_panel.calibration_mode == "confusion_matrix"
    assert rubric.judge_panel.correctness_prior == 0.25
    assert rubric.judge_panel.judge_sensitivity == 0.96
    assert rubric.judge_panel.judge_false_positive_rate == 0.01


def test_open_ended_judge_rubric_prefers_xml_answer_tags(mock_client) -> None:
    mock_client.set_default_response("CORRECT")
    rubric = make_rubric(task_type="open_ended", judge_client=mock_client)
    state = vf.State(
        prompt=[{"role": "user", "content": "How many compounds are active?"}],
        completion=[
            {
                "role": "assistant",
                "content": "Reasoning text with noise. <answer>3</answer>",
            }
        ],
        answer="3",
        task="open",
        timing={"start_time": 0.0, "scoring_ms": 0.0, "total_ms": 0.0},
    )

    asyncio.run(rubric.score_rollout(state))

    assert state["reward"] == 1.0
    assert mock_client.last_call_kwargs["prompt"][1].content == (
        "Target: 3\nResponse: 3\n"
    )


def test_extract_final_answer_fails_closed_without_final_answer() -> None:
    assert extract_final_answer("Energy-time uncertainty principle.") is None


def test_open_ended_judge_rubric_does_not_substring_match_incorrect(
    mock_client,
) -> None:
    mock_client.set_default_response("INCORRECT")
    rubric = make_rubric(task_type="open_ended", judge_client=mock_client)
    state = vf.State(
        prompt=[{"role": "user", "content": "What principle applies?"}],
        completion=[
            {"role": "assistant", "content": "<answer>Newton's laws.</answer>"}
        ],
        answer="Energy-time uncertainty principle.",
        task="open",
        timing={"start_time": 0.0, "scoring_ms": 0.0, "total_ms": 0.0},
    )

    asyncio.run(rubric.score_rollout(state))

    assert state["reward"] == 0.0


def test_open_ended_custom_judge_prompt_requires_positive_label(mock_client) -> None:
    with pytest.raises(ValueError, match="judge_positive_label"):
        make_rubric(
            task_type="open_ended",
            judge_client=mock_client,
            judge_prompt="Target: {answer}\nResponse: {response}",
        )


def test_open_ended_custom_judge_prompt_uses_explicit_positive_label(
    mock_client,
) -> None:
    mock_client.set_default_response("SAME")
    rubric = make_rubric(
        task_type="open_ended",
        judge_client=mock_client,
        judge_prompt_pack=None,
        judge_prompt="Target: {answer}\nResponse: {response}",
        judge_positive_label="SAME",
    )
    state = vf.State(
        prompt=[{"role": "user", "content": "What principle applies?"}],
        completion=[
            {
                "role": "assistant",
                "content": "<answer>Energy-time uncertainty.</answer>",
            }
        ],
        answer="Energy-time uncertainty.",
        task="open",
        timing={"start_time": 0.0, "scoring_ms": 0.0, "total_ms": 0.0},
    )

    asyncio.run(rubric.score_rollout(state))

    assert state["reward"] == 1.0


def test_open_ended_judge_prompt_pack_loads_grader_and_matcher() -> None:
    prompts = resolve_judge_prompts("default")
    grader = prompts["grader"]
    matcher = prompts["matcher"]

    assert grader.positive == "CORRECT"
    assert grader.negative == "INCORRECT"
    assert "Target: {answer}" in grader.user
    assert matcher.positive == "SAME"
    assert matcher.negative == "DIFFERENT"
    assert 'Answer A: "{answer}"' in matcher.user


def test_judge_label_normalizer_is_exact_first_token() -> None:
    assert normalize_verdict_token("CORRECT") == "CORRECT"
    assert normalize_verdict_token("INCORRECT") == "INCORRECT"
    assert normalize_verdict_token("INCORRECT because 0.32 != 0.33") == "INCORRECT"


def test_normalize_gpqa_open_ended_style_row() -> None:
    raw = Dataset.from_list(
        [
            {
                "idx": 1,
                "question": "Derive the condition.",
                "answer": "Use the energy-time uncertainty principle.",
                "record_id": "rec-open",
                "domain": "Physics",
            }
        ]
    )

    dataset = normalize_hf_dataset(
        raw,
        task_type="open_ended",
        question_key="question",
        answer_key="answer",
        example_id_key="record_id",
        task_name="gpqa_open_ended",
        info_keys=["domain"],
    )

    row = dataset[0]
    assert row["example_id"] == "rec-open"
    assert row["answer"] == "Use the energy-time uncertainty principle."
    assert row["task"] == "gpqa_open_ended"
    assert row["info"] == '{"domain":"Physics","task_type":"open_ended"}'


def test_hf_singleturn_loads_without_dataset_args() -> None:
    env = load_hf_singleturn()

    row = env.get_dataset()[0]
    assert row["answer"] == "4"


def test_hf_debate_loads_without_dataset_args() -> None:
    env = load_hf_debate()

    row = env.get_dataset()[0]
    assert row["answer"] == "B"
    assert env.members == ["debater_a", "debater_b", "judge"]


def test_hf_singleturn_accepts_in_memory_mcq_dataset() -> None:
    raw = Dataset.from_list(
        [
            {
                "question": "2+2?",
                "choices": ["3", "4", "5", "6"],
                "answer": 1,
            }
        ]
    )

    env = load_hf_singleturn(
        dataset=raw,
        task_type="mcq",
        choices_key="choices",
        answer_format="index",
    )

    row = env.get_dataset()[0]
    assert row["answer"] == "B"
    assert "<answer>" in row["prompt"][0]["content"]


def test_hf_singleturn_accepts_in_memory_math_dataset() -> None:
    raw = Dataset.from_list([{"problem": "Compute 2+2.", "answer": "4"}])

    env = load_hf_singleturn(
        dataset=raw,
        task_type="math",
        question_key="problem",
        answer_key="answer",
    )

    row = env.get_dataset()[0]
    assert row["answer"] == "4"
    assert "\\boxed{}" in row["prompt"][0]["content"]


def test_hf_debate_accepts_in_memory_mcq_dataset() -> None:
    raw = Dataset.from_list(
        [
            {
                "question": "2+2?",
                "choices": ["3", "4", "5", "6"],
                "answer": 1,
            }
        ]
    )

    env = load_hf_debate(
        dataset=raw,
        task_type="mcq",
        choices_key="choices",
        answer_format="index",
    )

    row = env.get_dataset()[0]
    assert row["answer"] == "B"
    assert env.members == ["debater_a", "debater_b", "judge"]


def test_hf_debate_rejects_mcq_labels_not_supported_by_prompt_pack() -> None:
    raw = Dataset.from_list(
        [
            {
                "question": "Pick E.",
                "choices": ["A0", "B0", "C0", "D0", "E0"],
                "answer": 4,
            }
        ]
    )

    env = load_hf_debate(
        dataset=raw,
        task_type="mcq",
        choices_key="choices",
        answer_format="index",
    )

    with pytest.raises(ValueError, match="accepts MCQ labels"):
        env.get_dataset()


def test_hf_debate_rejects_system_prompt() -> None:
    with pytest.raises(ValueError, match="system_prompt"):
        load_hf_debate(system_prompt="unused")


def test_hf_debate_uses_nano_for_grader_and_matcher(mock_client) -> None:
    raw = Dataset.from_list(
        [{"question": "2+2?", "choices": ["3", "4", "5", "6"], "answer": 1}]
    )

    env = load_hf_debate(
        dataset=raw,
        task_type="mcq",
        choices_key="choices",
        answer_format="index",
        judge_client=mock_client,
    )

    assert env.rubric.grader is not None
    assert env.rubric.matcher is not None
    assert env.rubric.grader.judge_model == "gpt-5.4-nano"
    assert env.rubric.matcher.judge_model == "gpt-5.4-nano"
    assert env.rubric.grader.judge_system_prompt is not None
    assert env.rubric.grader.judge_system_prompt.startswith(
        "Grade: CORRECT or INCORRECT."
    )
    assert env.rubric.matcher.judge_system_prompt is not None
    assert env.rubric.matcher.judge_system_prompt.startswith("Compare two answers")
