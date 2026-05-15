"""Seam tests for the real JudgeRubric class.

These tests live in the fork's own test suite (not the orchestrator's
test_debate_env.py) so they import and exercise the production
JudgeRubric class — the orchestrator test file stubs
``verifiers.rubrics.judge_rubric`` with a minimal ``_StubJudgeRubric``
mirror to avoid pulling httpx/openai transitively, which means
composition-level tests there can only assert on the stub contract.
This file closes that gap by loading the real module, which is
importable in the fork's own venv.

Contract under test:
  - ``judge()`` renders the ``judge_prompt`` format string with
    ``{question}``, ``{answer}``, ``{response}`` substitutions drawn
    from the caller-supplied arguments, and routes the rendered text
    through ``vf.Client.get_response`` with the configured ``model``
    and normalized ``sampling_args``. ``state`` is threaded through as
    a kwarg.
  - ``state["judge_response"]`` acts as a per-rendered-prompt verdict
    cache so a single rollout's repeated grader calls coalesce.
"""

from __future__ import annotations

import asyncio

import pytest

from verifiers.rubrics.judge_rubric import JudgeRubric
from verifiers.rubrics.judge_evidence_cache import (
    JudgeEvidenceSample,
    JudgePanelPolicy,
)
from verifiers.types import State, UserMessage


def test_judge_panel_policy_returns_soft_posterior_with_effective_sample_discount():
    policy = JudgePanelPolicy(
        positive_label="CORRECT",
        negative_label="INCORRECT",
        prior_alpha=1.0,
        prior_beta=1.0,
        repeated_call_correlation=0.5,
        reward_mode="soft",
    )

    decision = policy.decide(
        [
            JudgeEvidenceSample(raw_response="CORRECT", verdict="CORRECT"),
            JudgeEvidenceSample(raw_response="CORRECT again", verdict="CORRECT"),
            JudgeEvidenceSample(raw_response="INCORRECT", verdict="INCORRECT"),
        ]
    )

    assert decision.p_correct == pytest.approx(0.5714285714)
    assert decision.reward == pytest.approx(decision.p_correct)
    assert decision.hard_response == "CORRECT"
    assert decision.n_valid == 3
    assert decision.n_effective == pytest.approx(1.5)
    assert decision.raw_response == "CORRECT again"


def test_judge_panel_policy_can_use_calibrated_confusion_matrix():
    policy = JudgePanelPolicy(
        positive_label="CORRECT",
        negative_label="INCORRECT",
        calibration_mode="confusion_matrix",
        correctness_prior=0.25,
        judge_sensitivity=0.9,
        judge_false_positive_rate=0.1,
        repeated_call_correlation=1.0,
        reward_mode="soft",
    )

    positive = policy.decide(
        [JudgeEvidenceSample(raw_response="CORRECT", verdict="CORRECT")]
    )
    negative = policy.decide(
        [JudgeEvidenceSample(raw_response="INCORRECT", verdict="INCORRECT")]
    )
    duplicate_positives = policy.decide(
        [
            JudgeEvidenceSample(raw_response="CORRECT", verdict="CORRECT")
            for _ in range(5)
        ]
    )

    assert positive.p_correct == pytest.approx(0.75)
    assert negative.p_correct == pytest.approx(0.0357142857)
    assert duplicate_positives.p_correct == pytest.approx(positive.p_correct)
    assert duplicate_positives.n_effective == pytest.approx(1.0)


def test_judge_panel_policy_accepts_punctuated_verdict_tokens():
    policy = JudgePanelPolicy(
        positive_label="CORRECT",
        negative_label="INCORRECT",
    )

    positive = policy.decide(
        [JudgeEvidenceSample(raw_response="CORRECT.", verdict="CORRECT.")]
    )
    negative = policy.decide(
        [JudgeEvidenceSample(raw_response="INCORRECT.", verdict="INCORRECT.")]
    )

    assert positive.hard_response == "CORRECT"
    assert positive.n_valid == 1
    assert negative.hard_response == "INCORRECT"
    assert negative.n_valid == 1


@pytest.mark.asyncio
async def test_judge_rubric_routes_rendered_prompt_model_and_args(mock_client):
    """The composition seam. Asserts (a) the rendered prompt actually
    carries the substituted {question}/{answer}/{response} fragments, not
    the raw format string, (b) the configured ``judge_model`` reaches the
    client, (c) ``judge_sampling_args`` are forwarded with the
    ``max_tokens -> max_completion_tokens`` normalization, (d) ``state``
    is threaded through as a kwarg.

    A bug in the template renderer, the sampling-args normalization, or
    the state passthrough would slip past a verdict-only assertion. This
    test catches all four at the seam.
    """
    mock_client.set_default_response("CORRECT")
    rubric = JudgeRubric(
        judge_client=mock_client,
        judge_model="test-model",
        judge_prompt=(
            "Q: {question}\nGT: {answer}\nResp: {response}\nReply CORRECT or INCORRECT."
        ),
        # Use the legacy max_tokens form so the normalization path runs.
        judge_sampling_args={"temperature": 0.0, "max_tokens": 64},
    )
    state: State = State(_seed=True)
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    completion = [{"role": "assistant", "content": "four"}]

    verdict = await rubric.judge(
        prompt=prompt, completion=completion, answer="4", state=state
    )

    assert verdict == "CORRECT"
    assert mock_client.call_count == 1

    # --- (a) the rendered prompt carries the substituted fragments ---
    routed = mock_client.last_call_kwargs
    routed_prompt = routed["prompt"]
    assert isinstance(routed_prompt, list) and len(routed_prompt) == 1
    rendered = routed_prompt[0]
    # Real client path wraps in UserMessage, but MockClient receives it
    # verbatim. Pull content via dict OR attr to stay stable across
    # Pydantic/dict forms.
    rendered_content = (
        rendered.get("content")
        if isinstance(rendered, dict)
        else getattr(rendered, "content", "")
    )
    assert "Q: What is 2+2?" in rendered_content
    assert "GT: 4" in rendered_content
    assert "Resp: four" in rendered_content
    # Confirm renderer actually substituted rather than leaving braces.
    assert "{question}" not in rendered_content
    assert "{answer}" not in rendered_content
    assert "{response}" not in rendered_content

    # --- (b) model passthrough ---
    assert routed["model"] == "test-model"

    # --- (c) sampling args normalized: max_tokens -> max_completion_tokens,
    #         None-valued keys stripped ---
    sampling_args = routed["sampling_args"]
    assert sampling_args == {"temperature": 0.0, "max_completion_tokens": 64}
    assert "max_tokens" not in sampling_args

    # --- (d) state threaded through as a kwarg ---
    assert routed.get("state") is state


@pytest.mark.asyncio
async def test_judge_rubric_strips_none_sampling_arg_values(mock_client):
    """None-valued ``max_completion_tokens`` / ``max_tokens`` must be
    dropped before routing — passing ``None`` through triggers a
    ``pydantic_core.ValidationError`` on real Chat Completions backends."""
    mock_client.set_default_response("YES")
    rubric = JudgeRubric(
        judge_client=mock_client,
        judge_model="test-model",
        judge_prompt="Q: {question} A: {answer} R: {response}",
        judge_sampling_args={
            "temperature": 0.0,
            "max_completion_tokens": None,
            "max_tokens": None,
            "top_p": None,
        },
    )
    state: State = State(_seed=True)
    await rubric.judge(
        prompt=[{"role": "user", "content": "q"}],
        completion=[{"role": "assistant", "content": "a"}],
        answer="a",
        state=state,
    )
    sampling_args = mock_client.last_call_kwargs["sampling_args"]
    assert sampling_args == {"temperature": 0.0}
    assert all(v is not None for v in sampling_args.values())


@pytest.mark.asyncio
async def test_judge_rubric_extracts_question_from_typed_message(mock_client):
    mock_client.set_default_response("CORRECT")
    rubric = JudgeRubric(
        judge_client=mock_client,
        judge_model="test-model",
        judge_prompt="Q: {question}\nA: {answer}\nR: {response}",
    )

    await rubric.judge(
        prompt=[UserMessage(content="What is 2+2?")],
        completion=[{"role": "assistant", "content": "four"}],
        answer="4",
        state=State(_seed=True),
    )

    rendered = mock_client.last_call_kwargs["prompt"][0]
    rendered_content = (
        rendered.get("content")
        if isinstance(rendered, dict)
        else getattr(rendered, "content", "")
    )
    assert "Q: What is 2+2?" in rendered_content


@pytest.mark.asyncio
async def test_judge_rubric_routes_system_and_user_prompts(mock_client):
    mock_client.set_default_response("CORRECT")
    rubric = JudgeRubric(
        judge_client=mock_client,
        judge_model="test-model",
        judge_system_prompt="Grade: CORRECT or INCORRECT. One word.",
        judge_prompt="Target: {answer}\nResponse: {response}",
    )

    await rubric.judge(
        prompt=[UserMessage(content="Ignored by this template")],
        completion=[{"role": "assistant", "content": "0.33"}],
        answer="0.32",
        state=State(_seed=True),
    )

    routed_prompt = mock_client.last_call_kwargs["prompt"]
    assert [getattr(msg, "role", None) for msg in routed_prompt] == ["system", "user"]
    assert routed_prompt[0].content == "Grade: CORRECT or INCORRECT. One word."
    assert routed_prompt[1].content == "Target: 0.32\nResponse: 0.33"


@pytest.mark.asyncio
async def test_judge_rubric_caches_via_state_response(mock_client):
    """Cache seam: repeat calls with the same rendered prompt on the
    same state must be served from ``state['judge_response']`` without
    hitting the backend. This is the coalescing path that lets
    DebateRubric's ``_grade`` and ``_match`` both fire with the same
    target without double-billing the judge."""
    mock_client.set_default_response("INCORRECT")
    rubric = JudgeRubric(
        judge_client=mock_client,
        judge_model="test-model",
        judge_prompt="Q: {question}\nA: {answer}\nR: {response}",
    )
    state: State = State(_seed=True)
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    completion = [{"role": "assistant", "content": "five"}]

    first = await rubric.judge(
        prompt=prompt, completion=completion, answer="4", state=state
    )
    assert first == "INCORRECT"
    assert mock_client.call_count == 1
    cache = state.get("judge_response")
    assert isinstance(cache, dict) and len(cache) == 1

    second = await rubric.judge(
        prompt=prompt, completion=completion, answer="4", state=state
    )
    assert second == "INCORRECT"
    assert mock_client.call_count == 1  # served from cache


@pytest.mark.asyncio
async def test_judge_rubric_reuses_cache_across_states(mock_client):
    mock_client.set_default_response("CORRECT")
    rubric = JudgeRubric(
        judge_client=mock_client,
        judge_model="test-model",
        judge_prompt="Q: {question}\nA: {answer}\nR: {response}",
    )
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    completion = [{"role": "assistant", "content": "four"}]

    first = await rubric.judge(
        prompt=prompt,
        completion=completion,
        answer="4",
        state=State(_seed=True),
    )
    second_state: State = State(_seed=True)
    second = await rubric.judge(
        prompt=prompt,
        completion=completion,
        answer="4",
        state=second_state,
    )

    assert first == second == "CORRECT"
    assert mock_client.call_count == 1
    assert rubric.judge_cache_stats["misses"] == 1
    assert rubric.judge_cache_stats["hits"] == 1
    assert isinstance(second_state.get("judge_response"), dict)


@pytest.mark.asyncio
async def test_judge_rubric_reuses_persistent_evidence_across_instances(
    tmp_path, mock_client
):
    cache_path = tmp_path / "judge-evidence.sqlite3"
    mock_client.set_default_response("CORRECT")
    first_rubric = JudgeRubric(
        judge_client=mock_client,
        judge_model="test-model",
        judge_prompt="Q: {question}\nA: {answer}\nR: {response}",
        judge_persistent_cache_path=str(cache_path),
        judge_rubric_family="strict_equivalence",
        judge_variant_id="strict",
    )
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    completion = [{"role": "assistant", "content": "four"}]

    first = await first_rubric.judge(
        prompt=prompt,
        completion=completion,
        answer="4",
        state=State(example_id="q1"),
    )

    assert first == "CORRECT"
    assert mock_client.call_count == 1

    second_client = type(mock_client)()
    second_client.set_default_response("INCORRECT")
    second_rubric = JudgeRubric(
        judge_client=second_client,
        judge_model="different-model-is-metadata-not-identity",
        judge_prompt="Different wording: {answer} vs {response}",
        judge_sampling_args={"temperature": 1.0, "max_completion_tokens": 999},
        judge_persistent_cache_path=str(cache_path),
        judge_rubric_family="strict_equivalence",
        judge_variant_id="strict",
    )

    second = await second_rubric.judge(
        prompt=prompt,
        completion=completion,
        answer="4",
        state=State(example_id="q1"),
    )

    assert second == "CORRECT"
    assert second_client.call_count == 0
    assert second_rubric.judge_cache_stats["persistent_hits"] == 1
    assert second_rubric.judge_cache_stats["persistent_decision_hits"] == 1


@pytest.mark.asyncio
async def test_judge_rubric_persistent_hit_records_raw_response_and_decision(
    tmp_path, mock_client
):
    cache_path = tmp_path / "judge-evidence.sqlite3"
    mock_client.set_default_response("CORRECT\nextra rationale")
    first_rubric = JudgeRubric(
        judge_client=mock_client,
        judge_model="test-model",
        judge_prompt="Q: {question}\nA: {answer}\nR: {response}",
        judge_positive_label="CORRECT",
        judge_negative_label="INCORRECT",
        judge_persistent_cache_path=str(cache_path),
        judge_rubric_family="strict_equivalence",
        judge_variant_id="strict",
    )
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    completion = [{"role": "assistant", "content": "four"}]

    assert (
        await first_rubric.judge(prompt, completion, "4", State(example_id="q1"))
        == "CORRECT"
    )

    second_client = type(mock_client)()
    second_client.set_default_response("INCORRECT")
    second_rubric = JudgeRubric(
        judge_client=second_client,
        judge_model="test-model",
        judge_prompt="Q: {question}\nA: {answer}\nR: {response}",
        judge_positive_label="CORRECT",
        judge_negative_label="INCORRECT",
        judge_persistent_cache_path=str(cache_path),
        judge_rubric_family="strict_equivalence",
        judge_variant_id="strict",
    )
    state = State(example_id="q1")

    verdict = await second_rubric.judge(prompt, completion, "4", state)

    assert verdict == "CORRECT"
    assert second_client.call_count == 0
    assert state["judge_response"]
    assert next(iter(state["judge_response"].values())) == "CORRECT\nextra rationale"
    decision = state["judge_decision_last"]
    assert decision["hard_response"] == "CORRECT"
    assert decision["p_correct"] == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_judge_rubric_persistent_evidence_separates_variants(
    tmp_path, mock_client
):
    cache_path = tmp_path / "judge-evidence.sqlite3"
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    completion = [{"role": "assistant", "content": "four"}]

    mock_client.set_default_response("CORRECT")
    strict = JudgeRubric(
        judge_client=mock_client,
        judge_model="test-model",
        judge_prompt="Q: {question}\nA: {answer}\nR: {response}",
        judge_persistent_cache_path=str(cache_path),
        judge_rubric_family="strict_equivalence",
        judge_variant_id="strict",
    )
    assert (
        await strict.judge(prompt, completion, "4", State(example_id="q1")) == "CORRECT"
    )

    variant_client = type(mock_client)()
    variant_client.set_default_response("INCORRECT")
    numeric_guard = JudgeRubric(
        judge_client=variant_client,
        judge_model="test-model",
        judge_prompt="Numbers must match exactly: {answer} vs {response}",
        judge_persistent_cache_path=str(cache_path),
        judge_rubric_family="strict_equivalence",
        judge_variant_id="numeric_guard",
    )

    assert (
        await numeric_guard.judge(prompt, completion, "4", State(example_id="q1"))
        == "INCORRECT"
    )
    assert variant_client.call_count == 1
    assert numeric_guard.judge_cache_stats["persistent_misses"] == 1


@pytest.mark.asyncio
async def test_judge_rubric_min_persistent_samples_bypasses_process_cache(
    tmp_path, mock_client
):
    cache_path = tmp_path / "judge-evidence.sqlite3"
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    completion = [{"role": "assistant", "content": "four"}]
    rubric = JudgeRubric(
        judge_client=mock_client,
        judge_model="test-model",
        judge_prompt="Q: {question}\nA: {answer}\nR: {response}",
        judge_persistent_cache_path=str(cache_path),
        judge_persistent_cache_min_samples=2,
        judge_rubric_family="strict_equivalence",
        judge_variant_id="strict",
    )

    mock_client.set_default_response("CORRECT")
    assert (
        await rubric.judge(prompt, completion, "4", State(example_id="q1")) == "CORRECT"
    )
    mock_client.set_default_response("INCORRECT")
    assert (
        await rubric.judge(prompt, completion, "4", State(example_id="q1"))
        == "INCORRECT"
    )

    mock_client.set_default_response("SHOULD_NOT_CALL")
    assert (
        await rubric.judge(prompt, completion, "4", State(example_id="q1"))
        == "INCORRECT"
    )
    assert mock_client.call_count == 2
    assert rubric.judge_cache_stats["persistent_hits"] == 1


@pytest.mark.asyncio
async def test_judge_rubric_min_persistent_samples_bypasses_inflight_dedupe(
    tmp_path, mock_client
):
    cache_path = tmp_path / "judge-evidence.sqlite3"
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    completion = [{"role": "assistant", "content": "four"}]
    responses = ["CORRECT", "INCORRECT"]

    async def slow_get_response(prompt, model, sampling_args, tools=None, **kwargs):
        mock_client.call_count += 1
        await asyncio.sleep(0.01)
        mock_client.default_response = responses.pop(0)
        return mock_client._make_response(prompt)

    mock_client.get_response = slow_get_response
    rubric = JudgeRubric(
        judge_client=mock_client,
        judge_model="test-model",
        judge_prompt="Q: {question}\nA: {answer}\nR: {response}",
        judge_persistent_cache_path=str(cache_path),
        judge_persistent_cache_min_samples=2,
        judge_rubric_family="strict_equivalence",
        judge_variant_id="strict",
    )

    verdicts = await asyncio.gather(
        rubric.judge(prompt, completion, "4", State(example_id="q1")),
        rubric.judge(prompt, completion, "4", State(example_id="q1")),
    )

    assert sorted(verdicts) == ["CORRECT", "INCORRECT"]
    assert mock_client.call_count == 2
    assert rubric.judge_cache_stats["inflight_hits"] == 0
    assert rubric.judge_cache_stats["persistent_writes"] == 2


@pytest.mark.asyncio
async def test_judge_rubric_coalesces_concurrent_identical_calls(mock_client):
    mock_client.set_default_response("CORRECT")

    async def slow_get_response(prompt, model, sampling_args, tools=None, **kwargs):
        mock_client.call_count += 1
        mock_client.last_call_kwargs = {
            "prompt": prompt,
            "model": model,
            "sampling_args": sampling_args,
            "tools": tools,
            **kwargs,
        }
        await asyncio.sleep(0.01)
        return mock_client._make_response(prompt)

    mock_client.get_response = slow_get_response
    rubric = JudgeRubric(
        judge_client=mock_client,
        judge_model="test-model",
        judge_prompt="Q: {question}\nA: {answer}\nR: {response}",
    )
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    completion = [{"role": "assistant", "content": "four"}]

    verdicts = await asyncio.gather(
        *[
            rubric.judge(
                prompt=prompt,
                completion=completion,
                answer="4",
                state=State(_seed=True),
            )
            for _ in range(8)
        ]
    )

    assert verdicts == ["CORRECT"] * 8
    assert mock_client.call_count == 1
    assert rubric.judge_cache_stats["misses"] == 1
    assert rubric.judge_cache_stats["inflight_hits"] == 7


@pytest.mark.asyncio
async def test_judge_rubric_retries_transient_backend_errors(mock_client):
    mock_client.set_default_response("CORRECT")
    attempts = 0

    async def flaky_get_response(prompt, model, sampling_args, tools=None, **kwargs):
        nonlocal attempts
        attempts += 1
        mock_client.call_count += 1
        if attempts < 3:
            raise RuntimeError("empty grader response")
        return mock_client._make_response(prompt)

    mock_client.get_response = flaky_get_response
    rubric = JudgeRubric(
        judge_client=mock_client,
        judge_model="test-model",
        judge_prompt="Q: {question}\nA: {answer}\nR: {response}",
        judge_max_retries=2,
        judge_retry_delay_s=0,
    )

    verdict = await rubric.judge(
        prompt=[{"role": "user", "content": "What is 2+2?"}],
        completion=[{"role": "assistant", "content": "four"}],
        answer="4",
        state=State(_seed=True),
    )

    assert verdict == "CORRECT"
    assert mock_client.call_count == 3
    assert rubric.judge_cache_stats["retries"] == 2
