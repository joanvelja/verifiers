from __future__ import annotations

import pytest
import verifiers as vf

from verifiers.protocols.debate import resolve_prompts
from verifiers.protocols.debate.fields import FieldSpec
from verifiers.protocols.debate.rubric import DebateRubric
from verifiers.types import State


def _state(*, answer: str, debater_a: str, debater_b: str, winner: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": "Which answer is correct?"}],
        "answer": answer,
        "trajectory": [
            {
                "completion": [{"role": "assistant", "content": debater_a}],
                "extras": {
                    "member_id": "debater_a",
                    "phase": "propose",
                    "fields": {"answer": debater_a},
                },
            },
            {
                "completion": [{"role": "assistant", "content": debater_b}],
                "extras": {
                    "member_id": "debater_b",
                    "phase": "propose",
                    "fields": {"answer": debater_b},
                },
            },
            {
                "completion": [{"role": "assistant", "content": winner}],
                "extras": {
                    "member_id": "judge",
                    "phase": "final",
                    "fields": {"decision": winner},
                },
            },
        ],
    }


def _rubric(truth_member: str | None = None) -> DebateRubric:
    return DebateRubric(
        truth_member=truth_member,
        members=["debater_a", "debater_b", "judge"],
        prompts=resolve_prompts("selfplay"),
    )


class _FailingDiagnosticRubric(DebateRubric):
    async def verdict(
        self,
        answer: str,
        target: str,
        question: str,
        spec: FieldSpec | None,
        kind: str,
        state: State,
    ) -> bool:
        raise vf.Error(f"{kind} diagnostic failed")


@pytest.mark.asyncio
async def test_symmetric_debate_has_inert_episode_scalar_and_diagnostic_truth() -> None:
    score = await _rubric().build_marscore(
        _state(answer="B", debater_a="A", debater_b="B", winner="debater_b")
    )

    rewards = {m.member_id: m.reward for m in score.members}
    assert rewards == {"debater_a": -1.0, "debater_b": 1.0, "judge": 0.0}
    assert score.episode_scalar == 0.0
    assert score.episode_metrics["any_debater_correct"] == 1.0
    assert score.episode_metrics["all_debaters_correct"] == 0.0
    assert score.episode_metrics["judge_selected_correct"] == 1.0
    assert score.episode_metrics["judge_selected_correct_given_any_correct"] == 1.0
    assert "truth_member_correct" not in score.episode_metrics
    assert "truth_member_won" not in score.episode_metrics


@pytest.mark.asyncio
async def test_assigned_truth_member_is_explicit_and_tracks_side_success() -> None:
    score = await _rubric("debater_b").build_marscore(
        _state(answer="B", debater_a="A", debater_b="B", winner="debater_b")
    )

    assert score.episode_scalar == 1.0
    assert score.episode_metrics["truth_member_correct"] == 1.0
    assert score.episode_metrics["truth_member_won"] == 1.0


def test_truth_member_must_be_declared_member() -> None:
    with pytest.raises(ValueError, match="truth_member"):
        _rubric("debater_c")


@pytest.mark.asyncio
async def test_diagnostic_failures_do_not_override_judge_reward() -> None:
    rubric = _FailingDiagnosticRubric(
        members=["debater_a", "debater_b", "judge"],
        prompts=resolve_prompts("selfplay"),
    )

    score = await rubric.build_marscore(
        _state(answer="B", debater_a="A", debater_b="B", winner="debater_b")
    )

    rewards = {m.member_id: m.reward for m in score.members}
    member_metrics = {m.member_id: m.metrics for m in score.members}
    assert rewards == {"debater_a": -1.0, "debater_b": 1.0, "judge": 0.0}
    assert member_metrics["debater_a"]["grader_error"] == 1.0
    assert member_metrics["debater_b"]["grader_error"] == 1.0
    assert score.episode_metrics["matcher_error"] == 1.0
    assert "agreement" not in score.episode_metrics
    assert "judge_selected_correct" not in score.episode_metrics
