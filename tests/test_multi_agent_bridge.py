from __future__ import annotations

from typing import Any

import pytest
from verifiers.multi_agent_bridge import rollout_to_member_rollouts
from verifiers.rubrics.multi_agent_rubric import MultiAgentRubric
from verifiers.types import (
    ClientConfig,
    GenerationTarget,
    MARScore,
    MemberGenerationPlan,
    MemberScore,
)
from verifiers.utils.save_utils import state_to_output


def _step(member_id: str) -> dict[str, Any]:
    return {
        "extras": {"member_id": member_id},
        "advantage": None,
        "reward": None,
        "tokens": None,
        "is_truncated": False,
        "prompt": [],
        "completion": [],
        "response": None,
        "trajectory_id": "t-1",
    }


def _output(**overrides: Any) -> dict[str, Any]:
    output: dict[str, Any] = {
        "example_id": 7,
        "task": "real_task",
        "sampling_args": {"temperature": 0.7, "max_tokens": 128},
        "trajectory_id": "t-1",
        "trajectory": [_step("A")],
        "mar_score": MARScore(
            members=[MemberScore(member_id="A", reward=1.0)],
            episode_scalar=1.0,
        ).model_dump(exclude_none=True),
    }
    output.update(overrides)
    return output


def test_bridge_preserves_output_task() -> None:
    members = rollout_to_member_rollouts(_output(task="task_a"))
    assert all(member["task"] == "task_a" for member in members)


def test_bridge_preserves_full_sampling_args() -> None:
    members = rollout_to_member_rollouts(_output())
    assert members[0]["sampling_args"] == {"temperature": 0.7, "max_tokens": 128}


def test_bridge_preserves_tool_defs_for_member_token_backfill() -> None:
    tool_defs = [
        {
            "type": "function",
            "function": {"name": "lookup", "description": "Look up a fact."},
        }
    ]
    members = rollout_to_member_rollouts(_output(tool_defs=tool_defs))
    assert members[0]["tool_defs"] == tool_defs


def test_bridge_defaults_temperature_when_missing_but_keeps_other_args() -> None:
    members = rollout_to_member_rollouts(_output(sampling_args={"max_tokens": 64}))
    assert members[0]["sampling_args"] == {"temperature": 1.0, "max_tokens": 64}


@pytest.mark.parametrize(
    "missing_key",
    ["task", "example_id", "sampling_args", "trajectory_id", "mar_score", "trajectory"],
)
def test_bridge_keyerror_on_missing_canonical_field(missing_key: str) -> None:
    output = _output()
    output.pop(missing_key)
    with pytest.raises(KeyError, match=missing_key):
        rollout_to_member_rollouts(output)


def test_bridge_rejects_unknown_member_in_trajectory() -> None:
    output = _output(trajectory=[_step("A"), _step("GHOST")])
    with pytest.raises(ValueError, match="GHOST"):
        rollout_to_member_rollouts(output)


def test_bridge_rejects_non_errored_member_with_no_steps() -> None:
    output = _output(
        trajectory=[_step("A")],
        mar_score=MARScore(
            members=[
                MemberScore(member_id="A", reward=1.0),
                MemberScore(member_id="B", reward=-1.0),
            ],
            episode_scalar=1.0,
        ).model_dump(exclude_none=True),
    )

    with pytest.raises(ValueError, match="B"):
        rollout_to_member_rollouts(output)


def test_bridge_allows_errored_member_with_no_steps() -> None:
    output = _output(
        error={"error": "ModelError"},
        trajectory=[_step("A")],
        mar_score=MARScore(
            members=[
                MemberScore(member_id="A", reward=0.0),
                MemberScore(member_id="B", reward=0.0),
            ],
            episode_scalar=0.0,
        ).model_dump(exclude_none=True),
    )

    members = rollout_to_member_rollouts(output)
    by_id = {member["member_id"]: member for member in members}
    assert len(by_id["A"]["trajectory"]) == 1
    assert by_id["B"]["trajectory"] == []


def test_bridge_episode_id_from_trajectory_id() -> None:
    members = rollout_to_member_rollouts(_output(trajectory_id="abc-123"))
    assert all(member["episode_id"] == "abc-123" for member in members)


@pytest.mark.parametrize(
    "reserved_key",
    [
        "reward",
        "example_id",
        "task",
        "metrics",
        "trajectory",
        "error_chain",
        "long_error_chain",
    ],
)
def test_marscore_episode_metrics_reserved_key_rejected(reserved_key: str) -> None:
    mar = MARScore(
        members=[MemberScore(member_id="A", reward=0.0)],
        episode_scalar=0.0,
        episode_metrics={reserved_key: 1.0},
    )
    with pytest.raises(ValueError, match=reserved_key):
        mar.to_metrics_flat()


@pytest.mark.parametrize(
    "reserved_key",
    ["reward", "example_id", "task", "metrics", "error_chain", "long_error_chain"],
)
def test_marscore_member_metrics_reserved_key_rejected(reserved_key: str) -> None:
    mar = MARScore(
        members=[MemberScore(member_id="A", reward=1.0, metrics={reserved_key: 0.5})],
        episode_scalar=1.0,
    )
    with pytest.raises(ValueError, match=reserved_key):
        mar.to_metrics_flat()


def test_marscore_non_reserved_keys_project_normally() -> None:
    mar = MARScore(
        members=[
            MemberScore(member_id="A", reward=1.0, metrics={"accuracy": 0.7}),
            MemberScore(member_id="B", reward=-1.0, metrics={"accuracy": 0.3}),
        ],
        episode_scalar=1.0,
        episode_metrics={"agreement": 0.5},
    )
    assert mar.to_metrics_flat() == {
        "agreement": 0.5,
        "reward/A": 1.0,
        "reward/B": -1.0,
        "accuracy/A": 0.7,
        "accuracy/B": 0.3,
    }


class _FakeMARubric(MultiAgentRubric):
    members = ["A"]

    async def build_marscore(self, state):
        return MARScore(
            members=[MemberScore(member_id="A", reward=0.0)],
            episode_scalar=0.0,
        )


class _MissingMembersRubric(MultiAgentRubric):
    async def build_marscore(self, state):
        return MARScore(
            members=[MemberScore(member_id="A", reward=0.0)],
            episode_scalar=0.0,
        )


class _EmptyMembersRubric(MultiAgentRubric):
    members = []

    async def build_marscore(self, state):
        return MARScore(members=[], episode_scalar=0.0)


class _DuplicateMembersRubric(MultiAgentRubric):
    members = ["A", "A"]

    async def build_marscore(self, state):
        return MARScore(
            members=[MemberScore(member_id="A", reward=0.0)],
            episode_scalar=0.0,
        )


def test_multi_agent_rubric_requires_members() -> None:
    with pytest.raises(ValueError, match="members"):
        _MissingMembersRubric()

    with pytest.raises(ValueError, match="members"):
        _EmptyMembersRubric()

    with pytest.raises(ValueError, match="Duplicate"):
        _DuplicateMembersRubric()


def test_multi_agent_rubric_converts_scoring_error_to_marscore() -> None:
    state = {"error": RuntimeError("boom")}
    rubric = _FakeMARubric()

    import asyncio

    asyncio.run(rubric.score_rollout(state))

    mar = state["mar_score"]
    assert mar.episode_scalar == 0.0
    assert mar.episode_metrics == {"errored_rollout": 1.0}
    assert mar.episode_error == {
        "error_type": "RuntimeError",
        "error_phase": "rollout",
    }


def test_multi_agent_rubric_prompt_too_long_records_error() -> None:
    # Prove-it regression for the fan-out crash (#16). A prompt_too_long rollout
    # must set state["error"], not only an errored mar_score. Without it the
    # fan-out bridge sees a declared member with no trajectory step + error=None
    # and RAISES (see test_bridge_rejects_non_errored_member_with_no_steps),
    # which kills the whole orchestrator run instead of dropping one bad rollout.
    # With the error set, the bridge drops it cleanly
    # (see test_bridge_allows_errored_member_with_no_steps).
    import asyncio

    state = {"prompt_too_long": True}
    asyncio.run(_FakeMARubric().score_rollout(state))

    assert state.get("error") is not None, (
        "prompt_too_long must record state['error'] so the bridge drops the "
        "errored rollout instead of raising on a member with no trajectory steps"
    )
    mar = state["mar_score"]
    assert mar.episode_metrics == {"errored_rollout": 1.0}
    assert mar.episode_error["error_type"] == "prompt_too_long"


def test_state_to_output_projects_marscore_and_member_metrics(make_state) -> None:
    state = make_state(reward=123.0)
    state["task"] = "debate_task"
    state["sampling_args"] = {"max_tokens": 64}
    state["trajectory_id"] = "episode-1"
    state["mar_score"] = MARScore(
        members=[
            MemberScore(member_id="A", reward=1.0, metrics={"accuracy": 0.7}),
            MemberScore(member_id="B", reward=-1.0, parse_error_count=2),
        ],
        episode_scalar=0.25,
        episode_metrics={"agreement": 0.5},
    )

    output = state_to_output(state, state_columns=["sampling_args", "trajectory_id"])

    assert output["task"] == "debate_task"
    assert output["reward"] == 0.25
    assert output["metrics"] == {
        "agreement": 0.5,
        "reward/A": 1.0,
        "reward/B": -1.0,
        "accuracy/A": 0.7,
        "parse_errors/B": 2.0,
    }
    assert output["sampling_args"] == {"max_tokens": 64}
    assert output["trajectory_id"] == "episode-1"
    assert output["mar_score"]["episode_scalar"] == 0.25


def test_state_to_output_rejects_reserved_state_column(make_state) -> None:
    state = make_state()
    with pytest.raises(ValueError, match="standard output field"):
        state_to_output(state, state_columns=["reward"])


def test_member_generation_plan_validates_and_selects_target() -> None:
    target = GenerationTarget(
        client=ClientConfig(api_base_url="http://localhost:8000/v1"),
        model="learner",
        sampling_args={"temperature": 0.2},
    )
    plan = MemberGenerationPlan(members={"debater_a": target})

    assert plan.target_for("debater_a").model == "learner"
    with pytest.raises(KeyError, match="debater_b"):
        plan.target_for("debater_b")
    with pytest.raises(ValueError, match="non-empty"):
        MemberGenerationPlan(members={"": target})
