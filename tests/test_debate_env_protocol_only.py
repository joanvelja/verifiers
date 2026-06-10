from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import jinja2
import jinja2.sandbox
import pytest

from verifiers.protocols.debate.prompts import DebatePrompts
from verifiers.protocols.debate.env import DebateEnv, load_environment
from verifiers.envs.multi_agent_kernel import StaticSchedule, TurnSlot


_je = jinja2.sandbox.SandboxedEnvironment(undefined=jinja2.StrictUndefined)


def _stub_rubric(members: list[str]) -> MagicMock:
    rubric = MagicMock()
    rubric.members = members
    rubric.score_rollout = AsyncMock(return_value=None)
    rubric.dummy_score_rollout = AsyncMock(return_value=None)
    rubric.cleanup = AsyncMock(return_value=None)
    rubric.teardown = AsyncMock(return_value=None)
    return rubric


def _pack() -> DebatePrompts:
    members = ("debater_a", "debater_b", "judge")
    return DebatePrompts(
        system={m: _je.from_string("system") for m in members},
        user={
            "debater_a": {
                "propose": _je.from_string("argue"),
                "critique": _je.from_string("rebut"),
            },
            "debater_b": {
                "propose": _je.from_string("argue"),
                "critique": _je.from_string("rebut"),
            },
            "judge": {"final": _je.from_string("decide")},
        },
        question={m: _je.from_string("{{ task_prompt }}") for m in members},
        fields={},
        think_visibility={},
        opponent_wrap=None,
        judges={},
        source_ref="<test>",
    )


_SCHEDULE = StaticSchedule(
    (
        TurnSlot(slot_id=0, agents=("debater_a",), phase="propose"),
        TurnSlot(slot_id=1, agents=("debater_b",), phase="propose"),
        TurnSlot(slot_id=2, agents=("judge",), phase="final"),
    )
)
_MEMBERS = ["debater_a", "debater_b", "judge"]


def _build(**kwargs) -> DebateEnv:
    return DebateEnv(
        schedule=_SCHEDULE,
        prompts=_pack(),
        members=list(_MEMBERS),
        rubric=_stub_rubric(list(_MEMBERS)),
        dataset=lambda: None,
        **kwargs,
    )


def test_debate_factory_rejects_misspelled_kwargs() -> None:
    with pytest.raises(TypeError, match="truth_membr"):
        load_environment(
            schedule_slots=[
                {"slot_id": 0, "agents": ["debater_a"], "phase": "propose"},
                {"slot_id": 1, "agents": ["debater_b"], "phase": "propose"},
                {"slot_id": 2, "agents": ["judge"], "phase": "final"},
            ],
            members=list(_MEMBERS),
            prompts=_pack(),
            truth_membr="debater_a",
            dataset=lambda: None,
        )


@pytest.mark.asyncio
async def test_rollout_records_generation_timing_and_runs_cleanup(mock_client):
    env = _build()
    cleanup = AsyncMock()
    env.cleanup = cleanup

    output = await env.run_rollout(
        {
            "prompt": [{"role": "user", "content": "Which option is correct?"}],
            "answer": "A",
            "example_id": "timing-cleanup",
            "info": {},
        },
        mock_client,
        "test-model",
        {},
    )

    cleanup.assert_awaited_once()
    timing = output["timing"]
    assert timing["generation"]["start"] > 0.0
    assert timing["generation"]["end"] >= timing["generation"]["start"]
    assert 0.0 <= timing["total"] < 60.0
