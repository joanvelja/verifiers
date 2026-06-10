from unittest.mock import AsyncMock, MagicMock

import jinja2
import jinja2.sandbox
import pytest

from verifiers.protocols.debate.prompts import DebatePrompts
from verifiers.protocols.debate.env import DebateEnv
from verifiers.envs.multi_agent_kernel import (
    ContentChannels,
    KernelState,
    StaticSchedule,
    TurnSlot,
    apply_action,
)
from verifiers.types import AssistantMessage, State, UserMessage


_JINJA = jinja2.sandbox.SandboxedEnvironment(undefined=jinja2.StrictUndefined)
_MEMBERS = ["debater_a", "debater_b", "judge"]
_SCHEDULE = StaticSchedule(
    (
        TurnSlot(slot_id=0, agents=("debater_a",), phase="propose"),
        TurnSlot(slot_id=1, agents=("debater_b",), phase="propose"),
        TurnSlot(slot_id=2, agents=("debater_a",), phase="critique"),
        TurnSlot(slot_id=3, agents=("debater_b",), phase="critique"),
        TurnSlot(slot_id=4, agents=("judge",), phase="final"),
    )
)
_SIMULTANEOUS_SCHEDULE = StaticSchedule(
    (
        TurnSlot(
            slot_id=0,
            agents=("debater_a", "debater_b"),
            phase="propose",
        ),
        TurnSlot(
            slot_id=1,
            agents=("debater_a", "debater_b"),
            phase="critique",
        ),
        TurnSlot(slot_id=2, agents=("judge",), phase="final"),
    )
)


def _template(source: str):
    return _JINJA.from_string(source)


def _prompts() -> DebatePrompts:
    return DebatePrompts(
        system={member: _template("system {{ viewer_id }}") for member in _MEMBERS},
        user={
            "debater_a": {
                "propose": _template("A propose {{ round_index }}/{{ num_rounds }}"),
                "critique": _template("A critique {{ round_index }}/{{ num_rounds }}"),
            },
            "debater_b": {
                "propose": _template("B propose {{ round_index }}/{{ num_rounds }}"),
                "critique": _template("B critique {{ round_index }}/{{ num_rounds }}"),
            },
            "judge": {"final": _template("judge final")},
        },
        question={member: _template("{{ task_prompt }}") for member in _MEMBERS},
        fields={},
        think_visibility={},
        opponent_wrap=None,
        judges={},
        source_ref="<test>",
    )


def _rubric():
    rubric = MagicMock()
    rubric.members = list(_MEMBERS)
    rubric.score_rollout = AsyncMock(return_value=None)
    rubric.dummy_score_rollout = AsyncMock(return_value=None)
    rubric.cleanup = AsyncMock(return_value=None)
    rubric.teardown = AsyncMock(return_value=None)
    return rubric


class CountingDebateEnv(DebateEnv):
    def __init__(self, schedule: StaticSchedule = _SCHEDULE) -> None:
        super().__init__(
            schedule=schedule,
            prompts=_prompts(),
            members=list(_MEMBERS),
            rubric=_rubric(),
            dataset=lambda: None,
        )
        self.own_turn_renders = 0
        self.opponent_renders = 0

    def _render_own_turn(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.own_turn_renders += 1
        return super()._render_own_turn(*args, **kwargs)

    def _render_opponent_message(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.opponent_renders += 1
        return super()._render_opponent_message(*args, **kwargs)


def _state_with_transcript(*actions: tuple[str, str]) -> State:
    kernel = KernelState(slot_index=0)
    for member_id, content in actions:
        result = apply_action(
            kernel,
            _SCHEDULE,
            member_id,
            content,
            token_count=1,
            channels=ContentChannels(public=content),
        )
        kernel = result.new_state

    state = State()
    state["prompt"] = [UserMessage(content="Which option is correct?")]
    state["answer"] = "A"
    state["_kernel"] = kernel
    return state


def _dump(messages):
    return [message.model_dump() for message in messages]


def _base_state() -> State:
    state = State()
    state["prompt"] = [UserMessage(content="Which option is correct?")]
    state["answer"] = "A"
    state["_kernel"] = KernelState(slot_index=0)
    return state


def _apply(
    kernel: KernelState,
    schedule: StaticSchedule,
    member_id: str,
    content: str,
) -> KernelState:
    result = apply_action(
        kernel,
        schedule,
        member_id,
        content,
        token_count=1,
        channels=ContentChannels(public=content),
    )
    return result.new_state


@pytest.mark.asyncio
async def test_simultaneous_replay_preserves_member_actual_io_prefix() -> None:
    env = CountingDebateEnv(_SIMULTANEOUS_SCHEDULE)
    state = _base_state()
    propose_slot = _SIMULTANEOUS_SCHEDULE.current_slot(state["_kernel"])
    assert propose_slot is not None

    original_b_prompt = await env._prepare_prompt(state, "debater_b", propose_slot)
    state["_kernel"] = _apply(
        state["_kernel"],
        _SIMULTANEOUS_SCHEDULE,
        "debater_a",
        "opening A",
    )
    state["_kernel"] = _apply(
        state["_kernel"],
        _SIMULTANEOUS_SCHEDULE,
        "debater_b",
        "opening B",
    )

    critique_slot = _SIMULTANEOUS_SCHEDULE.current_slot(state["_kernel"])
    assert critique_slot is not None
    critique_prompt = await env._prepare_prompt(state, "debater_b", critique_slot)
    expected_prefix = [
        *original_b_prompt,
        AssistantMessage(content="opening B"),
    ]

    assert _dump(critique_prompt[: len(expected_prefix)]) == _dump(expected_prefix)
    assert "[debater_a] opening A" in critique_prompt[len(expected_prefix)].content


@pytest.mark.asyncio
async def test_debate_prompt_body_cache_preserves_prompt_output() -> None:
    env = CountingDebateEnv()
    state = _state_with_transcript(
        ("debater_a", "opening A"),
        ("debater_b", "opening B"),
    )
    slot = TurnSlot(slot_id=2, agents=("debater_a",), phase="critique")

    cached = await env.build_prompt(state, "debater_a", slot)
    state.pop("_debate_prompt_body_cache")
    fresh = await env.build_prompt(state, "debater_a", slot)

    assert _dump(cached) == _dump(fresh)


@pytest.mark.asyncio
async def test_debate_prompt_body_cache_renders_only_new_transcript_suffix() -> None:
    env = CountingDebateEnv()
    state = _state_with_transcript(
        ("debater_a", "opening A"),
        ("debater_b", "opening B"),
    )
    slot = TurnSlot(slot_id=2, agents=("debater_a",), phase="critique")

    await env.build_prompt(state, "debater_a", slot)
    assert env.own_turn_renders == 1
    assert env.opponent_renders == 1

    await env.build_prompt(state, "debater_a", slot)
    assert env.own_turn_renders == 1
    assert env.opponent_renders == 1

    result = apply_action(
        state["_kernel"],
        _SCHEDULE,
        "debater_a",
        "critique A",
        token_count=1,
        channels=ContentChannels(public="critique A"),
    )
    state["_kernel"] = result.new_state

    await env.build_prompt(state, "debater_a", slot)
    assert env.own_turn_renders == 2
    assert env.opponent_renders == 1


@pytest.mark.asyncio
async def test_debate_prompt_body_cache_invalidates_replaced_transcript_prefix() -> (
    None
):
    env = CountingDebateEnv()
    state = _state_with_transcript(
        ("debater_a", "opening A"),
        ("debater_b", "opening B"),
    )
    slot = TurnSlot(slot_id=2, agents=("debater_a",), phase="critique")

    await env.build_prompt(state, "debater_a", slot)
    state["_kernel"] = _state_with_transcript(
        ("debater_a", "replacement A"),
        ("debater_b", "replacement B"),
    )["_kernel"]

    messages = await env.build_prompt(state, "debater_a", slot)
    contents = [message.content for message in messages]

    assert any("replacement A" in content for content in contents)
    assert any("replacement B" in content for content in contents)
    assert not any("opening A" in content for content in contents)
    assert not any("opening B" in content for content in contents)


@pytest.mark.asyncio
async def test_debate_prompt_body_cache_invalidates_replaced_prefix_with_extra_suffix() -> (
    None
):
    env = CountingDebateEnv()
    state = _state_with_transcript(
        ("debater_a", "opening A"),
        ("debater_b", "opening B"),
    )
    slot = TurnSlot(slot_id=2, agents=("debater_a",), phase="critique")

    await env.build_prompt(state, "debater_a", slot)
    state["_kernel"] = _state_with_transcript(
        ("debater_a", "replacement A"),
        ("debater_b", "replacement B"),
        ("debater_a", "new A"),
    )["_kernel"]

    messages = await env.build_prompt(state, "debater_a", slot)
    contents = [message.content for message in messages]

    assert any("replacement A" in content for content in contents)
    assert any("replacement B" in content for content in contents)
    assert any("new A" in content for content in contents)
    assert not any("opening A" in content for content in contents)
    assert not any("opening B" in content for content in contents)
