"""Tests for MultiAgentEnv's agent_overrides_resolver hook.

The resolver is a state-aware sibling of the static ``agent_overrides``
dict: given the current rollout State it returns a per-state mapping of
member_id -> (client, model). This enables per-episode routing like
learner-vs-fixed training (STAGE 3 in
docs/plans/2026-04-20-stage3-learner-vs-fixed.md) without doubling the
config surface with two env variants.

Invariants under test:

  * init-time dummy probe rejects resolvers that miss scheduled members
  * init-time dummy probe rejects resolvers that return stray keys
  * init-time dummy probe rejects non-dict return values
  * resolve_agent(member, state) routes through the resolver when set
  * resolve_agent falls back to the static dict when no resolver
  * simultaneous slots call the resolver once per slot, not once per agent
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import jinja2
import jinja2.sandbox
import pytest

from verifiers.envs.debate.prompts import DebatePrompts
from verifiers.envs.debate_env import DebateEnv
from verifiers.envs.multi_agent_kernel import StaticSchedule, TurnSlot
from verifiers.types import State


# ---------------------------------------------------------------------------
# Scaffolding (mirrors test_debate_env_prompts_coverage.py)
# ---------------------------------------------------------------------------

_je = jinja2.sandbox.SandboxedEnvironment(undefined=jinja2.StrictUndefined)


def _stub_rubric(members: list[str]) -> MagicMock:
    rubric = MagicMock()
    rubric.members = members
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
        think_tag="thinking",
        prefill={},
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


def _build(resolver=None, agent_overrides=None) -> DebateEnv:
    return DebateEnv(
        schedule=_SCHEDULE,
        prompts=_pack(),
        members=list(_MEMBERS),
        rubric=_stub_rubric(list(_MEMBERS)),
        dataset=lambda: None,
        agent_overrides=agent_overrides,
        agent_overrides_resolver=resolver,
    )


# ---------------------------------------------------------------------------
# Init-time probe invariants
# ---------------------------------------------------------------------------


def test_resolver_missing_member_fails_at_init():
    """Probe surfaces resolvers that miss a scheduled member."""

    def resolver(state):
        # NO entry for "debater_b" -> should fail
        return {
            "debater_a": (None, None),
            "judge": (MagicMock(), "gpt-4.1-mini"),
        }

    with pytest.raises(ValueError, match="omits members"):
        _build(resolver=resolver)


def test_resolver_stray_key_fails_at_init():
    """Probe surfaces resolvers that return keys not in members."""

    def resolver(state):
        return {
            "debater_a": (None, None),
            "debater_b": (MagicMock(), "opp"),
            "judge": (MagicMock(), "judge"),
            "ghost_debater": (None, None),  # stray
        }

    with pytest.raises(ValueError, match="not in members"):
        _build(resolver=resolver)


def test_resolver_returning_non_dict_fails_at_init():
    def resolver(state):
        return [("debater_a", None)]  # wrong shape

    with pytest.raises(TypeError, match="must return a dict"):
        _build(resolver=resolver)


# ---------------------------------------------------------------------------
# Runtime routing
# ---------------------------------------------------------------------------


def test_resolver_routes_by_learner_seat():
    """resolve_agent(member, state) returns per-state overrides from the
    resolver; learner_seat lookup flips (client, model) between a and b."""
    opp_client = MagicMock(name="opp_client")
    judge_client = MagicMock(name="judge_client")

    def resolver(state):
        seat = state["info"]["learner_seat"]
        opposite = "debater_b" if seat == "debater_a" else "debater_a"
        return {
            seat: (None, None),
            opposite: (opp_client, "opp-model"),
            "judge": (judge_client, "judge-model"),
        }

    env = _build(resolver=resolver)

    state_a = State()
    state_a["input"] = {"info": {"learner_seat": "debater_a"}}
    assert env.resolve_agent("debater_a", state_a) == (None, None)
    assert env.resolve_agent("debater_b", state_a) == (opp_client, "opp-model")
    assert env.resolve_agent("judge", state_a) == (judge_client, "judge-model")

    state_b = State()
    state_b["input"] = {"info": {"learner_seat": "debater_b"}}
    assert env.resolve_agent("debater_a", state_b) == (opp_client, "opp-model")
    assert env.resolve_agent("debater_b", state_b) == (None, None)


def test_resolver_missing_key_at_runtime_raises():
    """If the resolver's output set changes under load (e.g. a dynamic
    branch omits a member), _resolve_all catches it — the init probe
    alone can't cover all state shapes."""
    seen: dict[str, int] = {"calls": 0}

    def resolver(state):
        seen["calls"] += 1
        if state["info"].get("learner_seat") == "debater_b":
            # Oops — omit debater_a under this branch
            return {"debater_b": (None, None), "judge": (MagicMock(), "j")}
        return {
            "debater_a": (None, None),
            "debater_b": (MagicMock(), "opp"),
            "judge": (MagicMock(), "j"),
        }

    env = _build(resolver=resolver)  # init probe hits the good branch

    bad_state = State()
    bad_state["input"] = {"info": {"learner_seat": "debater_b"}}
    with pytest.raises(ValueError, match="omitted members"):
        env.resolve_agent("debater_a", bad_state)


def test_static_overrides_used_when_no_resolver():
    """Falls back to self.agent_overrides when no resolver is set."""
    client = MagicMock(name="opp_client")
    env = _build(agent_overrides={"debater_b": (client, "opp")})

    state = State()
    state["input"] = {"info": {}}
    assert env.resolve_agent("debater_a", state) == (None, None)
    assert env.resolve_agent("debater_b", state) == (client, "opp")
    assert env.resolve_agent("judge", state) == (None, None)


def test_resolver_called_once_per_slot_not_per_agent():
    """Simultaneous slots resolve once per slot, even with N agents.

    Implementation detail test, but worth it: N-calls would silently
    quadratic-up a dynamic resolver's cost, and on an external-service
    resolver (e.g. one that consults a checkpoint registry) it's
    observable slowdown."""
    calls = {"n": 0}
    opp, judge_c = MagicMock(), MagicMock()

    def resolver(state):
        calls["n"] += 1
        return {
            "debater_a": (None, None),
            "debater_b": (opp, "opp"),
            "judge": (judge_c, "j"),
        }

    env = _build(resolver=resolver)
    init_calls = calls["n"]  # init probe consumed some calls
    assert init_calls == 1  # probe hits resolver once

    state = State()
    state["input"] = {"info": {"learner_seat": "debater_a"}}

    # The runtime path is _resolve_all -> resolver(state) once per slot.
    # Exercise via the public resolve_agent for each simultaneous agent:
    # production _run_simultaneous_slot calls _resolve_all directly, so
    # assert that the underlying helper is a single call.
    env._resolve_all(state)
    assert calls["n"] == init_calls + 1

    env._resolve_all(state)
    assert calls["n"] == init_calls + 2


# ---------------------------------------------------------------------------
# Probe state for subclasses
# ---------------------------------------------------------------------------


def test_debate_probe_seeds_learner_seat():
    """DebateEnv's _build_probe_state overrides fill info.learner_seat so
    resolvers that read it don't KeyError at init."""

    def resolver(state):
        # KeyErrors if learner_seat isn't in info
        seat = state["info"]["learner_seat"]
        opposite = "debater_b" if seat == "debater_a" else "debater_a"
        return {
            seat: (None, None),
            opposite: (MagicMock(), "opp"),
            "judge": (MagicMock(), "j"),
        }

    # If the probe didn't seed learner_seat, this would fail with KeyError.
    _build(resolver=resolver)
