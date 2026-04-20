"""Tests for MultiAgentEnv's agent_bindings_fn hook.

The bindings fn is a state-aware sibling of the static ``agent_bindings``
dict: given the current rollout State it returns a per-state mapping of
member_id -> (client, model). This enables per-episode routing like
external-opponent debate training (STAGE 3 in
docs/plans/2026-04-20-stage3-learner-vs-fixed.md) without doubling the
config surface with two env variants.

Invariants under test:

  * init-time dummy probe rejects bindings fns that miss scheduled members
  * init-time dummy probe rejects bindings fns that return stray keys
  * init-time dummy probe rejects non-dict return values
  * get_agent_binding(member, state) routes through the fn when set
  * get_agent_binding falls back to the static dict when no fn
  * simultaneous slots call the fn once per slot, not once per agent
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


def _build(bindings_fn=None, agent_bindings=None) -> DebateEnv:
    return DebateEnv(
        schedule=_SCHEDULE,
        prompts=_pack(),
        members=list(_MEMBERS),
        rubric=_stub_rubric(list(_MEMBERS)),
        dataset=lambda: None,
        agent_bindings=agent_bindings,
        agent_bindings_fn=bindings_fn,
    )


# ---------------------------------------------------------------------------
# Init-time probe invariants
# ---------------------------------------------------------------------------


def test_bindings_fn_missing_member_fails_at_init():
    """Probe surfaces bindings fns that miss a scheduled member."""

    def bindings_fn(state):
        # NO entry for "debater_b" -> should fail
        return {
            "debater_a": (None, None),
            "judge": (MagicMock(), "gpt-4.1-mini"),
        }

    with pytest.raises(ValueError, match="omits members"):
        _build(bindings_fn=bindings_fn)


def test_bindings_fn_stray_key_fails_at_init():
    """Probe surfaces bindings fns that return keys not in members."""

    def bindings_fn(state):
        return {
            "debater_a": (None, None),
            "debater_b": (MagicMock(), "opp"),
            "judge": (MagicMock(), "judge"),
            "ghost_debater": (None, None),  # stray
        }

    with pytest.raises(ValueError, match="not in members"):
        _build(bindings_fn=bindings_fn)


def test_bindings_fn_returning_non_dict_fails_at_init():
    def bindings_fn(state):
        return [("debater_a", None)]  # wrong shape

    with pytest.raises(TypeError, match="must return a dict"):
        _build(bindings_fn=bindings_fn)


# ---------------------------------------------------------------------------
# Runtime routing
# ---------------------------------------------------------------------------


def test_bindings_fn_routes_by_learner_seat():
    """get_agent_binding(member, state) returns per-state bindings from
    the fn; learner_seat lookup flips (client, model) between a and b."""
    opp_client = MagicMock(name="opp_client")
    judge_client = MagicMock(name="judge_client")

    def bindings_fn(state):
        seat = state["info"]["learner_seat"]
        opposite = "debater_b" if seat == "debater_a" else "debater_a"
        return {
            seat: (None, None),
            opposite: (opp_client, "opp-model"),
            "judge": (judge_client, "judge-model"),
        }

    env = _build(bindings_fn=bindings_fn)

    state_a = State()
    state_a["input"] = {"info": {"learner_seat": "debater_a"}}
    assert env.get_agent_binding("debater_a", state_a) == (None, None)
    assert env.get_agent_binding("debater_b", state_a) == (opp_client, "opp-model")
    assert env.get_agent_binding("judge", state_a) == (judge_client, "judge-model")

    state_b = State()
    state_b["input"] = {"info": {"learner_seat": "debater_b"}}
    assert env.get_agent_binding("debater_a", state_b) == (opp_client, "opp-model")
    assert env.get_agent_binding("debater_b", state_b) == (None, None)


def test_bindings_fn_returning_non_dict_at_runtime_raises():
    """A dynamic branch that returns a non-dict iterable with correct
    member names would pass a naive membership check (since ``set(list)``
    works) but crash opaquely at the ``.get`` call later. _get_bindings
    re-checks shape at runtime to surface a clear TypeError."""

    def bindings_fn(state):
        seat = state["info"].get("learner_seat")
        if seat == "debater_b":
            # Buggy branch: returns a list that happens to contain the
            # right member names — passes set-difference but breaks .get.
            return ["debater_a", "debater_b", "judge"]
        return {
            "debater_a": (None, None),
            "debater_b": (MagicMock(), "opp"),
            "judge": (MagicMock(), "j"),
        }

    env = _build(bindings_fn=bindings_fn)  # init probe hits the dict branch

    bad_state = State()
    bad_state["input"] = {"info": {"learner_seat": "debater_b"}}
    with pytest.raises(TypeError, match="must return a dict"):
        env.get_agent_binding("debater_a", bad_state)


def test_bindings_fn_missing_key_at_runtime_raises():
    """If the fn's output set changes under load (e.g. a dynamic branch
    omits a member), _get_bindings catches it — the init probe alone
    can't cover all state shapes."""
    seen: dict[str, int] = {"calls": 0}

    def bindings_fn(state):
        seen["calls"] += 1
        if state["info"].get("learner_seat") == "debater_b":
            # Oops — omit debater_a under this branch
            return {"debater_b": (None, None), "judge": (MagicMock(), "j")}
        return {
            "debater_a": (None, None),
            "debater_b": (MagicMock(), "opp"),
            "judge": (MagicMock(), "j"),
        }

    env = _build(bindings_fn=bindings_fn)  # init probe hits the good branch

    bad_state = State()
    bad_state["input"] = {"info": {"learner_seat": "debater_b"}}
    with pytest.raises(ValueError, match="omitted members"):
        env.get_agent_binding("debater_a", bad_state)


def test_static_bindings_used_when_no_fn():
    """Falls back to self.agent_bindings when no bindings fn is set."""
    client = MagicMock(name="opp_client")
    env = _build(agent_bindings={"debater_b": (client, "opp")})

    state = State()
    state["input"] = {"info": {}}
    assert env.get_agent_binding("debater_a", state) == (None, None)
    assert env.get_agent_binding("debater_b", state) == (client, "opp")
    assert env.get_agent_binding("judge", state) == (None, None)


def test_bindings_fn_called_once_per_slot_not_per_agent():
    """Simultaneous slots resolve once per slot, even with N agents.

    Implementation detail test, but worth it: N-calls would silently
    quadratic-up a dynamic bindings fn's cost, and on an external-service
    fn (e.g. one that consults a checkpoint registry) it's observable
    slowdown."""
    calls = {"n": 0}
    opp, judge_c = MagicMock(), MagicMock()

    def bindings_fn(state):
        calls["n"] += 1
        return {
            "debater_a": (None, None),
            "debater_b": (opp, "opp"),
            "judge": (judge_c, "j"),
        }

    env = _build(bindings_fn=bindings_fn)
    init_calls = calls["n"]  # init probe consumed some calls
    assert init_calls == 1  # probe hits fn once

    state = State()
    state["input"] = {"info": {"learner_seat": "debater_a"}}

    # The runtime path is _get_bindings -> bindings_fn(state) once per
    # slot. Exercise the underlying helper directly, as production
    # _run_simultaneous_slot does.
    env._get_bindings(state)
    assert calls["n"] == init_calls + 1

    env._get_bindings(state)
    assert calls["n"] == init_calls + 2


# ---------------------------------------------------------------------------
# Probe state for subclasses
# ---------------------------------------------------------------------------


def test_debate_probe_seeds_learner_seat():
    """DebateEnv's _build_probe_state fills info.learner_seat so debate
    bindings fns that read it don't KeyError at init."""

    def bindings_fn(state):
        # KeyErrors if learner_seat isn't in info
        seat = state["info"]["learner_seat"]
        opposite = "debater_b" if seat == "debater_a" else "debater_a"
        return {
            seat: (None, None),
            opposite: (MagicMock(), "opp"),
            "judge": (MagicMock(), "j"),
        }

    # If the probe didn't seed learner_seat, this would fail with KeyError.
    _build(bindings_fn=bindings_fn)
