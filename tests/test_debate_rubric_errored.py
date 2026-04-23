"""Tests for ``DebateRubric.build_errored_marscore``.

Guards two invariants:
  1. Valid trajectories have parse_error_count preserved as the docstring
     promises.
  2. A trajectory step missing ``extras['member_id']`` is a schema violation
     that propagates (previously silently zeroed via
     ``except ValueError: steps_by_mid = {}``).
"""

from __future__ import annotations

from typing import Any

import pytest

from verifiers.envs.debate import resolve_prompts
from verifiers.envs.debate_rubric import DebateRubric


@pytest.fixture
def rubric() -> DebateRubric:
    return DebateRubric(
        truth_member="debater_a",
        members=["debater_a", "debater_b", "judge"],
        prompts=resolve_prompts("default"),
    )


def _valid_trajectory() -> list[dict[str, Any]]:
    """Trajectory where debater_a has one parse_error step."""
    return [
        {"extras": {"member_id": "debater_a", "parse_error": "bad-tag"}},
        {"extras": {"member_id": "debater_a"}},
        {"extras": {"member_id": "debater_b"}},
        {"extras": {"member_id": "judge"}},
    ]


def _malformed_trajectory() -> list[dict[str, Any]]:
    """Trajectory missing member_id on one step — triggers split_by_member."""
    return [
        {"extras": {"member_id": "debater_a", "parse_error": "bad-tag"}},
        {"extras": {"some_other_key": "garbage"}},
        {"extras": {"member_id": "debater_b"}},
        {"extras": {"member_id": "judge"}},
    ]


def test_preserves_parse_error_count_on_valid_trajectory(
    rubric: DebateRubric,
) -> None:
    state = {"trajectory": _valid_trajectory()}
    score = rubric.build_errored_marscore(
        state, error_type="synthetic", error_phase="test"
    )
    by_id = {m.member_id: m for m in score.members}
    assert by_id["debater_a"].parse_error_count == 1
    assert by_id["debater_b"].parse_error_count == 0
    assert by_id["judge"].parse_error_count == 0
    assert score.episode_error == {
        "error_type": "synthetic",
        "error_phase": "test",
    }


def test_propagates_value_error_on_missing_member_id(rubric: DebateRubric) -> None:
    state = {"trajectory": _malformed_trajectory()}
    with pytest.raises(ValueError, match="member_id"):
        rubric.build_errored_marscore(
            state, error_type="synthetic", error_phase="test"
        )
