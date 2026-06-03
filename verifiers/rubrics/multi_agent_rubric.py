"""Base rubric contract for member-attributed multi-agent scoring."""

import asyncio
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Mapping

import verifiers as vf
from verifiers.rubrics.rubric import Rubric
from verifiers.types import MARScore, MemberScore, State, TrajectoryStep


class MultiAgentRubric(Rubric):
    """Base class for multi-agent scoring.

    Subclasses implement ``build_marscore``. The base class owns the
    error boundary and writes ``state["mar_score"]`` for serialization.
    """

    members: list[str]

    def __init__(self, members: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if members is not None:
            self.members = members
        self._validate_members()

    def _validate_members(self) -> None:
        members = getattr(self, "members", None)
        if not members:
            raise ValueError("MultiAgentRubric.members must be non-empty")
        if len(members) != len(set(members)):
            raise ValueError(f"Duplicate member_id in MultiAgentRubric: {members}")

    @abstractmethod
    async def build_marscore(self, state: State) -> MARScore: ...

    @staticmethod
    def split_by_member(
        state_or_output: Mapping[str, Any],
    ) -> dict[str, list[TrajectoryStep]]:
        """Group trajectory steps by ``extras['member_id']``."""
        output: dict[str, list[TrajectoryStep]] = defaultdict(list)
        for step in state_or_output["trajectory"]:
            member_id = step["extras"].get("member_id")
            if member_id is None:
                raise ValueError(
                    f"TrajectoryStep missing extras['member_id']: {step!r}"
                )
            output[member_id].append(step)
        return dict(output)

    def build_errored_marscore(
        self, state: State, *, error_type: str, error_phase: str
    ) -> MARScore:
        return MARScore(
            members=[
                MemberScore(member_id=member_id, reward=0.0)
                for member_id in self.members
            ],
            episode_scalar=0.0,
            episode_metrics={"errored_rollout": 1.0},
            episode_error={"error_type": error_type, "error_phase": error_phase},
        )

    async def score_rollout(self, state: State) -> None:
        if state.get("prompt_too_long", False):
            # Record the error so the fan-out bridge treats this as a properly
            # errored rollout (drop the member with no trajectory step) rather
            # than raising on the inconsistency and aborting the whole step.
            state["error"] = vf.OverlongPromptError(
                "prompt exceeded the model's max length before scoring (prompt_too_long)"
            )
            state["mar_score"] = self.build_errored_marscore(
                state, error_type="prompt_too_long", error_phase="rollout"
            )
            return

        existing_error = state.get("error")
        if existing_error is not None:
            state["mar_score"] = self.build_errored_marscore(
                state,
                error_type=type(existing_error).__name__,
                error_phase="rollout",
            )
            return

        try:
            state["mar_score"] = await self.build_marscore(state)
        except vf.Error as error:
            state["error"] = error
            state["mar_score"] = self.build_errored_marscore(
                state,
                error_type=type(error).__name__,
                error_phase="scoring",
            )

    async def score_group(self, states: list[State]) -> None:
        await asyncio.gather(*(self.score_rollout(state) for state in states))
