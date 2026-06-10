"""Split a multi-agent RolloutOutput into per-member training rollouts."""

from collections.abc import Mapping, Sequence
from typing import Any

from .types import MARScore, MemberRollout, RolloutOutput


def _member_generation(member_id: str, steps: Sequence[Any]) -> dict[str, Any] | None:
    generations: list[dict[str, Any]] = []
    for step in steps:
        extras = step.get("extras")
        if not isinstance(extras, Mapping):
            continue
        generation = extras.get("generation")
        if generation is None:
            continue
        if not isinstance(generation, Mapping):
            raise TypeError(
                f"Trajectory generation metadata for {member_id!r} must be a mapping"
            )
        generations.append(dict(generation))
    if not generations:
        return None
    first = generations[0]
    for generation in generations[1:]:
        if generation != first:
            raise ValueError(
                f"member_id={member_id!r} has inconsistent generation targets across "
                "trajectory steps"
            )
    return first


def rollout_to_member_rollouts(
    output: RolloutOutput | Mapping[str, Any],
) -> list[MemberRollout]:
    """Project one episode-level rollout into one rollout per member.

    Inputs are expected to be a ``RolloutOutput`` produced by
    ``state_to_output``, which guarantees ``task``, ``example_id``,
    ``sampling_args``, ``trajectory_id`` and ``mar_score``. Missing any
    of these is a contract violation upstream — we ``KeyError`` rather
    than silently substituting defaults that would corrupt training
    identity (RAE baselines key on ``(env_name, example_id)``; the member
    enters via the antithetic sign, not the baseline key).

    ``trajectory`` is ALSO required and must be present as a key on the
    output. ``state_to_output`` only includes it when the caller passes
    ``state_columns=["trajectory", ...]``; absent trajectory would cause
    ``split_by_member`` to silently return ``{}`` and the bridge to emit
    empty-trajectory member rollouts — silently discarding all
    token-level training data. Fail loud instead.

    Trajectory member ids must match ``MARScore.members`` exactly: any
    extra ``extras['member_id']`` would be silently dropped by the
    per-member projection, masking a rubric/env wiring bug.
    """
    # Lazy import: verifiers/__init__.py loads this module before Parser
    # is bound, and MultiAgentRubric transitively pulls Rubric which
    # resolves vf.Parser at class-body evaluation time.
    from .rubrics.multi_agent_rubric import MultiAgentRubric

    mar_raw = output["mar_score"]
    mar = mar_raw if isinstance(mar_raw, MARScore) else MARScore.model_validate(mar_raw)

    task = output["task"]
    example_id = output["example_id"]
    sampling_args = output["sampling_args"]
    episode_id = output["trajectory_id"]
    rollout_error = output.get("error")

    if "trajectory" not in output:
        raise KeyError(
            "trajectory missing from RolloutOutput — the multi-agent "
            "bridge requires per-step trajectory data. Add 'trajectory' "
            "to state_columns when saving outputs for training (e.g. "
            "``--state-columns trajectory``)."
        )

    steps_by_member = MultiAgentRubric.split_by_member(output)
    expected_members = {m.member_id for m in mar.members}
    extra_members = set(steps_by_member) - expected_members
    if extra_members:
        raise ValueError(
            f"trajectory contains member_id(s) {sorted(extra_members)} not in "
            f"MARScore.members={sorted(expected_members)}; rubric/env "
            "out-of-sync — silently dropping these steps would corrupt "
            "training data"
        )
    missing_members = expected_members - set(steps_by_member)
    if missing_members and rollout_error is None:
        raise ValueError(
            f"MARScore member_id(s) {sorted(missing_members)} have no trajectory "
            "steps, but the rollout did not record an error; emitting empty "
            "member trajectories would silently corrupt training data"
        )

    member_rollouts: list[MemberRollout] = []
    for member in mar.members:
        member_steps = steps_by_member.get(member.member_id, [])
        generation = _member_generation(member.member_id, member_steps)
        member_sampling_args = (
            dict(generation.get("sampling_args") or {})
            if generation is not None
            else dict(sampling_args)
        )
        member_sampling_args.setdefault("temperature", 1.0)
        member_rollout = MemberRollout(
            example_id=example_id,
            task=task,
            trajectory=member_steps,
            sampling_args=member_sampling_args,
            error=rollout_error,
            reward=member.reward,
            episode_id=episode_id,
            member_id=member.member_id,
        )
        if generation is not None:
            member_rollout["generation"] = generation
            model = generation.get("model")
            if isinstance(model, str):
                member_rollout["model"] = model
        member_rollouts.append(member_rollout)
    return member_rollouts
