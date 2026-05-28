"""Split a multi-agent RolloutOutput into per-member training rollouts."""

from typing import Any, Mapping

from .types import MARScore, MemberRollout, RolloutOutput


def rollout_to_member_rollouts(
    output: RolloutOutput | Mapping[str, Any],
) -> list[MemberRollout]:
    """Project one episode-level rollout into one rollout per member.

    The bridge fails loud on missing canonical fields because downstream
    trainers key baselines and samples by episode/member identity. Filling
    defaults here would silently corrupt training data.
    """
    from .rubrics.multi_agent_rubric import MultiAgentRubric

    mar_raw = output["mar_score"]
    mar = mar_raw if isinstance(mar_raw, MARScore) else MARScore.model_validate(mar_raw)

    task = output["task"]
    example_id = output["example_id"]
    sampling_args = output["sampling_args"]
    episode_id = output["trajectory_id"]
    rollout_error = output.get("error")
    tool_defs = output.get("tool_defs")

    if "trajectory" not in output:
        raise KeyError(
            "trajectory missing from RolloutOutput; the multi-agent bridge "
            "requires per-step trajectory data. Add 'trajectory' to "
            "state_columns when saving outputs for training."
        )

    member_sampling_args = dict(sampling_args)
    member_sampling_args.setdefault("temperature", 1.0)

    steps_by_member = MultiAgentRubric.split_by_member(output)
    expected_members = {member.member_id for member in mar.members}
    extra_members = set(steps_by_member) - expected_members
    if extra_members:
        raise ValueError(
            f"trajectory contains member_id(s) {sorted(extra_members)} not in "
            f"MARScore.members={sorted(expected_members)}; rubric/env "
            "out-of-sync"
        )
    missing_members = expected_members - set(steps_by_member)
    if missing_members and rollout_error is None:
        raise ValueError(
            f"MARScore member_id(s) {sorted(missing_members)} have no trajectory "
            "steps, but the rollout did not record an error"
        )

    member_rollouts = [
        MemberRollout(
            example_id=example_id,
            task=task,
            trajectory=steps_by_member.get(member.member_id, []),
            sampling_args=member_sampling_args,
            error=rollout_error,
            reward=member.reward,
            episode_id=episode_id,
            member_id=member.member_id,
        )
        for member in mar.members
    ]
    if tool_defs is not None:
        for member_rollout in member_rollouts:
            member_rollout["tool_defs"] = tool_defs
    return member_rollouts
