"""Debate rubric: ±1 zero-sum member reward + diagnostic metrics.

Member reward (single source of training signal):
    winner member → +1, loser member → −1, judge → 0, no decision → 0.

Episode scalar:
    inert (0.0) unless a pack explicitly declares ``truth_member``. When
    declared, the scalar is 1 iff the judge picked that member. Symmetric
    debate packs should leave ``truth_member=None`` so ground truth stays
    diagnostic-only.

Diagnostics (weight-0 telemetry; never feed reward):
    per-member: turns, parse_error_count, num_commits, num_unique_commits,
                accuracy, extraction_failed, initial_correct, final_correct,
                grader_error
    episode:    agreement, any_answer_member_correct, all_answer_members_correct,
                any_debater_correct, all_debaters_correct,
                judge_selected_correct, truth_member_correct, truth_member_won,
                winner, matcher_error
"""

import asyncio
from typing import Any

import verifiers as vf
from verifiers.clients import Client
from verifiers.protocols.debate.fields import EnumScoring, FieldSpec, classify_enum
from verifiers.protocols.debate.prompts import (
    DebatePrompts,
)
from verifiers.parsers.parser import Parser
from verifiers.rubrics.judge_rubric import JudgeRubric
from verifiers.rubrics.multi_agent_rubric import MultiAgentRubric
from verifiers.types import (
    AssistantMessage,
    MARScore,
    MemberScore,
    State,
    TrajectoryStep,
    UserMessage,
)
from verifiers.utils.judge_prompts import JudgeTemplate, normalize_verdict_token

# gpt-5.x are REASONING models: reasoning tokens count against
# max_completion_tokens. A 256 ceiling truncates mid-reasoning on hard grading
# instances (observed reasoning 187-256+ tokens) -> finish_reason="length",
# EMPTY content -> EmptyModelResponseError aborts the rollout. The verdict itself
# is one word; the budget exists for the reasoning, so give it real headroom
# (you pay for tokens actually used, not the ceiling). temperature MUST be the
# default 1.0 (gpt-5.x reject any other value).
JUDGE_SAMPLING_ARGS = {
    "temperature": 1.0,
    "max_completion_tokens": 2048,
    "reasoning_effort": "medium",
}


# -- Module-local helpers (pure, testable without a rubric instance) ------


def question_from_state(state: State) -> str:
    """Return the first user message's content from ``state['prompt']``,
    tolerating both dict and pydantic message shapes."""
    for msg in state.get("prompt") or []:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
        content = (
            msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        )
        if role == "user" and content:
            return str(content)
    return ""


def winning_member(trajectory: list[TrajectoryStep]) -> str | None:
    """Last judge step's decision, or None if absent / malformed.

    Reverse-walk and break on the first judge step — whether or not it
    carries a 'decision'. Falling back to an earlier judge would
    silently install a stale verdict.
    """
    for step in reversed(trajectory):
        extras = step["extras"]
        if extras.get("member_id") != "judge":
            continue
        return (extras.get("fields") or {}).get("decision")
    return None


def zero_sum_reward(member_id: str, winner: str | None) -> float:
    """+1 winner, −1 loser, 0 judge, 0 when no decision, 0 on tie.

    ``winner == "tie"`` is a first-class judge outcome in packs where
    the judge can declare a draw (selfplay.yaml does). Both debaters
    receive 0 — zero-sum holds (0 + 0 = 0) and the RAE baseline update
    is neutral.
    """
    if winner is None or winner == "tie" or member_id == "judge":
        return 0.0
    return 1.0 if member_id == winner else -1.0


def episode_scalar_from_winner(winner: str | None, truth_member: str | None) -> float:
    """Top-level scalar for legacy single-score consumers.

    Symmetric debate has no truth side, so the scalar is deliberately inert.
    Ground-truth correctness is emitted as diagnostics instead of becoming a
    rollout-level reward that orchestration code can threshold on.
    """
    if truth_member is None:
        return 0.0
    return 1.0 if winner == truth_member else 0.0


def member_snapshot(
    member_id: str,
    steps: list[TrajectoryStep],
    prompts: DebatePrompts,
) -> dict[str, Any]:
    """One forward pass. Returns plain dict (no new type — the keys are
    the schema): commits, latest_spec, latest_had_answer, turns,
    parse_errors.

    Semantics:
      - commits:           chronological answer commits.
      - latest_spec:       spec at the phase of the most recent commit.
      - latest_had_answer: whether the member's LAST step had
                           ``extras['fields']['answer']`` — distinguishes
                           "wrong answer" from "latest step unparseable".
      - turns:             total step count for this member.
      - parse_errors:      count of steps with ``extras['parse_error']``.
    """
    commits: list[str] = []
    latest_spec: FieldSpec | None = None
    latest_had_answer = False
    parse_errors = 0
    for step in steps:
        extras = step.get("extras", {})
        if extras.get("parse_error"):
            parse_errors += 1
        fields = extras.get("fields") or {}
        if "answer" in fields:
            commits.append(str(fields["answer"]))
            specs = prompts.get_field_specs(member_id, extras.get("phase", "")) or {}
            latest_spec = specs.get("answer")
            latest_had_answer = True
        else:
            latest_had_answer = False
    return {
        "commits": commits,
        "latest_spec": latest_spec,
        "latest_had_answer": latest_had_answer,
        "turns": len(steps),
        "parse_errors": parse_errors,
    }


def maybe_judge(
    prompts: DebatePrompts,
    kind: str,
    client: Client | None,
    model: str,
) -> JudgeRubric | None:
    """Build a JudgeRubric iff the pack declares the template AND a client
    is available. Otherwise return None — score-time callers raise
    ``vf.Error`` if they actually need the missing piece."""
    tmpl = prompts.judges.get(kind)
    if tmpl is None or client is None:
        return None
    return JudgeRubric(
        parser=Parser(),
        judge_client=client,
        judge_model=model,
        judge_system_prompt=tmpl.system,
        judge_prompt=tmpl.user,
        judge_sampling_args=JUDGE_SAMPLING_ARGS,
        judge_positive_label=tmpl.positive,
        judge_negative_label=tmpl.negative,
    )


def resolve_verdict(
    verdict: str, tmpl: JudgeTemplate, *, kind: str, source_ref: str
) -> bool:
    """Exact-token verdict match. Substring matching ('correct' in
    'incorrect') silently inverts labels — fail loud instead."""
    token = normalize_verdict_token(verdict)
    if token == tmpl.positive:
        return True
    if token == tmpl.negative:
        return False
    raise vf.Error(
        f"{kind} verdict {token!r} not in "
        f"{{{tmpl.positive!r}, {tmpl.negative!r}}} "
        f"(raw={verdict!r}); pack={source_ref}"
    )


# -- Rubric ---------------------------------------------------------------


class DebateRubric(MultiAgentRubric):
    """Debate scoring. Reward from judge outcome only; grader/matcher are
    diagnostic signals that never feed training reward."""

    def __init__(
        self,
        members: list[str],
        prompts: DebatePrompts,
        judge_client: Client | None = None,
        judge_model: str = "gpt-4.1-nano",
        truth_member: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(members=members, **kwargs)
        if truth_member is not None and truth_member not in self.members:
            raise ValueError(
                f"truth_member={truth_member!r} is not in members={self.members!r}"
            )
        if truth_member == "judge":
            raise ValueError("truth_member cannot be 'judge'")
        self.truth_member = truth_member
        self.prompts = prompts
        self.grader = maybe_judge(prompts, "grader", judge_client, judge_model)
        self.matcher = maybe_judge(prompts, "matcher", judge_client, judge_model)
        # Precomputed: does the pack declare an 'answer' field anywhere for
        # this member? Controls whether we attempt accuracy/extraction_failed.
        self.member_declares_answer: dict[str, bool] = {
            mid: any(
                "answer" in (phase_fields or {}) for phase_fields in phases.values()
            )
            for mid, phases in prompts.fields.items()
        }

    async def build_marscore(self, state: State) -> MARScore:
        if "answer" not in state:
            raise KeyError(
                "state missing 'answer' (ground truth) — dataset schema violation"
            )
        target = str(state["answer"])
        question = question_from_state(state)
        steps_by_mid = self.split_by_member(state)
        winner = winning_member(state["trajectory"])
        if winner is None and "judge" in self.prompts.fields:
            raise vf.Error(
                f"Pack {self.prompts.source_ref!r} declares a judge but no "
                "decision was produced (early termination, malformed verdict, "
                "or missing judge step). Refusing to fall back to answer "
                "grading — that would silently generate fake training signal."
            )
        # Validate the judge's decision is an allowed member (minus the
        # judge itself — judges are neutral by convention) or "tie".
        # Without this check a typo / out-of-schema verdict would drop
        # through ``zero_sum_reward`` and both debaters would get −1 each
        # (member_id == winner fails for all), producing fake training
        # reward instead of an errored rollout.
        if winner is not None:
            valid_winners = (set(self.members) - {"judge"}) | {"tie"}
            if winner not in valid_winners:
                raise vf.Error(
                    f"Judge decision {winner!r} not in allowed set "
                    f"{sorted(valid_winners)}; pack={self.prompts.source_ref!r}"
                )

        snaps = {
            mid: member_snapshot(mid, steps_by_mid.get(mid, []), self.prompts)
            for mid in self.members
        }

        per_member_metrics, episode_metrics = await self._emit_diagnostics(
            snaps, target, question, state, winner
        )

        members: list[MemberScore] = []
        # First/final commits per debater feed downstream mind-change
        # analysis; absent when the debater produced no parseable <answer>.
        episode_categorical: dict[str, str | None] = {"winner": winner}
        for mid in self.members:
            m = snaps[mid]
            members.append(
                MemberScore(
                    member_id=mid,
                    reward=zero_sum_reward(mid, winner),
                    parse_error_count=m["parse_errors"],
                    metrics=per_member_metrics[mid],
                )
            )
            if mid != "judge" and m["commits"]:
                episode_categorical[f"first_answer/{mid}"] = m["commits"][0]
                episode_categorical[f"final_answer/{mid}"] = m["commits"][-1]

        return MARScore(
            members=members,
            episode_scalar=episode_scalar_from_winner(winner, self.truth_member),
            episode_metrics=episode_metrics,
            episode_categorical=episode_categorical,
        )

    def build_errored_marscore(
        self, state: State, *, error_type: str, error_phase: str
    ) -> MARScore:
        """Zero-reward MARScore with parse_error_count preserved for debug.

        Propagates ValueError from ``split_by_member`` — a trajectory
        step missing ``extras['member_id']`` is a schema violation that
        should fail loud, not get silently zeroed.
        """
        steps_by_mid = self.split_by_member(state)
        snaps = {
            mid: member_snapshot(mid, steps_by_mid.get(mid, []), self.prompts)
            for mid in self.members
        }
        return MARScore(
            members=[
                MemberScore(
                    member_id=mid,
                    reward=0.0,
                    parse_error_count=snaps[mid]["parse_errors"],
                )
                for mid in self.members
            ],
            episode_scalar=0.0,
            episode_metrics={"errored_rollout": 1.0},
            episode_error={"error_type": error_type, "error_phase": error_phase},
        )

    # -- diagnostics (weight-0; do not feed reward) ----------------------

    async def _emit_diagnostics(
        self,
        snaps: dict[str, dict[str, Any]],
        target: str,
        question: str,
        state: State,
        winner: str | None,
    ) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        """Compute per-member + episode diagnostic metrics in one pass.

        Returns ``(per_member_metrics, episode_metrics)``. All values are
        weight-0 telemetry; none feed reward.

        Per-member phase (``per_member[mid]``):
          turns, num_commits, num_unique_commits, flipped; and — gated on
          non-judge + target + pack declaring 'answer' for this member —
          accuracy, final_correct, extraction_failed, initial_correct.
          ``flipped`` is first != final; differs from num_unique_commits>1
          on return-trips like [A,B,A] (unique=2, flipped=0). Orthogonal
          to correctness — see initial_correct/final_correct for the
          (good, bad) decomposition.

        Episode phase (``episode_metrics``):
          * ``agreement``: matcher verdict on the two debaters' final
            answers; present only when both have committed.
          * correctness metrics: diagnostic-only ground-truth telemetry.
            They never feed member rewards. ``*_answer_member_correct``
            aggregates only members whose pack role declares an answer field;
            legacy ``*_debater_correct`` aliases are emitted only when every
            non-judge debater declares an answer. ``truth_member_*`` metrics
            are present only when the pack explicitly declares a truth side.
          * diagnostic judge failures set ``grader_error``/``matcher_error``
            instead of invalidating judge-derived member rewards.
        """

        async def diagnostic_verdict(
            answer: str,
            target: str,
            spec: FieldSpec | None,
            kind: str,
        ) -> bool | None:
            try:
                return await self.verdict(answer, target, question, spec, kind, state)
            except vf.Error:
                return None

        per_member: dict[str, dict[str, float]] = {}
        episode: dict[str, float] = {}
        answer_members: list[str] = []
        final_correct_by_member: dict[str, float] = {}
        member_commits: dict[str, list[str]] = {}
        grader_jobs: list[tuple[str, str, asyncio.Task[bool | None]]] = []

        for mid in self.members:
            m = snaps[mid]
            dst: dict[str, float] = {"turns": float(m["turns"])}
            per_member[mid] = dst
            if mid == "judge":
                continue
            declares_answer = self.member_declares_answer.get(mid, False)
            if declares_answer:
                answer_members.append(mid)
            seq = m["commits"]
            dst["num_commits"] = float(len(seq))
            dst["num_unique_commits"] = float(len(set(seq)))
            dst["flipped"] = 1.0 if len(seq) >= 2 and seq[0] != seq[-1] else 0.0
            if not target or not declares_answer:
                continue
            if not m["latest_had_answer"]:
                final_correct_by_member[mid] = 0.0
                if m["turns"]:
                    dst["extraction_failed"] = 1.0
                continue
            spec = m["latest_spec"]
            member_commits[mid] = seq
            grader_jobs.append(
                (
                    mid,
                    "final",
                    asyncio.create_task(
                        diagnostic_verdict(seq[-1], target, spec, "grader")
                    ),
                )
            )
            if seq[0] != seq[-1]:
                grader_jobs.append(
                    (
                        mid,
                        "initial",
                        asyncio.create_task(
                            diagnostic_verdict(seq[0], target, spec, "grader")
                        ),
                    )
                )

        debaters = [
            (mid, m) for mid, m in snaps.items() if mid != "judge" and m["commits"]
        ]
        matcher_task: asyncio.Task[bool | None] | None = None
        if len(debaters) >= 2:
            (_, a), (_, b) = debaters[0], debaters[1]
            spec = a["latest_spec"] or b["latest_spec"]
            matcher_task = asyncio.create_task(
                diagnostic_verdict(b["commits"][-1], a["commits"][-1], spec, "matcher")
            )

        diagnostic_tasks = [task for _, _, task in grader_jobs]
        if matcher_task is not None:
            diagnostic_tasks.append(matcher_task)
        if diagnostic_tasks:
            await asyncio.gather(*diagnostic_tasks)

        for mid, field, task in grader_jobs:
            verdict = task.result()
            dst = per_member[mid]
            if verdict is None:
                dst["grader_error"] = 1.0
                continue
            value = float(verdict)
            if field == "final":
                dst["accuracy"] = value
                dst["final_correct"] = value
                final_correct_by_member[mid] = value
                dst["extraction_failed"] = 0.0
            else:
                dst["initial_correct"] = value

        for mid, seq in member_commits.items():
            dst = per_member[mid]
            if seq[0] == seq[-1] and "final_correct" in dst:
                dst["initial_correct"] = dst["final_correct"]

        if answer_members and all(
            mid in final_correct_by_member for mid in answer_members
        ):
            correctness = [
                final_correct_by_member.get(mid, 0.0) for mid in answer_members
            ]
            any_correct = any(value == 1.0 for value in correctness)
            all_correct = float(
                bool(correctness) and all(value == 1.0 for value in correctness)
            )
            episode["any_answer_member_correct"] = float(any_correct)
            episode["all_answer_members_correct"] = all_correct
            debater_members = {mid for mid in self.members if mid != "judge"}
            if set(answer_members) == debater_members:
                episode["any_debater_correct"] = float(any_correct)
                episode["all_debaters_correct"] = all_correct
            if winner in answer_members:
                selected_correct = final_correct_by_member.get(winner, 0.0)
                episode["judge_selected_correct"] = selected_correct
                if any_correct:
                    episode["judge_selected_correct_given_any_correct"] = (
                        selected_correct
                    )
            if self.truth_member in answer_members:
                episode["truth_member_correct"] = final_correct_by_member.get(
                    self.truth_member, 0.0
                )
                episode["truth_member_won"] = float(winner == self.truth_member)

        if matcher_task is not None:
            verdict = matcher_task.result()
            if verdict is None:
                episode["matcher_error"] = 1.0
            else:
                episode["agreement"] = float(verdict)

        return per_member, episode

    async def verdict(
        self,
        answer: str,
        target: str,
        question: str,
        spec: FieldSpec | None,
        kind: str,
        state: State,
    ) -> bool:
        """MCQ short-circuit when spec is EnumScoring; else LLM (grader or
        matcher). Matcher is just an asymmetric grader — A is target, B is
        response — so both routes share this primitive."""
        if spec and isinstance(spec.scoring, EnumScoring):
            c = classify_enum(str(answer), spec.scoring.values)
            t = classify_enum(str(target), spec.scoring.values)
            return c.is_valid and t.is_valid and c.canonical == t.canonical
        judge = self.grader if kind == "grader" else self.matcher
        if judge is None:
            raise vf.Error(
                f"Open-ended {kind} needs a judge template; "
                f"pack={self.prompts.source_ref!r}"
            )
        raw = await judge.judge(
            prompt=[UserMessage(content=question)],
            completion=[AssistantMessage(content=str(answer))],
            answer=str(target),
            state=state,
        )
        return resolve_verdict(
            raw,
            self.prompts.judges[kind],
            kind=kind,
            source_ref=self.prompts.source_ref,
        )
