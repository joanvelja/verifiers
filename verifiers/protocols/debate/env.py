"""DebateEnv: thin MultiAgentEnv subclass for debate protocols.

Keeps only debate-specific concerns:
  * DebatePrompts prompt pack (keyed by ``member_id``)
  * XML field extraction from the public channel
  * Opponent visibility derived from ``think_visibility``
  * Opponent attribution wrapping
  * Final debate transcript rendered into ``state['completion']``

Generic machinery (rollout loop, atomic simultaneous commit, stop
conditions, kernel threading, member-scoped cache, trajectory step append) is
inherited from :class:`MultiAgentEnv`.
"""

import logging
from typing import Any, cast

from verifiers.protocols.debate.channels import merge_provider_reasoning, parse_channels
from verifiers.protocols.debate.parsing import extract_fields
from verifiers.protocols.debate.prompts import (
    DebatePrompts,
    build_context,
    resolve_prompts,
)
from verifiers.envs.multi_agent_kernel import (
    ContentChannels,
    KernelState,
    SlotProgram,
    StaticSchedule,
    TurnSlot,
    Utterance,
    causal_transcript_view,
)
from verifiers.protocols.debate.rubric import question_from_state
from verifiers.envs.multi_agent_env import MultiAgentEnv, VisibilityMode
from verifiers.types import (
    AssistantMessage,
    Messages,
    Message,
    Response,
    State,
    SystemMessage,
    UserMessage,
)

_log = logging.getLogger(__name__)
_PROMPT_BODY_CACHE_KEY = "_debate_prompt_body_cache"


class DebateEnv(MultiAgentEnv):
    """Debate-specific MultiAgentEnv.

    Subclasses :class:`MultiAgentEnv` and specialises four things:
      1. :meth:`build_prompt`  -- render via DebatePrompts
      2. :meth:`extract_fields` -- XML field parsing from public channel
      3. :meth:`visibility_policy` -- derived from ``think_visibility``
      4. :meth:`render_completion` -- flatten trajectory into messages

    ``member_id`` is the single participant label. Prompt packs key
    ``system``/``user``/``fields``/``think_visibility`` by ``member_id``
    directly (e.g. ``debater_a``, ``debater_b``, ``judge``).
    """

    def __init__(
        self,
        schedule: SlotProgram,
        prompts: DebatePrompts,
        members: list[str],
        **kwargs: Any,
    ) -> None:
        super().__init__(
            schedule=schedule,
            members=members,
            **kwargs,
        )

        # Cross-check 1: env.members must equal rubric.members exactly.
        # Silent drift desyncs round_index (env) from per-member reward
        # attribution (rubric) and yields plausible-but-wrong training
        # signal. Rubric contract (MultiAgentRubric) guarantees the attr.
        rubric_members = getattr(self.rubric, "members", None)
        if rubric_members is None:
            raise TypeError(
                "DebateEnv requires a rubric with a members attribute so rewards "
                "can be attributed per member"
            )
        rubric_members = list(rubric_members)
        if rubric_members != list(members):
            raise ValueError(
                f"DebateEnv.members != rubric.members\n"
                f"  env    : {list(members)}\n"
                f"  rubric : {rubric_members}\n"
                f"Both must be identical (same ids, same order) -- "
                f"round_index and reward attribution key off them."
            )

        # Cross-check 2: for StaticSchedule, unique slot agents must equal
        # the declared members set. Dynamic SlotProgram implementations
        # are exempt (agent set may be data-dependent).
        if isinstance(schedule, StaticSchedule):
            slot_agents: set[str] = set()
            for slot in schedule._slots:
                slot_agents.update(slot.agents)
            member_set = set(members)
            if slot_agents != member_set:
                raise ValueError(
                    f"DebateEnv.members != unique agents in StaticSchedule\n"
                    f"  members          : {sorted(member_set)}\n"
                    f"  schedule agents  : {sorted(slot_agents)}\n"
                    f"  in members only  : {sorted(member_set - slot_agents)}\n"
                    f"  in schedule only : {sorted(slot_agents - member_set)}"
                )

        # Cross-check 3: schedule × prompts coverage. Every (member_id,
        # phase) in the schedule needs a renderable prompt. Two failure
        # modes if not — the LOUD one (system missing → KeyError on the
        # first turn, ~minutes of wasted compute) and the SILENT one
        # (question/effective instruction missing → empty prompt, model
        # gets a malformed turn with no upstream signal). Catch both at
        # init so the operator hits the error before a rollout completes.
        # Dynamic SlotProgram implementations are exempt — their agent ×
        # phase set isn't enumerable at init time.
        if isinstance(schedule, StaticSchedule):
            self._validate_prompts_cover_schedule(prompts, schedule)

        self.prompts = prompts

    @staticmethod
    def _validate_prompts_cover_schedule(
        prompts: DebatePrompts, schedule: StaticSchedule
    ) -> None:
        """Fail loud at init if the prompts pack doesn't cover every
        (member_id, phase) pair the schedule will hit. Lookups mirror
        DebatePrompts.render_{system,question,instruction} so the check
        rejects exactly what would silently break or KeyError later.
        """
        missing_system: set[str] = set()
        missing_question: set[str] = set()
        # (member_id, phase) pairs with no effective instruction source:
        # no user template/default, no think instruction, no field instruction.
        missing_instruction: set[tuple[str, str]] = set()
        for slot in schedule._slots:
            for member_id in slot.agents:
                phase = slot.phase
                if member_id not in prompts.system:
                    missing_system.add(member_id)
                if member_id not in prompts.question:
                    missing_question.add(member_id)
                user_block = prompts.user.get(member_id, {})
                has_user_instruction = phase in user_block or "default" in user_block
                has_think_instruction = (
                    prompts.think_visibility.get(member_id, "disabled") != "disabled"
                )
                has_field_instruction = bool(prompts.get_field_specs(member_id, phase))
                if not (
                    has_user_instruction
                    or has_think_instruction
                    or has_field_instruction
                ):
                    missing_instruction.add((member_id, phase))
        if not (missing_system or missing_question or missing_instruction):
            return

        lines = [
            f"DebateEnv prompts pack does not cover the schedule "
            f"(pack source: {prompts.source_ref!r}).",
        ]
        if missing_system:
            lines.append(
                f"  system: missing for member(s) {sorted(missing_system)} "
                f"— DebatePrompts.render_system would KeyError on the "
                f"first turn for any of these"
            )
        if missing_question:
            lines.append(
                f"  question: missing for member(s) "
                f"{sorted(missing_question)} — turn would render with "
                f"NO question (silent failure: model gets prompt with "
                f"system + transcript only)"
            )
        if missing_instruction:
            by_member: dict[str, list[str]] = {}
            for m, p in missing_instruction:
                by_member.setdefault(m, []).append(p)
            for m, phases in sorted(by_member.items()):
                lines.append(
                    f"  instruction[{m!r}]: missing phase(s) "
                    f"{sorted(set(phases))} — no user phase template, no "
                    f"'default' fallback, no think instruction, and no "
                    f"field instruction (silent failure: model has to infer "
                    f"the task from system+transcript)"
                )
        lines.append(f"  pack has system keys: {sorted(prompts.system)}")
        lines.append(f"  pack has user keys:   {sorted(prompts.user)}")
        lines.append(
            "Add the missing templates to the YAML pack, or set a "
            "'default' phase block under the relevant user[member_id] "
            "or add an equivalent think/field instruction source."
        )
        raise ValueError("\n".join(lines))

    def _member_round_count(self, member_id: str) -> int:
        """Count schedule slots where ``member_id`` participates as agent.

        Correct under:
          - sequential schedules (one agent per slot) — gives slots/N.
          - simultaneous schedules (N agents per slot) — gives per-member
            slot count, not the full slot count.
          - judge-inclusive schedules where a member (e.g. judge) appears
            in fewer slots than debaters.
        Falls back to 1 for dynamic (non-iterable) schedules; construction-
        time cross-check rejects mismatch between members and schedule
        agent set for the static case.
        """
        slots = getattr(self.schedule, "_slots", None)
        if slots is None:
            return 1
        count = sum(1 for slot in slots if member_id in slot.agents)
        return max(count, 1)

    # -- visibility policy ---------------------------------------------------

    def _render_opponent_message(self, utt: Utterance, viewer_id: str) -> UserMessage:
        """Render one opponent utterance as a user message for ``viewer_id``.

        Selects raw vs. public channel per visibility_policy and wraps with
        speaker attribution.

        Quarantined utterances (``parse_error`` set, including empty-public
        commits from reasoning-mode budget exhaustion) render as a single
        line that names the missed phase explicitly. A bare ``(no response)``
        marker proved ambiguous in practice: reasoning-mode opponents spent
        thousands of tokens of their own reasoning budget asking "what does
        this mean in context?" before falling back to evaluating the
        previous turn. Naming the missed phase lets the model skip the
        meta-reasoning and proceed.
        """
        if utt.parse_error is not None:
            return UserMessage(
                content=(
                    f"[{utt.member_id}] (no {utt.phase} produced — "
                    f"opponent did not emit a visible response this turn)"
                )
            )
        vis = self.visibility_policy(utt, viewer_id)
        content = utt.raw_content if vis == "full" else utt.public_channel
        content = self.prompts.wrap_opponent(
            utt.phase,
            content,
            member_id=utt.member_id,
            viewer_id=viewer_id,
        )
        return UserMessage(content=content)

    def visibility_policy(self, utt: Utterance, viewer_id: str) -> VisibilityMode:
        if utt.member_id == viewer_id:
            return "full"
        vis = self.prompts.think_visibility.get(utt.member_id, "disabled")
        if vis == "open":
            return "full"
        if vis == "visible_to_judge" and viewer_id == "judge":
            return "full"
        return "public_only"

    def split_response_channels(
        self, response: Response, member_id: str, slot: TurnSlot
    ) -> tuple[str, ContentChannels]:
        raw, _ = super().split_response_channels(response, member_id, slot)
        raw = merge_provider_reasoning(raw, response, self.prompts.think_tag)
        return raw, parse_channels(raw, self.prompts.think_tag)

    # -- prompt construction -------------------------------------------------

    async def build_prompt(
        self, state: State, member_id: str, slot: TurnSlot
    ) -> Messages:
        """Render the prompt for ``member_id`` at ``slot``.

        Monotonic-extension invariant (base-class contract): for a fixed
        member, the slot-N+1 prompt is equal byte-for-byte to the slot-N
        prompt on its leading messages and only adds a suffix. To achieve
        this, each own-turn in the transcript is rendered as the pair
        ``[instruction_that_preceded_it, assistant=raw_content]`` -- i.e.
        we re-render the instruction for that turn's phase/round_index.
        Opponent turns are rendered as wrapped user messages. The
        current turn's instruction + optional assistant-prefill sit at
        the tail. No contiguous-user-message consolidation (that would
        split one message into two across boundaries that shift when
        history grows).
        """
        kernel_state: KernelState = state["_kernel"]
        question = question_from_state(state)
        num_rounds = self._member_round_count(member_id)
        body, current_round = self._render_cached_transcript_body(
            state,
            member_id,
            kernel_state.transcript,
            num_rounds,
            question,
        )
        ctx_current = self._build_prompt_context(
            state, member_id, slot.phase, current_round, num_rounds, question
        )

        msgs: list[Message] = [
            SystemMessage(content=self.prompts.render_system(member_id, ctx_current)),
        ]
        q_text = self.prompts.render_question(member_id, ctx_current)
        if q_text:
            msgs.append(UserMessage(content=q_text))

        msgs.extend(body)
        msgs.extend(self._render_current_suffix(member_id, slot, ctx_current))
        return msgs

    def _render_cached_transcript_body(
        self,
        state: State,
        member_id: str,
        transcript: tuple[Utterance, ...],
        num_rounds: int,
        question: str,
    ) -> tuple[list[Message], int]:
        """Render committed transcript messages incrementally for ``member_id``.

        The head (system/question) and current suffix are intentionally
        rendered fresh by ``build_prompt``. Only the committed transcript body
        is cached; it depends on immutable utterances plus stable task fields,
        and is the part that otherwise caused every prompt build to rescan the
        whole episode.
        """
        raw_cache_root = state.get(_PROMPT_BODY_CACHE_KEY)
        if isinstance(raw_cache_root, dict):
            cache_root = cast(dict[str, dict[str, object]], raw_cache_root)
        else:
            cache_root = {}
            state[_PROMPT_BODY_CACHE_KEY] = cache_root

        raw_cache = cache_root.get(member_id)
        answer = state["answer"]
        raw_cache_processed = raw_cache.get("processed", 0) if raw_cache else 0
        cache_processed = (
            raw_cache_processed
            if isinstance(raw_cache_processed, int)
            else len(transcript) + 1
        )
        if (
            raw_cache is None
            or raw_cache.get("question") != question
            or raw_cache.get("answer") != answer
            or raw_cache.get("num_rounds") != num_rounds
            or cache_processed > len(transcript)
        ):
            cache: dict[str, object] = {
                "question": question,
                "answer": answer,
                "num_rounds": num_rounds,
                "processed": 0,
                "own_round": 0,
                "body": [],
                "source_transcript": (),
            }
            cache_root[member_id] = cache
        else:
            cache = raw_cache

        raw_body = cache.get("body")
        body = cast(list[Message], raw_body) if isinstance(raw_body, list) else []
        raw_own_round = cache.get("own_round", 0)
        own_round = raw_own_round if isinstance(raw_own_round, int) else 0
        raw_processed = cache.get("processed", 0)
        processed = raw_processed if isinstance(raw_processed, int) else 0
        raw_source_transcript = cache.get("source_transcript", ())
        source_transcript = (
            cast(tuple[Utterance, ...], raw_source_transcript)
            if isinstance(raw_source_transcript, tuple)
            else ()
        )
        # Normal kernel traffic is append-only; direct build_prompt callers may
        # still swap in a replacement KernelState. Keep the cache honest by
        # validating the already-rendered prefix before rendering only the
        # suffix. This comparison is much cheaper than re-running the Jinja
        # render path for every historical utterance.
        if (
            processed
            and source_transcript is not transcript
            and transcript[:processed] != source_transcript[:processed]
        ):
            body = []
            own_round = 0
            processed = 0

        for utt in causal_transcript_view(transcript, member_id, start=processed):
            if utt.member_id == member_id:
                body.extend(
                    self._render_own_turn(
                        utt, member_id, own_round, num_rounds, question, state
                    )
                )
                own_round += 1
            else:
                body.append(self._render_opponent_message(utt, member_id))
            processed += 1

        cache["processed"] = processed
        cache["own_round"] = own_round
        cache["body"] = body
        cache["source_transcript"] = transcript
        return list(body), own_round

    # -- build_prompt helpers ------------------------------------------------

    def _build_prompt_context(
        self,
        state: State,
        member_id: str,
        phase: str,
        round_index: int,
        num_rounds: int,
        question: str,
    ) -> dict[str, Any]:
        """Assemble the Jinja context dict for one (member, phase, round).

        ``num_rounds`` is per-member (see ``_member_round_count``) so
        is_first_round / is_last_round flags stay correct under
        simultaneous and judge-inclusive schedules.
        """
        return build_context(
            task_prompt=question,
            viewer_id=member_id,
            phase=phase,
            round_index=round_index,
            num_rounds=num_rounds,
            answer=state["answer"],
        )

    def _render_own_turn(
        self,
        utt: Utterance,
        member_id: str,
        round_index: int,
        num_rounds: int,
        question: str,
        state: State,
    ) -> list[Message]:
        """Render one own-turn utterance as ``[instruction, assistant=raw]``.

        Re-renders the instruction that preceded ``utt`` using the past
        turn's own (phase, round_index), then emits the verbatim
        ``raw_content`` as the assistant message. ``round_index`` is the
        positional own-turn counter (N-th own commit = round N),
        independent of slot_id so sparse / semantic slot_ids don't
        produce nonsensical round labels.

        Quarantined own-turns are skipped entirely. There's no useful
        assistant content to anchor on (parse failure or empty-public
        commit), and replaying a lone ``<thinking>`` block as the
        previous assistant message confuses reasoning models into
        thinking they're still in that earlier phase. KV-cache
        monotonic-extension is unaffected — a quarantined commit
        added no usable prefix tokens to invalidate.
        """
        if utt.parse_error is not None:
            return []
        past_ctx = self._build_prompt_context(
            state, member_id, utt.phase, round_index, num_rounds, question
        )
        msgs: list[Message] = []
        past_instr = self.prompts.render_instruction(member_id, utt.phase, past_ctx)
        if past_instr:
            msgs.append(UserMessage(content=past_instr))
        msgs.append(AssistantMessage(content=utt.raw_content))
        return msgs

    def _render_current_suffix(
        self, member_id: str, slot: TurnSlot, ctx_current: dict[str, Any]
    ) -> list[Message]:
        """Render the tail of the prompt: current instruction + optional prefill."""
        msgs: list[Message] = []
        instruction = self.prompts.render_instruction(
            member_id, slot.phase, ctx_current
        )
        if instruction:
            msgs.append(UserMessage(content=instruction))
        prefill = self.prompts.render_prefill(member_id, slot.phase, ctx_current)
        if prefill:
            msgs.append(AssistantMessage(content=prefill))
        return msgs

    # -- field extraction ----------------------------------------------------

    async def extract_fields(
        self, public_channel: str, member_id: str, slot: TurnSlot
    ) -> dict[str, Any] | None:
        """Extract XML fields from the public channel, per member+phase specs.

        Parse ambiguity (duplicate schema tags, raised as ValueError) is a
        per-step signal: we log and return None so the step is still
        appended and the rubric's failed_members path fires
        ``extraction_failed/{mid}=1.0``. Terminating would discard other
        members' valid commits.
        """
        specs = self.prompts.get_field_specs(member_id, slot.phase)
        if not specs:
            return None
        try:
            return extract_fields(public_channel, specs)
        except ValueError as exc:
            _log.warning(
                "field extraction failed for member=%s phase=%s: %s",
                member_id,
                slot.phase,
                exc,
            )
            return None

    # -- completion rendering ------------------------------------------------

    async def render_completion(self, state: State) -> None:
        state["completion"] = [
            msg for step in state["trajectory"] for msg in step["completion"]
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load_environment(**kwargs: Any) -> DebateEnv:
    """Construct a DebateEnv from a prompt pack and a static schedule.

    The factory is a pure dispatcher: it converts stringly-typed kwargs
    into typed component objects (schedule, prompts, judge client) and
    wires them into a rubric + env. It performs NO validation — each
    component validates its own preconditions at construction:

      * ``DebatePrompts.__post_init__``  — pack invariants (verdict-token
        collisions, non-empty judge templates)
      * ``DebateRubric.__init__``        — judge-client gate, client type
      * ``DebateEnv.__init__``           — cross-component coherence
        (env.members == rubric.members, schedule agents ⊆ members)

    Usage::

        env = load_environment(
            schedule_slots=[
                {"slot_id": 0, "agents": ["debater_a"], "phase": "propose"},
                {"slot_id": 1, "agents": ["debater_b"], "phase": "propose"},
                {"slot_id": 2, "agents": ["judge"],     "phase": "final"},
            ],
            members=["debater_a", "debater_b", "judge"],
            prompts_ref="selfplay",
            eval_dataset=my_dataset,
        )

    Required: schedule_slots, members.
    Prompt source (exactly one): ``prompts_ref`` (str, registry lookup)
    or ``prompts`` (already-built DebatePrompts).
    Optional: truth_member, judge_client, judge_model, dataset/eval_dataset.
    """
    from verifiers.protocols.debate.rubric import DebateRubric

    schedule = StaticSchedule(
        tuple(
            TurnSlot(
                slot_id=s["slot_id"],
                agents=tuple(s["agents"]),
                phase=s.get("phase", ""),
            )
            for s in kwargs.pop("schedule_slots")
        )
    )
    prompts = _resolve_prompts_arg(
        kwargs.pop("prompts_ref", None),
        kwargs.pop("prompts", None),
    )
    rubric = DebateRubric(
        truth_member=kwargs.pop("truth_member", None),
        members=kwargs["members"],  # don't pop — env needs it too
        prompts=prompts,
        judge_client=kwargs.pop("judge_client", None),
        judge_model=kwargs.pop("judge_model", "gpt-4.1-nano"),
    )
    return DebateEnv(
        schedule=schedule,
        prompts=prompts,
        members=kwargs.pop("members"),
        rubric=rubric,
        **kwargs,
    )


def _resolve_prompts_arg(
    prompts_ref: str | None, prompts: DebatePrompts | None
) -> DebatePrompts:
    """Return a DebatePrompts from whichever source the caller provided.

    Exactly one of ``prompts_ref`` (registry lookup) or ``prompts`` (typed
    object) must be given. Pure conversion; ``DebatePrompts.__post_init__``
    runs the intrinsic pack validation.
    """
    if prompts is not None and prompts_ref is not None:
        raise ValueError("Provide exactly one of 'prompts_ref' or 'prompts'")
    if prompts is not None:
        return prompts
    if prompts_ref is not None:
        return resolve_prompts(prompts_ref)
    raise ValueError("Must provide either 'prompts_ref' or 'prompts'")
