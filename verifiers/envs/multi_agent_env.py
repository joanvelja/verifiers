"""MultiAgentEnv: generic N-agent rollout loop on the Environment contract.

Abstracts out the agent-agnostic machinery shared by multi-agent
environments (debate, RPS, PD, proposer-solver, ...):

- Slot-scheduled rollout loop (sequential and simultaneous barriers).
- Stop conditions with priority ordering (error > token_limit >
  schedule_exhausted > prompt_too_long).
- Per-member request identity for cache partitioning and runtime routing.
- Atomic simultaneous-slot commit (all commits land or none do).

Subclasses implement only the domain-specific bits: ``build_prompt``,
``render_completion``, optional ``extract_fields`` / ``visibility_policy``.

Design note — NOT a MultiTurnEnv subclass: ``MultiTurnEnv.rollout`` is
``@final`` and shaped for a single (env → agent → env) conversation.
Multi-agent rollouts are N speakers sharing a transcript — a different
shape that warrants a sibling of MultiTurnEnv, not a subclass.
"""

import asyncio
import logging
import time
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Literal, final

import verifiers as vf
from verifiers.clients import Client
from verifiers.envs.multi_agent_kernel import (
    ContentChannels,
    KernelState,
    SlotProgram,
    TurnSlot,
    Utterance,
    apply_action,
)
from verifiers.envs.request_context import ModelRequestContext
from verifiers.types import (
    Messages,
    Response,
    ResponseMessage,
    RolloutInput,
    SamplingArgs,
    State,
    TrajectoryStep,
)
from verifiers.utils.message_utils import (
    fold_consecutive_user_messages,
    maybe_normalize_messages,
)
from verifiers.utils.response_utils import (
    parse_response_message,
    parse_response_tokens,
)
from verifiers.utils.usage_utils import StateUsageTracker

_log = logging.getLogger(__name__)

VisibilityMode = Literal["full", "public_only", "hidden"]
_PREFIX_CANDIDATE_INDICES_KEY = "_multi_agent_prefix_candidate_indices"


class MultiAgentEnv(vf.Environment):
    """Generic N-agent rollout environment.

    Rollout contract::

        init_state → KernelState(slot_index=0)
        while not is_completed:
            slot = schedule.current_slot(kernel)
            if slot is None: break
            if len(slot.agents) > 1: run_simultaneous_slot
            else:                   run_sequential_slot
        render_completion

    Monotonic prompt invariant (subclass CONTRACT):
        ``build_prompt(state, A, slot_N+1)`` MUST structurally extend
        ``build_prompt(state, A, slot_N)`` by appending at most a
        [user, assistant-prefill] pair at the tail. No prior messages
        modified or removed. Violating this defeats the vLLM prefix-cache
        reuse path in ``OpenAIChatCompletionsTokenClient`` and forces
        O(T²) tokenization over a T-turn episode.
    """

    is_multi_agent: bool = True

    def __init__(
        self,
        *,
        schedule: SlotProgram,
        members: list[str],
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.is_multi_agent = True
        if not members:
            raise ValueError("MultiAgentEnv requires a non-empty members list")
        if len(members) != len(set(members)):
            raise ValueError(f"MultiAgentEnv.members contains duplicates: {members}")
        self.schedule: SlotProgram = schedule
        self.members: list[str] = list(members)
        self.timeout_seconds = timeout_seconds
        self.max_total_completion_tokens: int = -1

    def set_max_total_completion_tokens(self, max_total_completion_tokens: int) -> None:
        """Set the maximum total completion tokens for this environment."""
        self.max_total_completion_tokens = max_total_completion_tokens

    def mark_timed_out(self, state: State) -> None:
        state["timed_out"] = True
        state["is_completed"] = True
        state["stop_condition"] = "timeout_reached"
        state["error"] = vf.RolloutTimeoutError(
            "multi-agent rollout exceeded timeout_seconds"
        )

    # -- abstract: subclass must implement -----------------------------------

    @abstractmethod
    async def build_prompt(
        self, state: State, member_id: str, slot: TurnSlot
    ) -> Messages:
        """Build the prompt for ``member_id`` at ``slot``.

        MUST satisfy the monotonic extension invariant (see class docstring).
        """

    @abstractmethod
    async def render_completion(self, state: State) -> None:
        """Populate ``state['completion']`` from the trajectory."""

    # -- optional hooks (sensible defaults) ----------------------------------

    async def extract_fields(
        self, public_channel: str, member_id: str, slot: TurnSlot
    ) -> dict[str, Any] | None:
        """Extract structured fields from a committed public channel.

        Default: no structured fields.
        """
        return None

    def visibility_policy(self, utt: Utterance, viewer_id: str) -> VisibilityMode:
        """Control what ``viewer_id`` sees of ``utt``.

        Default: opponents see public only; authors see full.
        """
        if utt.member_id == viewer_id:
            return "full"
        return "public_only"

    def split_response_channels(
        self, response: Response, member_id: str, slot: TurnSlot
    ) -> tuple[str, ContentChannels]:
        """Turn a raw provider response into protocol channels.

        Core multi-agent has no private-channel syntax. Protocol subclasses
        such as debate can override this to strip XML, merge provider reasoning,
        or quarantine malformed text before the kernel commits it.
        """
        raw = _coerce_text_content(response.message.content)
        return raw, ContentChannels(public=raw.strip())

    @staticmethod
    def _branch_state(state: State) -> State:
        """Clone rollout state for a simultaneous request branch.

        ``State.__setitem__`` forwards writes to input fields like
        ``prompt``/``info`` into ``state["input"]``. A shallow copy would share
        that input mapping and let branch-local writes poison the parent
        rollout, so branch states get their own input copy.
        """
        branch = State(state)
        if "input" in state:
            branch["input"] = deepcopy(state["input"])
        branch["metrics"] = {}
        return branch

    async def _get_member_response(
        self,
        state: State,
        prompt: Messages,
        member_id: str,
        slot: TurnSlot,
        *,
        request_context: ModelRequestContext,
    ) -> Response:
        try:
            return await self.get_model_response(
                state,
                prompt,
                request_context=request_context,
            )
        except vf.ReasoningOnlyEmptyResponseError as exc:
            _log.warning(
                "member=%s slot=%s emitted reasoning but no visible content; "
                "kernel will quarantine this turn.",
                member_id,
                slot.slot_id,
            )
            response = Response(
                id=f"reasoning-only-empty:{member_id}:{slot.slot_id}",
                created=int(time.time()),
                model=state["model"],
                usage=exc.usage,
                message=ResponseMessage(
                    content="",
                    reasoning_content=exc.reasoning_content,
                    finish_reason="stop",
                    is_truncated=False,
                    tokens=exc.tokens,
                    tool_calls=None,
                ),
            )
            if request_context.usage_tracker is not None:
                request_context.usage_tracker.increment_from_response(response)
            return response

    # -- prompt preparation --------------------------------------------------

    async def _prepare_prompt(
        self, state: State, agent: str, slot: TurnSlot
    ) -> Messages:
        """Subclass ``build_prompt`` → normalize at boundary → fold.

        Pipeline order matters. ``maybe_normalize_messages`` guards the
        subclass boundary: if a subclass returns raw dicts it warns once
        and promotes to typed. ``fold_consecutive_user_messages`` then
        runs typed-in → typed-out, preserving types through to the
        model-request and the renderer bridge. Folding collapses adjacent
        user runs so chat-template rendering stays in-distribution AND the
        renderer-client bridge stays valid: ``_is_valid_incremental_tail``
        only accepts a SINGLE trailing user message, so an unfolded
        [user, user] continuation tail (opponent + instruction) would miss
        the bridge on every turn and force a full re-render. Speaker
        provenance is carried inside the folded content by per-protocol
        attribution (DebateEnv: ``opponent_wrap`` labeling), not by message
        boundaries.
        """
        prompt = await self.build_prompt(state, agent, slot)
        prompt = maybe_normalize_messages(prompt, field_name="multi_agent_prompt")
        return fold_consecutive_user_messages(prompt)

    def _get_prefix_candidate_indices(
        self, state: State, member_id: str
    ) -> tuple[int, ...]:
        """Return trajectory indices that can anchor ``member_id``'s next prompt.

        The trajectory is a flat transcript across every member. Renderer/token
        continuation is per-member, so clients need a narrow candidate set to
        avoid repeatedly scanning unrelated turns and accidentally anchoring on
        another member with a compatible-looking prefix.
        """
        raw = state.get(_PREFIX_CANDIDATE_INDICES_KEY)
        if not isinstance(raw, dict):
            raw = self._bootstrap_prefix_candidate_indices(state)
            state[_PREFIX_CANDIDATE_INDICES_KEY] = raw

        indices = raw.get(member_id, ())
        return tuple(i for i in indices if isinstance(i, int))

    def _record_prefix_candidate_index(
        self, state: State, member_id: str, step_index: int
    ) -> None:
        raw = state.get(_PREFIX_CANDIDATE_INDICES_KEY)
        if not isinstance(raw, dict):
            raw = self._bootstrap_prefix_candidate_indices(state)
            state[_PREFIX_CANDIDATE_INDICES_KEY] = raw
        raw.setdefault(member_id, []).append(step_index)

    @staticmethod
    def _bootstrap_prefix_candidate_indices(state: State) -> dict[str, list[int]]:
        by_member: dict[str, list[int]] = {}
        for idx, step in enumerate(state.get("trajectory", []) or []):
            extras = step.get("extras") if isinstance(step, dict) else None
            member_id = extras.get("member_id") if isinstance(extras, dict) else None
            if isinstance(member_id, str):
                by_member.setdefault(member_id, []).append(idx)
        return by_member

    @staticmethod
    def _merge_metrics(state: State, metrics: object) -> None:
        if not isinstance(metrics, dict):
            return
        target = state.get("metrics")
        if not isinstance(target, dict):
            target = {}
            state["metrics"] = target
        for key, value in metrics.items():
            if not isinstance(key, str) or not isinstance(value, (int, float)):
                continue
            previous = target.get(key, 0.0)
            if not isinstance(previous, (int, float)):
                previous = 0.0
            target[key] = float(previous) + float(value)

    # -- stop conditions (priority-ordered) ----------------------------------

    @vf.stop(priority=100)
    async def has_error(self, state: State) -> bool:
        return state.get("error") is not None

    @vf.stop(priority=50)
    async def schedule_exhausted(self, state: State) -> bool:
        kernel = state.get("_kernel")
        if kernel is None:
            return False
        return self.schedule.current_slot(kernel) is None

    @vf.stop(priority=60)
    async def max_total_completion_tokens_reached(self, state: State) -> bool:
        if self.max_total_completion_tokens <= 0:
            return False
        usage = self.get_state_usage(state)
        if usage is None:
            return False
        return usage["output_tokens"] >= self.max_total_completion_tokens

    @vf.stop(priority=10)
    async def prompt_too_long(self, state: State) -> bool:
        return state.get("prompt_too_long", False)

    # -- final: rollout loop -------------------------------------------------

    @final
    async def rollout(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
        generation: vf.MemberGenerationPlan | None = None,
    ) -> State:
        state = await self.init_state(input, client, model, sampling_args, generation)

        async def rollout_loop() -> None:
            state["timing"].generation.start = time.time()
            state["_kernel"] = KernelState(slot_index=0)

            while not await self.is_completed(state):
                slot = self.schedule.current_slot(state["_kernel"])
                if slot is None:
                    break
                try:
                    if len(slot.agents) > 1:
                        await self._run_simultaneous_slot(state, slot)
                    else:
                        await self._run_sequential_slot(state, slot)
                except vf.OverlongPromptError:
                    state["prompt_too_long"] = True
                    state["is_truncated"] = True
                except vf.Error as e:
                    state["error"] = e

        try:
            await asyncio.wait_for(rollout_loop(), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            self.mark_timed_out(state)
        finally:
            try:
                await self.render_completion(state)
            finally:
                state["timing"].generation.end = time.time()
                await self.cleanup(state)
        return state

    @final
    async def _run_sequential_slot(self, state: State, slot: TurnSlot) -> None:
        agent = slot.agents[0]
        prompt = await self._prepare_prompt(state, agent, slot)
        parent_tracker = self._get_usage_tracker(state, create_if_missing=True)
        response = await self._get_member_response(
            state,
            prompt,
            agent,
            slot,
            request_context=ModelRequestContext(
                member_id=agent,
                usage_tracker=parent_tracker,
                prefix_candidate_indices=self._get_prefix_candidate_indices(
                    state, agent
                ),
            ),
        )
        content, channels = self.split_response_channels(response, agent, slot)
        token_count = _completion_token_count(response)

        result = apply_action(
            state["_kernel"],
            self.schedule,
            agent,
            content,
            token_count,
            channels,
        )
        utt = result.committed[0]
        fields = await self.extract_fields(utt.public_channel, agent, slot)
        step = await self._build_step(state, prompt, response, utt, fields)
        state["trajectory"].append(step)
        self._record_prefix_candidate_index(state, agent, len(state["trajectory"]) - 1)
        state["_kernel"] = result.new_state

    @final
    async def _run_simultaneous_slot(self, state: State, slot: TurnSlot) -> None:
        """Fully-staged atomic commit.

        Three-phase protocol — every phase runs to completion before the
        next begins, and nothing in state is mutated until the final
        publish step:

          1. Fan out model calls under TaskGroup. On first raise,
             TaskGroup cancels every sibling coroutine → no wasted
             tokens, no late completions leaking into the shared usage
             tracker after the slot is doomed.
          2. Stage: fold responses into a local kernel (the real
             state["_kernel"] is NOT touched), build per-agent
             TrajectorySteps, run extract_fields. Any raise here discards
             the local buffers entirely.
          3. Publish: append every TrajectoryStep, assign
             state["_kernel"] and merge per-agent usage trackers. These
             writes are all non-await, so if we reach phase 3 the slot
             succeeds atomically.
        """
        prompts = [await self._prepare_prompt(state, a, slot) for a in slot.agents]
        prefix_candidate_indices = [
            self._get_prefix_candidate_indices(state, a) for a in slot.agents
        ]

        # Per-agent request contexts isolate the prefix-cache partition key
        # (``member_id``) and usage accounting across concurrent branches
        # without cloning the shared rollout state. Every branch charges a
        # child tracker; only the publish phase merges them back into the
        # parent, so a doomed slot never leaks token usage.
        parent_tracker = self._get_usage_tracker(state, create_if_missing=True)
        per_agent_trackers: list[StateUsageTracker | None] = [
            parent_tracker.fork() if parent_tracker is not None else None
            for _ in slot.agents
        ]
        per_agent_states = [self._branch_state(state) for _ in slot.agents]

        # Phase 1: fan out concurrent model calls. On first raise, cancel
        # still-pending siblings — no wasted tokens, no late completions
        # leaking into the shared usage tracker after the slot is doomed.
        # Implemented with asyncio.wait(FIRST_EXCEPTION) to preserve the
        # cancel-on-first-raise semantics TaskGroup would give us while
        # staying compatible with Python 3.10 (TaskGroup is 3.11+ and
        # ``requires-python`` declares ``>=3.10``).
        responses: list[Response] = [None] * len(slot.agents)  # type: ignore[list-item]

        async def _run_one(idx: int) -> None:
            p = prompts[idx]
            responses[idx] = await self._get_member_response(
                per_agent_states[idx],
                p,
                slot.agents[idx],
                slot,
                request_context=ModelRequestContext(
                    member_id=slot.agents[idx],
                    usage_tracker=per_agent_trackers[idx],
                    prefix_candidate_indices=prefix_candidate_indices[idx],
                ),
            )

        tasks = [asyncio.create_task(_run_one(i)) for i in range(len(slot.agents))]

        # ``asyncio.wait`` itself can be cancelled by the parent (rollout
        # timeout, Ctrl-C, encompassing wait_for) — if that happens while
        # tasks are inflight they keep running, leak tokens into the
        # shared usage tracker, and can late-commit into the wrong slot.
        # TaskGroup would have drained them on ``__aexit__``; we replicate
        # that behaviour explicitly with try/finally.
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )
        except BaseException:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        # First-exception return — cancel the stragglers AND capture any
        # results they already produced. ``asyncio.gather(...,
        # return_exceptions=True)`` returns each task's result or
        # exception; pending tasks that had already raised before our
        # cancel took effect would otherwise be silently dropped, losing
        # their error for the priority picker below.
        pending_results: list[Any] = []
        if pending:
            for t in pending:
                t.cancel()
            pending_results = list(
                await asyncio.gather(*pending, return_exceptions=True)
            )

        # Collect real exceptions from ``done`` AND from any pending task
        # whose raise beat our cancel. Filter out the ``CancelledError``s
        # we ourselves triggered — those are housekeeping, not signal.
        real_errors: list[BaseException] = []
        for idx, task in enumerate(tasks):
            if task not in done:
                continue
            if task.cancelled():
                real_errors.append(
                    vf.Error(
                        f"simultaneous slot {slot.slot_id}: model task for "
                        f"member {slot.agents[idx]!r} was cancelled"
                    )
                )
                continue
            exc = task.exception()
            if exc is not None and not isinstance(exc, asyncio.CancelledError):
                real_errors.append(exc)
        for r in pending_results:
            if isinstance(r, BaseException) and not isinstance(
                r, asyncio.CancelledError
            ):
                real_errors.append(r)

        if real_errors:
            # Priority: OverlongPromptError first (most-specific recovery
            # in the rollout loop), then any other vf.Error, then bubble
            # the first remaining.
            chosen = (
                next(
                    (e for e in real_errors if isinstance(e, vf.OverlongPromptError)),
                    None,
                )
                or next(
                    (e for e in real_errors if isinstance(e, vf.Error)),
                    None,
                )
                or real_errors[0]
            )
            others = tuple(e for e in real_errors if e is not chosen)
            if others:
                _log_suppressed_peers(slot.slot_id, others)
            raise chosen

        # Phase 2: stage the fold into a LOCAL kernel + build trajectory
        # steps + extract fields. state["_kernel"] untouched.
        staged_kernel = state["_kernel"]
        staged_steps: list[TrajectoryStep] = []
        committed_utts: list[Utterance] = []
        for agent, response in zip(slot.agents, responses):
            content, channels = self.split_response_channels(response, agent, slot)
            token_count = _completion_token_count(response)
            result = apply_action(
                staged_kernel,
                self.schedule,
                agent,
                content,
                token_count,
                channels,
            )
            staged_kernel = result.new_state
            if result.committed:
                committed_utts.extend(result.committed)

        if len(committed_utts) != len(slot.agents):
            raise vf.Error(
                f"simultaneous slot {slot.slot_id}: expected "
                f"{len(slot.agents)} commits, got {len(committed_utts)}"
            )

        for agent, prompt, response, utt in zip(
            slot.agents, prompts, responses, committed_utts
        ):
            fields = await self.extract_fields(utt.public_channel, agent, slot)
            step = await self._build_step(state, prompt, response, utt, fields)
            staged_steps.append(step)

        # Phase 3: PUBLISH. Synchronous tail: trajectory appends + kernel
        # assignment + usage-tracker merge. No awaits, no raises after this
        # point. Merging is the accounting-side counterpart of the kernel
        # assignment — both happen iff the slot succeeds.
        next_step_index = len(state["trajectory"])
        for agent, step in zip(slot.agents, staged_steps):
            state["trajectory"].append(step)
            self._record_prefix_candidate_index(state, agent, next_step_index)
            next_step_index += 1
        state["_kernel"] = staged_kernel
        for branch_state in per_agent_states:
            self._merge_metrics(state, branch_state.get("metrics"))
        if isinstance(parent_tracker, StateUsageTracker):
            for child in per_agent_trackers:
                if child is not None:
                    parent_tracker.merge(child)

    # -- trajectory step build / append -------------------------------------

    async def _build_step(
        self,
        state: State,
        prompt: Messages,
        response: Response,
        utt: Utterance,
        fields: dict[str, Any] | None,
    ) -> TrajectoryStep:
        """Build a TrajectoryStep without mutating state.

        State is only mutated by the caller's ``trajectory.append``; the
        simultaneous-slot atomic staging phase constructs every step first
        (async: token parsing) and bails cleanly if any one raises.
        """
        completion_messages = await parse_response_message(response)
        tokens = await parse_response_tokens(response, self.max_seq_len)
        response_is_truncated = response.message.is_truncated or False
        is_truncated = response_is_truncated or (
            tokens is not None and tokens["is_truncated"]
        )
        extras: dict[str, Any] = {
            "member_id": utt.member_id,
            "phase": utt.phase,
        }
        generation = self.generation_metadata_for_member(state, utt.member_id)
        if generation is not None:
            extras["generation"] = generation
        if fields is not None:
            extras["fields"] = fields
        if utt.parse_error is not None:
            extras["parse_error"] = utt.parse_error
        return TrajectoryStep(
            prompt=prompt,
            completion=completion_messages,
            response=response,
            tokens=tokens,
            reward=None,
            advantage=None,
            is_truncated=is_truncated,
            trajectory_id=state["trajectory_id"],
            extras=extras,
        )


def _completion_token_count(response: Response) -> int:
    if response.message.tokens and response.message.tokens.completion_ids:
        return len(response.message.tokens.completion_ids)
    return 0


def _coerce_text_content(content: Any) -> str:
    """Coerce ``response.message.content`` to a string for the kernel.

    Provider returns either ``str`` or ``list[ContentPart]`` (multimodal).
    The MA pipeline stores text transcript entries. Text parts are joined and
    non-text parts are dropped.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                txt = part.get("text")
                if isinstance(txt, str):
                    text_parts.append(txt)
            else:
                txt = getattr(part, "text", None)
                if isinstance(txt, str):
                    text_parts.append(txt)
        return "".join(text_parts)
    return ""


def _log_suppressed_peers(slot_id: int, exceptions: tuple[BaseException, ...]) -> None:
    if exceptions:
        _log.warning(
            "simultaneous-slot %d suppressed peer exceptions: %r",
            slot_id,
            exceptions,
        )
