"""Generic N-agent rollout loop on the Environment contract."""

import asyncio
import logging
import time
from abc import abstractmethod
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

VisibilityMode = Literal["full", "public_only", "hidden"]
_PREFIX_CANDIDATE_INDICES_KEY = "_multi_agent_prefix_candidate_indices"


class MultiAgentEnv(vf.Environment):
    """Generic N-agent rollout environment."""

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

    @abstractmethod
    async def build_prompt(
        self, state: State, member_id: str, slot: TurnSlot
    ) -> Messages:
        """Build the prompt for ``member_id`` at ``slot``."""

    @abstractmethod
    async def render_completion(self, state: State) -> None:
        """Populate ``state['completion']`` from the trajectory."""

    async def extract_fields(
        self, public_channel: str, member_id: str, slot: TurnSlot
    ) -> dict[str, Any] | None:
        """Extract structured fields from a committed public channel."""
        return None

    def visibility_policy(self, utt: Utterance, viewer_id: str) -> VisibilityMode:
        """Control what ``viewer_id`` sees of ``utt``."""
        if utt.member_id == viewer_id:
            return "full"
        return "public_only"

    def split_response_channels(
        self, response: Response, member_id: str, slot: TurnSlot
    ) -> tuple[str, ContentChannels]:
        """Turn a raw provider response into protocol channels."""
        raw = _coerce_text_content(response.message.content)
        return raw, ContentChannels(public=raw.strip())

    async def _prepare_prompt(
        self, state: State, agent: str, slot: TurnSlot
    ) -> Messages:
        prompt = await self.build_prompt(state, agent, slot)
        prompt = maybe_normalize_messages(prompt, field_name="multi_agent_prompt")
        return fold_consecutive_user_messages(prompt)

    def _get_prefix_candidate_indices(
        self, state: State, member_id: str
    ) -> tuple[int, ...]:
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

    @final
    async def rollout(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        state = await self.init_state(input, client, model, sampling_args)
        state["metrics"] = {}

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
        response = await self.get_model_response(
            state,
            prompt,
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
        state["_kernel"] = result.new_state
        utt = result.committed[0]
        fields = await self.extract_fields(utt.public_channel, agent, slot)
        step = await self._build_step(state, prompt, response, utt, fields)
        state["trajectory"].append(step)
        self._record_prefix_candidate_index(state, agent, len(state["trajectory"]) - 1)

    @final
    async def _run_simultaneous_slot(self, state: State, slot: TurnSlot) -> None:
        prompts = [
            await self._prepare_prompt(state, agent, slot) for agent in slot.agents
        ]
        prefix_candidate_indices = [
            self._get_prefix_candidate_indices(state, agent) for agent in slot.agents
        ]
        parent_tracker = self._get_usage_tracker(state, create_if_missing=True)
        per_agent_trackers: list[StateUsageTracker | None] = [
            parent_tracker.fork() if parent_tracker is not None else None
            for _ in slot.agents
        ]
        per_agent_states = [State(state, metrics={}) for _ in slot.agents]
        responses: list[Response] = [None] * len(slot.agents)  # type: ignore[list-item]

        async def run_one(idx: int) -> None:
            responses[idx] = await self.get_model_response(
                per_agent_states[idx],
                prompts[idx],
                request_context=ModelRequestContext(
                    member_id=slot.agents[idx],
                    usage_tracker=per_agent_trackers[idx],
                    prefix_candidate_indices=prefix_candidate_indices[idx],
                ),
            )

        tasks = [asyncio.create_task(run_one(i)) for i in range(len(slot.agents))]
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        pending_results: list[Any] = []
        if pending:
            for task in pending:
                task.cancel()
            pending_results = list(
                await asyncio.gather(*pending, return_exceptions=True)
            )

        real_errors = [
            exc
            for task in done
            if (exc := task.exception()) is not None
            and not isinstance(exc, asyncio.CancelledError)
        ]
        real_errors.extend(
            result
            for result in pending_results
            if isinstance(result, BaseException)
            and not isinstance(result, asyncio.CancelledError)
        )

        if real_errors:
            chosen = (
                next(
                    (
                        err
                        for err in real_errors
                        if isinstance(err, vf.OverlongPromptError)
                    ),
                    None,
                )
                or next((err for err in real_errors if isinstance(err, vf.Error)), None)
                or real_errors[0]
            )
            others = tuple(err for err in real_errors if err is not chosen)
            _log_suppressed_peers(slot.slot_id, others)
            raise chosen

        staged_kernel = state["_kernel"]
        staged_steps: list[TrajectoryStep] = []
        committed_utts: list[Utterance] = []
        for agent, response in zip(slot.agents, responses, strict=True):
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
            slot.agents, prompts, responses, committed_utts, strict=True
        ):
            fields = await self.extract_fields(utt.public_channel, agent, slot)
            step = await self._build_step(state, prompt, response, utt, fields)
            staged_steps.append(step)

        next_step_index = len(state["trajectory"])
        for agent, step in zip(slot.agents, staged_steps, strict=True):
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

    async def _build_step(
        self,
        state: State,
        prompt: Messages,
        response: Response,
        utt: Utterance,
        fields: dict[str, Any] | None,
    ) -> TrajectoryStep:
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
    if not exceptions:
        return
    logging.getLogger(__name__).warning(
        "simultaneous-slot %d suppressed peer exceptions: %r",
        slot_id,
        exceptions,
    )
