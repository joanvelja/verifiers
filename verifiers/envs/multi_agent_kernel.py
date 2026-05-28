"""Pure-functional multi-agent episode kernel."""

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from verifiers.errors import KernelProtocolError


@dataclass(frozen=True)
class TurnSlot:
    """One step in the schedule."""

    slot_id: int
    agents: tuple[str, ...]
    phase: str = ""

    def __post_init__(self) -> None:
        if not self.agents:
            raise ValueError("TurnSlot.agents must be non-empty")
        if len(self.agents) != len(set(self.agents)):
            raise ValueError(f"TurnSlot.agents contains duplicates: {self.agents}")


@dataclass(frozen=True)
class ContentChannels:
    """Protocol-facing view of a raw model response."""

    public: str
    private: str | None = None
    parse_error: str | None = None


@dataclass(frozen=True)
class Utterance:
    """Structured agent output committed to the transcript."""

    member_id: str
    turn_index: int
    slot_id: int
    phase: str
    raw_content: str
    public_channel: str
    private_channel: str | None
    token_count: int
    parse_error: str | None = None


@dataclass(frozen=True)
class KernelState:
    """Immutable episode state."""

    slot_index: int
    transcript: tuple[Utterance, ...] = ()
    pending: MappingProxyType[str, Utterance] = field(
        default_factory=lambda: MappingProxyType({})
    )
    _active_slot: TurnSlot | None = None


@dataclass(frozen=True)
class ActionResult:
    new_state: KernelState
    committed: tuple[Utterance, ...]


@runtime_checkable
class SlotProgram(Protocol):
    """Returns the current slot, or None when the episode is finished."""

    def current_slot(self, state: KernelState) -> TurnSlot | None: ...


class StaticSchedule:
    """SlotProgram backed by a fixed tuple of TurnSlot objects."""

    def __init__(self, slots: tuple[TurnSlot, ...]) -> None:
        self._slots = slots

    def current_slot(self, state: KernelState) -> TurnSlot | None:
        if state.slot_index >= len(self._slots):
            return None
        return self._slots[state.slot_index]

    def __len__(self) -> int:
        return len(self._slots)


def causal_transcript_view(
    transcript: tuple[Utterance, ...],
    member_id: str,
    *,
    start: int = 0,
) -> tuple[Utterance, ...]:
    """Return committed utterances in the replay order seen by ``member_id``."""
    ordered: list[Utterance] = []
    idx = start
    while idx < len(transcript):
        group = _next_slot_group(transcript, idx)
        own = tuple(utt for utt in group if utt.member_id == member_id)
        if own:
            ordered.extend(own)
            ordered.extend(utt for utt in group if utt.member_id != member_id)
        else:
            ordered.extend(group)
        idx += len(group)
    return tuple(ordered)


def _next_slot_group(
    transcript: tuple[Utterance, ...],
    start: int,
) -> tuple[Utterance, ...]:
    turn_index = transcript[start].turn_index
    end = start + 1
    while end < len(transcript) and transcript[end].turn_index == turn_index:
        end += 1
    return transcript[start:end]


def apply_action(
    state: KernelState,
    program: SlotProgram,
    member_id: str,
    raw_content: str,
    token_count: int,
    channels: ContentChannels,
) -> ActionResult:
    """Pure reducer. Raises KernelProtocolError on protocol violations."""
    slot = (
        state._active_slot
        if state._active_slot is not None
        else program.current_slot(state)
    )

    if slot is None:
        raise KernelProtocolError("No active slot - episode is finished")

    if member_id not in slot.agents:
        raise KernelProtocolError(
            f"Member {member_id!r} is not scheduled for slot {slot.slot_id} "
            f"(expected one of {slot.agents})"
        )

    if member_id in state.pending:
        raise KernelProtocolError(
            f"Member {member_id!r} already submitted for slot {slot.slot_id}"
        )

    parse_error = channels.parse_error
    if parse_error is None and not channels.public.strip():
        parse_error = "empty public channel (model emitted no visible answer)"

    utterance = Utterance(
        member_id=member_id,
        turn_index=state.slot_index,
        slot_id=slot.slot_id,
        phase=slot.phase,
        raw_content=raw_content,
        public_channel=channels.public,
        private_channel=channels.private,
        token_count=token_count,
        parse_error=parse_error,
    )

    if len(slot.agents) == 1:
        return ActionResult(
            new_state=replace(
                state,
                slot_index=state.slot_index + 1,
                transcript=state.transcript + (utterance,),
            ),
            committed=(utterance,),
        )

    new_pending = {**state.pending, member_id: utterance}

    if len(new_pending) == len(slot.agents):
        committed = tuple(new_pending[agent] for agent in slot.agents)
        return ActionResult(
            new_state=replace(
                state,
                slot_index=state.slot_index + 1,
                transcript=state.transcript + committed,
                pending=MappingProxyType({}),
                _active_slot=None,
            ),
            committed=committed,
        )

    return ActionResult(
        new_state=replace(
            state,
            pending=MappingProxyType(new_pending),
            _active_slot=slot,
        ),
        committed=(),
    )
