"""Pure-functional multi-agent episode kernel."""

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from verifiers.errors import KernelProtocolError


@dataclass(frozen=True)
class TurnSlot:
    """One step in the schedule.

    len(agents) == 1 → sequential (commit immediately).
    len(agents) > 1  → simultaneous (barrier until all submit).
    """

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
    """Immutable episode state.

    ``_active_slot`` caches the slot for the current simultaneous barrier,
    guarding against non-deterministic SlotProgram implementations.
    """

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
    """SlotProgram backed by a fixed tuple of TurnSlots."""

    def __init__(self, slots: tuple[TurnSlot, ...]) -> None:
        self._slots = slots

    def current_slot(self, state: KernelState) -> TurnSlot | None:
        if state.slot_index >= len(self._slots):
            return None
        return self._slots[state.slot_index]

    def __len__(self) -> int:
        return len(self._slots)


def _natural_join(names: tuple[str, ...]) -> str:
    items = list(names)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def schedule_to_explainer(schedule: SlotProgram, *, judge_members: set[str]) -> str:
    """Describe a debate schedule to the judge in plain, non-jargon language.

    Derived from the slot structure alone, so the description stays truthful if
    the schedule changes (sequential, hybrid, or more rounds) — no protocol
    assumptions are hard-coded in the prompt. A multi-agent slot is a
    simultaneous turn (atomic commit: co-actors do not see each other that
    turn); a single-agent slot is sequential. Slots whose agents are all judges
    are skipped (the judge is the reader). Dynamic schedules that expose no
    enumerable ``_slots`` fall back to a generic line.
    """
    slots = getattr(schedule, "_slots", None)
    if not slots:
        return (
            "You're looking at everything the participants said, in order. "
            "Decide which of them you find more convincing."
        )

    speaking = [slot for slot in slots if not set(slot.agents) <= judge_members]
    if not speaking:
        return "You're looking at everything the participants said, in order."

    def _unseen_first(n: int) -> str:
        return (
            "so neither of them could see what the other was saying"
            if n == 2
            else "so none of them could see what the others were saying"
        )

    def _unseen_reply(n: int) -> str:
        return (
            "so neither saw the other's reply while writing their own"
            if n == 2
            else "so none of them saw the others' replies while writing their own"
        )

    sentences: list[str] = []
    for i, slot in enumerate(speaking):
        who = _natural_join(slot.agents)
        together = len(slot.agents) > 1
        if i == 0:
            if together:
                sentences.append(
                    f"To start, {who} each gave their answer at the same time, "
                    f"{_unseen_first(len(slot.agents))}."
                )
            else:
                sentences.append(f"To start, {who} gave their answer.")
        else:
            opener = "Finally, " if i == len(speaking) - 1 and len(speaking) > 2 else "Then "
            if together:
                sentences.append(
                    f"{opener}they each read everything said so far and wrote a reply "
                    f"at the same time, {_unseen_reply(len(slot.agents))}."
                )
            else:
                sentences.append(f"{opener}{who} read everything said so far and replied.")

    closing = "You're now looking at everything they wrote."
    last = speaking[-1]
    if len(speaking) >= 2 and len(last.agents) > 1:
        tail = (
            "neither of them had a chance to respond to the other's last reply"
            if len(last.agents) == 2
            else "none of them had a chance to respond to the others' last replies"
        )
        closing += f" Keep in mind that {tail}."
    return " ".join(sentences) + " " + closing


def causal_transcript_view(
    transcript: tuple[Utterance, ...],
    member_id: str,
    *,
    start: int = 0,
) -> tuple[Utterance, ...]:
    """Return committed utterances in the replay order seen by ``member_id``.

    The kernel stores one archival transcript: sequential commits in schedule
    order, and simultaneous commits as a contiguous slot group in canonical
    slot-agent order. For prompt replay, a participant's own utterance from a
    simultaneous barrier must appear before peer utterances from that same
    barrier; those peer utterances were not visible when the participant spoke.

    This is topology-only. Protocols still decide which channel contents are
    visible to a viewer.
    """
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
    """Pure reducer. Raises KernelProtocolError on protocol violations.

    The kernel knows turn order and commit semantics only. Protocols own
    channel parsing and pass the resulting public/private view in ``channels``.
    """
    slot = (
        state._active_slot
        if state._active_slot is not None
        else program.current_slot(state)
    )

    if slot is None:
        raise KernelProtocolError("No active slot — episode is finished")

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

    # Sequential: commit immediately
    if len(slot.agents) == 1:
        return ActionResult(
            new_state=replace(
                state,
                slot_index=state.slot_index + 1,
                transcript=state.transcript + (utterance,),
            ),
            committed=(utterance,),
        )

    # Simultaneous: buffer until all agents submit
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
