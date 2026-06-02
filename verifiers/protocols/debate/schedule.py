"""Plain-language rendering of a debate turn schedule for the judge prompt.

This is debate-specific (judge framing, answers/replies, ``judge_members``) and
deliberately lives in ``protocols/debate`` rather than ``multi_agent_kernel`` so
the generic schedule primitive carries no protocol policy.
"""

from verifiers.envs.multi_agent_kernel import SlotProgram


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
            opener = (
                "Finally, " if i == len(speaking) - 1 and len(speaking) > 2 else "Then "
            )
            if together:
                sentences.append(
                    f"{opener}they each read everything said so far and wrote a reply "
                    f"at the same time, {_unseen_reply(len(slot.agents))}."
                )
            else:
                sentences.append(
                    f"{opener}{who} read everything said so far and replied."
                )

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
