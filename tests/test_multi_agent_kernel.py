from verifiers.envs.multi_agent_kernel import (
    ContentChannels,
    KernelState,
    StaticSchedule,
    TurnSlot,
    apply_action,
    causal_transcript_view,
)


def _commit(
    schedule: StaticSchedule,
    actions: list[tuple[str, str]],
):
    kernel = KernelState(slot_index=0)
    for member_id, content in actions:
        result = apply_action(
            kernel,
            schedule,
            member_id,
            content,
            token_count=1,
            channels=ContentChannels(public=content),
        )
        kernel = result.new_state
    return kernel.transcript


def _ids(transcript) -> list[str]:  # noqa: ANN001
    return [f"{utt.member_id}:{utt.raw_content}" for utt in transcript]


def test_causal_transcript_view_preserves_sequential_transcript_order() -> None:
    schedule = StaticSchedule(
        (
            TurnSlot(slot_id=0, agents=("agent_a",), phase="open"),
            TurnSlot(slot_id=1, agents=("agent_b",), phase="reply"),
            TurnSlot(slot_id=2, agents=("judge",), phase="final"),
        )
    )
    transcript = _commit(
        schedule,
        [
            ("agent_a", "a0"),
            ("agent_b", "b1"),
            ("judge", "j2"),
        ],
    )

    expected = ["agent_a:a0", "agent_b:b1", "judge:j2"]
    assert _ids(causal_transcript_view(transcript, "agent_a")) == expected
    assert _ids(causal_transcript_view(transcript, "agent_b")) == expected
    assert _ids(causal_transcript_view(transcript, "judge")) == expected


def test_causal_transcript_view_preserves_reused_sequential_slot_ids() -> None:
    schedule = StaticSchedule(
        (
            TurnSlot(slot_id=0, agents=("agent_b",), phase="first"),
            TurnSlot(slot_id=0, agents=("agent_a",), phase="second"),
        )
    )
    transcript = _commit(
        schedule,
        [
            ("agent_b", "b0"),
            ("agent_a", "a1"),
        ],
    )

    expected = ["agent_b:b0", "agent_a:a1"]
    assert _ids(transcript) == expected
    assert _ids(causal_transcript_view(transcript, "agent_a")) == expected


def test_causal_transcript_view_replays_own_turn_first_in_simultaneous_group() -> None:
    schedule = StaticSchedule(
        (
            TurnSlot(
                slot_id=0,
                agents=("agent_a", "agent_b", "agent_c"),
                phase="round",
            ),
        )
    )
    transcript = _commit(
        schedule,
        [
            ("agent_c", "c0"),
            ("agent_b", "b0"),
            ("agent_a", "a0"),
        ],
    )

    assert _ids(transcript) == ["agent_a:a0", "agent_b:b0", "agent_c:c0"]
    assert _ids(causal_transcript_view(transcript, "agent_a")) == [
        "agent_a:a0",
        "agent_b:b0",
        "agent_c:c0",
    ]
    assert _ids(causal_transcript_view(transcript, "agent_b")) == [
        "agent_b:b0",
        "agent_a:a0",
        "agent_c:c0",
    ]
    assert _ids(causal_transcript_view(transcript, "agent_c")) == [
        "agent_c:c0",
        "agent_a:a0",
        "agent_b:b0",
    ]
    assert _ids(causal_transcript_view(transcript, "judge")) == [
        "agent_a:a0",
        "agent_b:b0",
        "agent_c:c0",
    ]


def test_causal_transcript_view_handles_mixed_sequential_simultaneous_suffixes() -> (
    None
):
    schedule = StaticSchedule(
        (
            TurnSlot(slot_id=0, agents=("agent_a",), phase="open"),
            TurnSlot(
                slot_id=1,
                agents=("agent_a", "agent_b", "agent_c"),
                phase="round",
            ),
            TurnSlot(slot_id=2, agents=("judge",), phase="final"),
        )
    )
    transcript = _commit(
        schedule,
        [
            ("agent_a", "a0"),
            ("agent_c", "c1"),
            ("agent_b", "b1"),
            ("agent_a", "a1"),
            ("judge", "j2"),
        ],
    )

    assert _ids(transcript) == [
        "agent_a:a0",
        "agent_a:a1",
        "agent_b:b1",
        "agent_c:c1",
        "judge:j2",
    ]
    assert _ids(causal_transcript_view(transcript, "agent_b", start=1)) == [
        "agent_b:b1",
        "agent_a:a1",
        "agent_c:c1",
        "judge:j2",
    ]
    assert _ids(causal_transcript_view(transcript, "agent_c", start=1)) == [
        "agent_c:c1",
        "agent_a:a1",
        "agent_b:b1",
        "judge:j2",
    ]
    assert _ids(causal_transcript_view(transcript, "judge", start=1)) == [
        "agent_a:a1",
        "agent_b:b1",
        "agent_c:c1",
        "judge:j2",
    ]
