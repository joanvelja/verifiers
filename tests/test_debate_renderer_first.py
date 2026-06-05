"""Renderer-first debate think-channel: air-gap, parity, no-fold, fail-loud.

Pins the contract of the renderer-first redesign (retire merge_provider_reasoning
fold; carry the renderer's structured reasoning_content end-to-end):

  * split_response_channels reads structured reasoning verbatim, fails loud on a
    reasoning-tag leak, quarantines (not crashes) an empty visible channel.
  * public_only cross-agent views OMIT the author's reasoning (air-gap).
  * open / visible_to_judge views FOLD the author's reasoning as labeled content
    (the renderer drops reasoning_content on user-role messages, so omission
    would silently regress these modes to empty — see workflow Lens ①).
  * own-turn replay carries reasoning_content, byte-matching the bridge anchor
    (parse_response_message) so the renderer-client bridge stays stable.
  * consecutive user messages ARE folded (renderer-bridge requirement:
    _is_valid_incremental_tail accepts only a single trailing user); opponent
    provenance is carried by opponent_wrap labeling inside the folded block.
"""

from dataclasses import replace

import pytest

from verifiers.envs.multi_agent_kernel import (
    ContentChannels,
    KernelState,
    StaticSchedule,
    TurnSlot,
    apply_action,
)
from verifiers.errors import ContentParseError
from verifiers.protocols.debate.channels import (
    reasoning_split_failed,
    structured_reasoning,
)
from verifiers.clients.renderer_client import _is_valid_incremental_tail
from verifiers.protocols.debate.env import DebateEnv
from verifiers.protocols.debate.prompts import resolve_prompts
from verifiers.protocols.debate.rubric import DebateRubric
from verifiers.types import (
    AssistantMessage,
    Response,
    ResponseMessage,
    State,
    UserMessage,
)
from verifiers.utils.response_utils import parse_response_message

_MEMBERS = ["debater_a", "debater_b", "judge"]
# Sequential schedule so the transcript is unambiguous for replay assertions.
_SCHEDULE = StaticSchedule(
    (
        TurnSlot(slot_id=0, agents=("debater_a",), phase="propose"),
        TurnSlot(slot_id=1, agents=("debater_b",), phase="propose"),
        TurnSlot(slot_id=2, agents=("debater_a",), phase="critique"),
        TurnSlot(slot_id=3, agents=("debater_b",), phase="critique"),
        TurnSlot(slot_id=4, agents=("judge",), phase="final"),
    )
)

_SECRET = "STEP-BY-STEP-SECRET-REASONING"
_PUBLIC_A = "My answer is A because the compound is chiral."


def _env(think_visibility: dict[str, str] | None = None) -> DebateEnv:
    """Real selfplay pack + real DebateRubric, optionally overriding
    think_visibility. No judge_client is wired (we never score here), so the
    rubric constructs offline."""
    prompts = resolve_prompts("selfplay")
    if think_visibility is not None:
        prompts = replace(prompts, think_visibility=think_visibility)
    rubric = DebateRubric(members=list(_MEMBERS), prompts=prompts)
    return DebateEnv(
        schedule=_SCHEDULE,
        prompts=prompts,
        members=list(_MEMBERS),
        rubric=rubric,
        dataset=lambda: None,
    )


def _commit(kernel: KernelState, member_id: str, public: str, reasoning: str | None):
    result = apply_action(
        kernel,
        _SCHEDULE,
        member_id,
        public,
        token_count=1,
        channels=ContentChannels(public=public, private=reasoning),
    )
    return result.new_state


def _response(content, reasoning=None, thinking_blocks=None) -> Response:
    """A real verifiers Response — what the renderer client hands the env after
    splitting reasoning out of the sampled tokens. No mock: drift (e.g. a
    missing tool_calls attr) would be a real bug, not a test artefact."""
    return Response(
        id="resp",
        created=0,
        model="test-model",
        message=ResponseMessage(
            content=content,
            reasoning_content=reasoning,
            thinking_blocks=thinking_blocks,
            finish_reason="stop",
            is_truncated=False,
        ),
    )


def _all_text(messages) -> str:
    return "\n".join(
        m.content for m in messages if isinstance(getattr(m, "content", None), str)
    )


# --------------------------------------------------------------------------- #
# split_response_channels — read structured reasoning, fail loud, quarantine
# --------------------------------------------------------------------------- #


def test_split_reads_structured_reasoning_verbatim():
    env = _env()
    slot = _SCHEDULE._slots[0]
    resp = _response(content=_PUBLIC_A, reasoning=_SECRET)
    visible, channels = env.split_response_channels(resp, "debater_a", slot)
    assert visible == _PUBLIC_A
    assert channels.public == _PUBLIC_A
    # Verbatim (no strip): own-turn replay sets this as reasoning_content and it
    # must byte-match the bridge anchor.
    assert channels.private == _SECRET
    assert channels.parse_error is None


def test_split_fails_loud_on_reasoning_leak():
    """A <think> tag surviving in visible content => renderer did not split =>
    CONTRACT violation => raise (would otherwise leak CoT to opponent/judge)."""
    env = _env()
    slot = _SCHEDULE._slots[0]
    leaked = _response(content=f"<think>{_SECRET}</think>{_PUBLIC_A}", reasoning=None)
    with pytest.raises(ContentParseError, match="reasoning-splitting renderer"):
        env.split_response_channels(leaked, "debater_a", slot)


def test_split_empty_visible_quarantines_not_crashes():
    """Empty visible channel is a model-quality event: no raise; the kernel
    sets parse_error so the turn is quarantined (zero reward), not a crash."""
    env = _env()
    slot = _SCHEDULE._slots[0]
    resp = _response(content="", reasoning=_SECRET)
    visible, channels = env.split_response_channels(resp, "debater_a", slot)
    assert visible == ""
    # parse_error is assigned by the kernel (apply_action), not the split.
    result = apply_action(
        KernelState(slot_index=0), _SCHEDULE, "debater_a", visible, 1, channels
    )
    assert result.committed[0].parse_error is not None


def test_structured_reasoning_is_verbatim_and_no_thinking_blocks_synthesis():
    # Verbatim incl. empty string (must byte-match the bridge anchor, which
    # stores reasoning_content verbatim).
    assert structured_reasoning(_response(_PUBLIC_A, reasoning=_SECRET)) == _SECRET
    assert structured_reasoning(_response(_PUBLIC_A, reasoning="")) == ""
    assert structured_reasoning(_response(_PUBLIC_A, reasoning=None)) is None
    # NO thinking_blocks fallback: the anchor leaves reasoning_content=None and
    # carries thinking_blocks separately, so synthesizing here would desync the
    # bridge. A thinking_blocks-only provider yields None (out of scope).
    resp = _response(
        _PUBLIC_A,
        reasoning=None,
        thinking_blocks=[{"thinking": "b1"}, {"thinking": "b2"}],
    )
    assert structured_reasoning(resp) is None


def test_reasoning_split_failed_only_on_surviving_block_when_reasoning_absent():
    # Split failed: no reasoning extracted AND a full block survives -> leak.
    assert reasoning_split_failed("<think>x</think> ans", reasoning=None) is True
    # Split succeeded (reasoning populated): a quoted/mentioned tag is NOT a leak.
    assert reasoning_split_failed("<think>x</think> ans", reasoning="real CoT") is False
    assert (
        reasoning_split_failed("wrap it in <think> tags", reasoning="real CoT") is False
    )
    # No block at all -> not a leak.
    assert reasoning_split_failed("clean answer", reasoning=None) is False
    # Lone (unbalanced) tag without a closing tag -> not a block -> not flagged
    # (the documented narrow blind spot; qwen split-failure leaves a full block).
    assert reasoning_split_failed("a lone <think> mention", reasoning=None) is False


# --------------------------------------------------------------------------- #
# Air-gap: public_only omits, visible_to_judge folds (judge only)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_public_only_omits_author_reasoning():
    """Default (think_visibility empty => public_only): the opponent's reasoning
    must NOT appear in a peer's prompt; the public argument must."""
    env = _env()
    kernel = _commit(KernelState(slot_index=0), "debater_a", _PUBLIC_A, _SECRET)
    state = State(answer="A", _kernel=kernel)
    state["prompt"] = [UserMessage(content="Q?")]

    b_prompt = await env.build_prompt(state, "debater_b", _SCHEDULE._slots[1])
    text = _all_text(b_prompt)
    assert _SECRET not in text  # air-gap holds
    assert _PUBLIC_A in text  # public argument is shown
    assert "debater_a" in text  # attributed


@pytest.mark.asyncio
async def test_visible_to_judge_exposes_reasoning_to_judge_only():
    """visible_to_judge: the judge SEES the author's reasoning (folded as labeled
    content); a peer debater under the same config does NOT."""
    env = _env(
        think_visibility={
            "debater_a": "visible_to_judge",
            "debater_b": "visible_to_judge",
        }
    )
    kernel = _commit(KernelState(slot_index=0), "debater_a", _PUBLIC_A, _SECRET)
    kernel = _commit(kernel, "debater_b", "B argues B", "b-secret")
    kernel = _commit(kernel, "debater_a", "A critique", "a-crit-secret")
    kernel = _commit(kernel, "debater_b", "B critique", "b-crit-secret")
    state = State(answer="A", _kernel=kernel)
    state["prompt"] = [UserMessage(content="Q?")]

    judge_prompt = await env.build_prompt(state, "judge", _SCHEDULE._slots[4])
    judge_text = _all_text(judge_prompt)
    assert _SECRET in judge_text  # reasoning reaches the judge (channel works)
    assert _PUBLIC_A in judge_text

    # A peer debater (debater_b viewing debater_a) must NOT see a's reasoning:
    # visible_to_judge exposes ONLY to the judge.
    b_prompt = await env.build_prompt(state, "debater_b", _SCHEDULE._slots[3])
    assert _SECRET not in _all_text(b_prompt)


# --------------------------------------------------------------------------- #
# Own-turn: structured reasoning_content, byte-matching the bridge anchor
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_own_turn_carries_reasoning_content_matching_bridge_anchor():
    """The own past-turn assistant message carries reasoning_content, and its
    exclude_none model_dump equals what parse_response_message stores on the
    bridge anchor -> the renderer-client prefix-match hits and reuses sampled
    tokens verbatim (no parity break)."""
    env = _env()
    kernel = _commit(KernelState(slot_index=0), "debater_a", _PUBLIC_A, _SECRET)
    kernel = _commit(kernel, "debater_b", "B argues", "b-secret")
    state = State(answer="A", _kernel=kernel)
    state["prompt"] = [UserMessage(content="Q?")]

    a_prompt = await env.build_prompt(state, "debater_a", _SCHEDULE._slots[2])
    own = [
        m
        for m in a_prompt
        if isinstance(m, AssistantMessage) and m.content == _PUBLIC_A
    ]
    assert len(own) == 1
    assert own[0].reasoning_content == _SECRET

    # Anchor parity: the own-turn replay must equal the completion that
    # parse_response_message reconstructs from the original response.
    anchor = (await parse_response_message(_response(_PUBLIC_A, _SECRET)))[0]
    assert own[0].model_dump(exclude_none=True) == anchor.model_dump(exclude_none=True)


# --------------------------------------------------------------------------- #
# Leak guard: a quoted tag with reasoning correctly split is NOT a rollout-kill
# --------------------------------------------------------------------------- #


def test_split_does_not_crash_on_quoted_think_tag_when_reasoning_split():
    """A debater that mentions the literal <think> tag in its public answer,
    while the renderer DID split its real reasoning into reasoning_content, must
    NOT raise (that would kill the whole rollout for a benign content event)."""
    env = _env()
    slot = _SCHEDULE._slots[0]
    resp = _response(
        content="Wrap your reasoning in <think> tags. My answer is A.",
        reasoning=_SECRET,  # the renderer split the REAL CoT here
    )
    visible, channels = env.split_response_channels(resp, "debater_a", slot)
    assert channels.private == _SECRET
    assert "<think>" in visible  # the quoted tag survives, unflagged


# --------------------------------------------------------------------------- #
# Fold + bridge: opponent folded into one user block (labeled), tail bridge-valid
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fold_merges_opponent_into_one_labeled_user_block():
    """The base fold collapses question+opponent+instruction into ONE user
    block (required for the renderer bridge). Provenance is preserved by the
    opponent_wrap LABELING inside that block, not by message boundaries — this
    is what fixes second-mover confusion."""
    env = _env()
    kernel = _commit(KernelState(slot_index=0), "debater_a", _PUBLIC_A, _SECRET)
    state = State(answer="A", _kernel=kernel)
    state["prompt"] = [UserMessage(content="Q?")]

    prompt = await env._prepare_prompt(state, "debater_b", _SCHEDULE._slots[1])
    user_msgs = [m for m in prompt if isinstance(m, UserMessage)]
    # Folded: a single leading user block (no opponent yet for b's propose vs a),
    # carrying the opponent's argument under an unmistakable attribution frame.
    assert len(user_msgs) == 1
    block = user_msgs[0].content
    assert _PUBLIC_A in block
    assert "debater_a" in block
    assert "written by your opponent, not you" in block  # opponent_wrap demarcation
    assert _SECRET not in block  # air-gap: a's reasoning is not in b's view


@pytest.mark.asyncio
async def test_continuation_tail_is_bridge_valid_after_fold():
    """The folded continuation tail is a SINGLE trailing user message, which the
    renderer-client bridge accepts (_is_valid_incremental_tail). Without the fold
    the tail is [user, user] and the bridge MISSES on every rebuttal turn -> full
    re-render -> own-reasoning drop. This is the regression test for that blocker."""
    env = _env()
    kernel = _commit(KernelState(slot_index=0), "debater_a", "A propose", "a-reason")
    kernel = _commit(kernel, "debater_b", "B propose", "b-reason")
    state = State(answer="A", _kernel=kernel)
    state["prompt"] = [UserMessage(content="Q?")]

    # debater_a's critique (slot 2) is a continuation off its own propose step.
    folded = await env._prepare_prompt(state, "debater_a", _SCHEDULE._slots[2])
    unfolded = await env.build_prompt(state, "debater_a", _SCHEDULE._slots[2])

    def tail_after_last_assistant(msgs):
        last_asst = max(
            (i for i, m in enumerate(msgs) if isinstance(m, AssistantMessage)),
            default=-1,
        )
        return [m.model_dump() for m in msgs[last_asst + 1 :]]

    folded_tail = tail_after_last_assistant(folded)
    unfolded_tail = tail_after_last_assistant(unfolded)

    # Folded: single trailing user -> bridge HIT.
    assert [m["role"] for m in folded_tail] == ["user"]
    assert _is_valid_incremental_tail(folded_tail) is True
    # Unfolded (what fold_user_messages=False produced): [user, user] -> bridge MISS.
    assert [m["role"] for m in unfolded_tail] == ["user", "user"]
    assert _is_valid_incremental_tail(unfolded_tail) is False
