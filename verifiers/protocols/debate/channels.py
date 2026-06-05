"""Debate reasoning channel (renderer-first).

The ``renderers`` package is the token authority. It splits the model's
chain-of-thought into the structured ``reasoning_content`` field at sampling
time (qwen35 token-splits the sampled completion at the ``</think>`` token id,
client-side, independent of any server reasoning parser). The env reads that
field VERBATIM — it never folds reasoning into inline ``<think>`` text and never
tag-parses it back out.

Air-gap (a debater's private reasoning must not reach the opponent or judge) is
enforced upstream by per-viewer message construction in :class:`DebateEnv`, not
here: own-turn replay keeps ``reasoning_content``; ``public_only`` cross-agent
views omit it; ``open``/``visible_to_judge`` views fold it back as the author's
labeled content. This module only (a) reads the structured reasoning and
(b) detects a leak when the renderer failed to split it.
"""

import re

from verifiers.types import Response

# A complete reasoning BLOCK surviving in the visible channel means the active
# renderer did NOT split reasoning into ``reasoning_content`` (a non-splitting
# renderer like gemma4/olmo3, an inline-only model, or no reasoning parser) —
# that would put private CoT in front of the opponent/judge. We key on a
# BALANCED ``<think>...</think>`` block (the qwen-family failure shape: a split
# failure leaves the full block in content), NOT a lone tag: a debater that
# legitimately QUOTES ``<think>`` in its public answer, while the renderer DID
# split its real reasoning, must not be treated as a leak.
_RESIDUAL_REASONING_BLOCK = re.compile(
    r"<\s*think(?:ing)?\s*>.*?</\s*think(?:ing)?\s*>", re.IGNORECASE | re.DOTALL
)


def structured_reasoning(response: Response) -> str | None:
    """Return the renderer's ``reasoning_content`` for ``response``, VERBATIM.

    The qwen35 renderer splits the sampled completion at the ``</think>`` token
    id into ``content`` vs ``reasoning_content`` (client-side, independent of
    any server reasoning parser). We return that field byte-for-byte — no
    ``.strip()``, and an empty string is returned as ``""`` not ``None`` — so
    the own-turn replay's ``AssistantMessage.reasoning_content`` is
    byte-identical to what ``parse_response_message`` stores on the
    renderer-client bridge anchor (``reasoning_content=response.message.
    reasoning_content``). Any divergence flips the prefix-match bridge
    hit↔miss.

    NO ``thinking_blocks`` fallback: the bridge anchor copies ``thinking_blocks``
    verbatim and leaves ``reasoning_content=None``, so synthesizing a
    ``reasoning_content`` from ``thinking_blocks`` here would DESYNC the bridge.
    A provider that carries reasoning only via ``thinking_blocks`` is out of
    scope for debate air-gap (the Qwen3.5 family populates ``reasoning_content``).
    """
    reasoning_content = getattr(response.message, "reasoning_content", None)
    if isinstance(reasoning_content, str):
        return reasoning_content
    return None


def reasoning_split_failed(visible: str, reasoning: str | None) -> bool:
    """True if the renderer did NOT split reasoning out of the visible channel.

    Leak signal: no structured reasoning was extracted (``reasoning`` falsy) AND
    a complete ``<think>...</think>`` block survives in the visible content. A
    lone/quoted tag with ``reasoning`` correctly populated is NOT a leak — the
    renderer split the real CoT and the model merely mentioned the token.
    Detects the qwen-family ``<think>``/``<thinking>`` conventions; other
    reasoning surfaces (gemma4's ``<|channel>thought``) are out of scope until
    their parsers populate ``reasoning_content``.
    """
    if reasoning:
        return False
    return bool(_RESIDUAL_REASONING_BLOCK.search(visible))
