"""API-shape profiles for OpenAI-compatible chat-completions endpoints.

A single ``OpenAIChatCompletionsClient`` talks to endpoints with different
accept-kwargs contracts: OpenAI/Azure reject vLLM-specific extras like
``top_k``, ``min_p``, ``cache_salt``; vLLM-compatible servers accept them.
Rather than sniff DNS (host allowlists decay the moment anyone adds a new
provider), the client declares its endpoint contract explicitly at
construction time via ``ApiProfile``.

The token client (vLLM-only) pins ``VLLM_PERMISSIVE``. The plain OpenAI
client defaults to ``OPENAI_STRICT`` — callers wrapping a vLLM server with
the plain client must pass ``profile=ApiProfile.VLLM_PERMISSIVE`` to opt
back in to vLLM extras. A one-shot logger.warning on the first stripped
key surfaces silent-degradation misconfigurations.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any


class ApiProfile(str, Enum):
    """Which request-shape contract the remote endpoint honors."""

    OPENAI_STRICT = "openai_strict"
    """Canonical OpenAI / Azure OpenAI chat-completions schema."""

    VLLM_PERMISSIVE = "vllm_permissive"
    """vLLM's OpenAI-compatible server: accepts top_k, min_p, cache_salt,
    return_token_ids, repetition_penalty, min_tokens, best_of via extra_body."""

    ANTHROPIC = "anthropic"
    """Anthropic messages API (kwargs normalized by AnthropicMessagesClient)."""

    NEMORL = "nemorl"
    """NeMo-RL chat-completions (kwargs normalized by NeMoRLChatCompletionsClient)."""


VLLM_ONLY_EXTRA_BODY_KEYS: frozenset[str] = frozenset(
    {
        "cache_salt",
        "top_k",
        "min_p",
        "return_token_ids",
        "repetition_penalty",
        "min_tokens",
        "best_of",
    }
)


def filter_sampling_args_for_profile(
    sampling_args: Mapping[str, Any],
    profile: ApiProfile,
) -> tuple[dict[str, Any], frozenset[str]]:
    """Return (filtered_sampling_args, stripped_key_set).

    ``stripped_key_set`` lets callers log on first strip without repeating
    per request. Pure / no-I/O — safe to call in the hot path.
    """
    out = dict(sampling_args)
    stripped: set[str] = set()

    if profile is ApiProfile.OPENAI_STRICT:
        # Top-level leaks: prime-rl's SamplingConfig.to_sampling_args places
        # top_k/min_p directly on the top-level kwargs dict, so strip there
        # as well as from extra_body.
        for k in VLLM_ONLY_EXTRA_BODY_KEYS:
            if k in out:
                out.pop(k)
                stripped.add(k)
        extra = out.get("extra_body")
        if isinstance(extra, Mapping):
            kept = {k: v for k, v in extra.items() if k not in VLLM_ONLY_EXTRA_BODY_KEYS}
            dropped = set(extra.keys()) - set(kept.keys())
            if dropped:
                stripped.update(dropped)
            if kept:
                out["extra_body"] = kept
            else:
                out.pop("extra_body", None)

    return out, frozenset(stripped)
