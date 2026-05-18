"""Debate channel parsing."""

import re

from verifiers.envs.multi_agent_kernel import ContentChannels
from verifiers.errors import ContentParseError
from verifiers.types import Response

_NATIVE_THINK_ALT = r"think(?:ing)?"
_NATIVE_THINK_PROBE = re.compile(r"<\s*(?:think|thinking)\s*>", re.IGNORECASE)


def _compile_tag(alt: str) -> tuple[re.Pattern, re.Pattern]:
    return (
        re.compile(rf"<(?:{alt})\b[^>]*>", re.IGNORECASE),
        re.compile(rf"</(?:{alt})\s*>", re.IGNORECASE),
    )


def _extract_one_block(raw: str, alt: str, label: str) -> tuple[str, str | None]:
    opener_re, closer_re = _compile_tag(alt)
    openers = list(opener_re.finditer(raw))
    closers = list(closer_re.finditer(raw))

    if not openers and not closers:
        return raw, None

    if len(openers) != len(closers):
        raise ContentParseError(
            f"parse_channels: unbalanced {label} markup "
            f"({len(openers)} opener(s), {len(closers)} closer(s))"
        )

    if len(openers) > 1:
        if openers[1].start() < closers[0].end():
            raise ContentParseError(
                f"parse_channels: nested {label} tags are not allowed"
            )
        raise ContentParseError(
            f"parse_channels: multiple {label} blocks found "
            f"({len(openers)}); expected at most one"
        )

    opener = openers[0]
    closer = closers[0]
    if closer.start() < opener.end():
        raise ContentParseError(f"parse_channels: {label} closer appears before opener")

    inner = raw[opener.end() : closer.start()]
    residual = raw[: opener.start()] + raw[closer.end() :]
    return residual, inner


def parse_channels(raw: str, tag: str) -> ContentChannels:
    configured_alt = (
        _NATIVE_THINK_ALT if tag in ("think", "thinking") else re.escape(tag)
    )
    try:
        residual, configured_inner = _extract_one_block(raw, configured_alt, f"<{tag}>")
        if configured_alt != _NATIVE_THINK_ALT:
            residual, _ = _extract_one_block(
                residual, _NATIVE_THINK_ALT, "<think>/<thinking>"
            )
    except ContentParseError as exc:
        return ContentChannels(public="", private=None, parse_error=str(exc))

    private = configured_inner.strip() if configured_inner is not None else None
    return ContentChannels(public=residual.strip(), private=private or None)


def merge_provider_reasoning(raw: str, response: Response, think_tag: str) -> str:
    reasoning = _provider_reasoning(response)
    if not reasoning:
        return raw
    if _NATIVE_THINK_PROBE.search(raw) or f"<{think_tag}".lower() in raw.lower():
        return raw
    return f"<{think_tag}>\n{reasoning}\n</{think_tag}>\n\n{raw}"


def _provider_reasoning(response: Response) -> str | None:
    msg = response.message
    parts: list[str] = []

    reasoning_content = getattr(msg, "reasoning_content", None)
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        parts.append(reasoning_content.strip())

    for block in getattr(msg, "thinking_blocks", None) or []:
        text = getattr(block, "thinking", None) or (
            block.get("thinking") if isinstance(block, dict) else None
        )
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())

    return "\n\n".join(parts) if parts else None
