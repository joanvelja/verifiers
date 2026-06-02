"""Prove-it: OpenRouter finish_reason='error' is no longer silently accepted.

A transient upstream-provider failure arrives as a 200 response with
choices[0].finish_reason == 'error'. The old client mapped that to
finish_reason=None and scored the (empty/truncated) verdict as if valid. Now
raise_from_native_response raises, and the client retries the single call up to
ClientConfig.max_retries before surfacing it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.errors import InvalidModelResponseError


def _msg(content: str = "x"):
    return SimpleNamespace(content=content, tool_calls=None, model_dump=lambda: {})


def _resp(finish_reason: str, content: str = "x"):
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason, message=_msg(content))]
    )


def test_raise_from_native_response_raises_on_error_finish_reason() -> None:
    # Reproduce target: the old code did NOT raise here (it accepted the
    # response and scored a non-verdict). After the fix it must raise.
    client = OpenAIChatCompletionsClient.__new__(OpenAIChatCompletionsClient)
    with pytest.raises(InvalidModelResponseError, match="finish_reason='error'"):
        asyncio.run(client.raise_from_native_response(_resp("error")))


def test_raise_from_native_response_accepts_stop() -> None:
    # A normal stop with content is unaffected.
    client = OpenAIChatCompletionsClient.__new__(OpenAIChatCompletionsClient)
    asyncio.run(client.raise_from_native_response(_resp("stop")))


def test_retry_decorator_retries_error_then_succeeds(monkeypatch) -> None:
    from verifiers.clients.openai_chat_completions_client import (
        retry_on_error_finish_reason,
    )

    async def _no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = {"n": 0}

    async def fake(self, *a, **k):
        calls["n"] += 1
        return _resp("error") if calls["n"] < 3 else _resp("stop")

    wrapped = retry_on_error_finish_reason(fake)
    self_ = SimpleNamespace(_config=SimpleNamespace(max_retries=5))
    resp = asyncio.run(wrapped(self_))
    assert calls["n"] == 3
    assert resp.choices[0].finish_reason == "stop"


def test_retry_decorator_gives_up_after_budget(monkeypatch) -> None:
    from verifiers.clients.openai_chat_completions_client import (
        retry_on_error_finish_reason,
    )

    async def _no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = {"n": 0}

    async def fake(self, *a, **k):
        calls["n"] += 1
        return _resp("error")

    wrapped = retry_on_error_finish_reason(fake)
    self_ = SimpleNamespace(_config=SimpleNamespace(max_retries=2))
    resp = asyncio.run(wrapped(self_))
    assert calls["n"] == 3  # max_retries + 1 total attempts
    # Still errored -> returned so raise_from_native_response surfaces it loudly.
    assert resp.choices[0].finish_reason == "error"
