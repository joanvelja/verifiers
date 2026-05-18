from typing import Any, cast

import pytest

from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.clients.openai_chat_completions_token_client import (
    OpenAIChatCompletionsTokenClient,
)
from verifiers.api_profile import ApiProfile
from verifiers.types import State


class _NoopClient:
    base_url = "http://localhost:8000/v1"

    def with_options(self, **kwargs):  # noqa: ANN003
        return self


class _RecordingClient(_NoopClient):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def post(
        self, path: str, body: dict[str, Any], cast_to: type, **kwargs: Any
    ) -> Any:
        self.calls.append({"path": path, "body": body, "cast_to": cast_to})
        return {"ok": True, "path": path, "body": body}


class _CreateRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return {"ok": True, "kwargs": kwargs}


class _ChatRecorder:
    def __init__(self) -> None:
        self.completions = _CreateRecorder()


class _FallbackRecordingClient(_NoopClient):
    def __init__(self) -> None:
        self.chat = _ChatRecorder()


class _PromptIdTestClient(OpenAIChatCompletionsTokenClient):
    def __init__(self, full_prompt_ids: list[int]) -> None:
        super().__init__(_NoopClient())
        self._full_prompt_ids = full_prompt_ids
        self.tokenize_models: list[str] = []

    async def to_native_prompt(self, messages):  # type: ignore[override]
        return cast(Any, messages), {}

    async def tokenize(  # type: ignore[override]
        self,
        messages,
        tools,
        model,
        extra_kwargs: dict = {},
        **kwargs,
    ) -> list[int]:
        self.tokenize_models.append(model)
        if isinstance(messages, str):
            assert messages == "World!"
            return [777]

        if messages == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World!"},
        ]:
            assert extra_kwargs == {"add_generation_prompt": False}
            return [1, 777, 999]

        return self._full_prompt_ids


class _NoTokenizeClient(OpenAIChatCompletionsTokenClient):
    def __init__(self) -> None:
        super().__init__(_NoopClient())

    async def to_native_prompt(self, messages):  # type: ignore[override]
        return cast(Any, messages), {}

    async def tokenize(  # type: ignore[override]
        self,
        messages,
        tools,
        model,
        extra_kwargs: dict = {},
        **kwargs,
    ) -> list[int]:
        raise AssertionError("tokenize should not be called without a prefix match")


def _make_step(
    prompt: list[dict[str, str]],
    completion: list[dict[str, str]],
    prompt_ids: list[int],
    completion_ids: list[int],
    extras: dict[str, Any] | None = None,
    trajectory_id: str | None = None,
) -> dict[str, Any]:
    step = {
        "prompt": prompt,
        "completion": completion,
        "tokens": {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
        },
    }
    if extras is not None:
        step["extras"] = extras
    if trajectory_id is not None:
        step["trajectory_id"] = trajectory_id
    return step


@pytest.mark.asyncio
async def test_token_client_first_turn_fallback_preserves_return_token_ids() -> None:
    recording_client = _FallbackRecordingClient()
    client = OpenAIChatCompletionsTokenClient(recording_client)

    assert client.profile is ApiProfile.VLLM_PERMISSIVE

    await client.get_native_response(
        prompt=cast(Any, [{"role": "user", "content": "hello"}]),
        model="test-model",
        sampling_args={"top_k": 20, "min_p": 0.1},
        tools=None,
        state=cast(State, {"model": "test-model", "trajectory": []}),
    )

    call = recording_client.chat.completions.calls[0]
    assert call["top_k"] == 20
    assert call["min_p"] == 0.1
    assert call["extra_body"]["return_token_ids"] is True


@pytest.mark.asyncio
async def test_get_prompt_ids_uses_largest_message_prefix_match():
    client = _PromptIdTestClient(full_prompt_ids=[1, 2, 3, 4, 999, 5])
    state = cast(
        State,
        {
            "model": "test-model",
            "trajectory": [
                _make_step(
                    prompt=[{"role": "user", "content": "u1"}],
                    completion=[{"role": "assistant", "content": "a1"}],
                    prompt_ids=[1],
                    completion_ids=[2],
                ),
                _make_step(
                    prompt=[
                        {"role": "user", "content": "u1"},
                        {"role": "assistant", "content": "a1"},
                        {"role": "user", "content": "u2"},
                    ],
                    completion=[{"role": "assistant", "content": "a2"}],
                    prompt_ids=[1, 2, 3],
                    completion_ids=[4],
                ),
            ],
        },
    )
    prompt_messages = cast(
        Any,
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
        ],
    )

    prompt_ids = await client.get_prompt_ids(state, prompt_messages, oai_tools=None)

    assert prompt_ids == [1, 2, 3, 4, 999, 5]


@pytest.mark.asyncio
async def test_get_prompt_ids_filters_by_member_id():
    client = _PromptIdTestClient(full_prompt_ids=[1, 2, 3, 99, 5])
    prompt_messages = cast(
        Any,
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "next"},
        ],
    )
    state = cast(
        State,
        {
            "model": "test-model",
            "trajectory": [
                _make_step(
                    prompt=[{"role": "user", "content": "u"}],
                    completion=[{"role": "assistant", "content": "a"}],
                    prompt_ids=[7],
                    completion_ids=[8, 99],
                    extras={"member_id": "agent_b"},
                ),
                _make_step(
                    prompt=[{"role": "user", "content": "u"}],
                    completion=[{"role": "assistant", "content": "a"}],
                    prompt_ids=[1],
                    completion_ids=[2, 99],
                    extras={"member_id": "agent_a"},
                ),
            ],
        },
    )

    prompt_ids = await client.get_prompt_ids(
        state, prompt_messages, oai_tools=None, member_id="agent_a"
    )

    assert prompt_ids is not None
    assert prompt_ids[:3] == [1, 2, 99]


@pytest.mark.asyncio
async def test_get_prompt_ids_uses_candidate_indices_without_flat_scan():
    client = _PromptIdTestClient(full_prompt_ids=[1, 2, 99, 5])
    prompt_messages = cast(
        Any,
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "next"},
        ],
    )
    state = cast(
        State,
        {
            "model": "test-model",
            "trajectory": [
                _make_step(
                    prompt=[{"role": "user", "content": "u"}],
                    completion=[{"role": "assistant", "content": "a"}],
                    prompt_ids=[1],
                    completion_ids=[2, 99],
                    extras={"member_id": "agent_a"},
                ),
                _make_step(
                    prompt=[{"role": "user", "content": "u"}],
                    completion=[{"role": "assistant", "content": "a"}],
                    prompt_ids=[7],
                    completion_ids=[8, 99],
                    extras={"member_id": "agent_b"},
                ),
            ],
        },
    )

    prompt_ids = await client.get_prompt_ids(
        state,
        prompt_messages,
        oai_tools=None,
        prefix_candidate_indices=(0,),
    )

    assert prompt_ids is not None
    assert prompt_ids[:3] == [1, 2, 99]


@pytest.mark.asyncio
async def test_get_prompt_ids_tokenizes_bridge_with_routed_model():
    client = _PromptIdTestClient(full_prompt_ids=[1, 2, 99, 5])
    prompt_messages = cast(
        Any,
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "next"},
        ],
    )
    state = cast(
        State,
        {
            "model": "rollout-default-model",
            "trajectory": [
                _make_step(
                    prompt=[{"role": "user", "content": "u"}],
                    completion=[{"role": "assistant", "content": "a"}],
                    prompt_ids=[1],
                    completion_ids=[2, 99],
                    extras={"member_id": "agent_a"},
                ),
            ],
        },
    )

    prompt_ids = await client.get_prompt_ids(
        state,
        prompt_messages,
        oai_tools=None,
        member_id="agent_a",
        model="routed-member-model",
    )

    assert prompt_ids is not None
    assert client.tokenize_models == ["routed-member-model", "routed-member-model"]


@pytest.mark.asyncio
async def test_get_prompt_ids_returns_none_when_no_prefix_match():
    client = _NoTokenizeClient()
    state = cast(
        State,
        {
            "model": "test-model",
            "trajectory": [
                _make_step(
                    prompt=[{"role": "user", "content": "old"}],
                    completion=[{"role": "assistant", "content": "reply"}],
                    prompt_ids=[1],
                    completion_ids=[2],
                )
            ],
        },
    )

    prompt_ids = await client.get_prompt_ids(
        state,
        cast(Any, [{"role": "user", "content": "new"}]),
        oai_tools=None,
    )

    assert prompt_ids is None


@pytest.mark.asyncio
async def test_get_native_response_falls_back_to_super_when_no_prefix_match(
    monkeypatch: pytest.MonkeyPatch,
):
    client = OpenAIChatCompletionsTokenClient(_NoopClient())
    sentinel = {"source": "super"}
    calls: list[dict[str, Any]] = []

    async def fake_get_prompt_ids(  # noqa: ANN001
        self,
        state,
        prompt_messages,
        oai_tools,
        chat_template_kwargs=None,
        member_id=None,
        prefix_candidate_indices=None,
        model=None,
    ):
        return None

    async def fake_super_get_native_response(  # noqa: ANN001
        self,
        prompt,
        model,
        sampling_args,
        tools=None,
        **kwargs,
    ):
        calls.append(
            {
                "prompt": prompt,
                "model": model,
                "sampling_args": sampling_args,
                "tools": tools,
            }
        )
        return sentinel

    monkeypatch.setattr(
        OpenAIChatCompletionsTokenClient, "get_prompt_ids", fake_get_prompt_ids
    )
    monkeypatch.setattr(
        OpenAIChatCompletionsClient,
        "get_native_response",
        fake_super_get_native_response,
    )

    state = cast(
        State,
        {
            "model": "test-model",
            "trajectory": [
                _make_step(
                    prompt=[{"role": "user", "content": "u1"}],
                    completion=[{"role": "assistant", "content": "a1"}],
                    prompt_ids=[1],
                    completion_ids=[2],
                )
            ],
        },
    )
    prompt = cast(Any, [{"role": "user", "content": "u2"}])

    response = await client.get_native_response(
        prompt=prompt,
        model="test-model",
        sampling_args={},
        tools=None,
        state=state,
    )

    assert response is sentinel
    assert state["metrics"]["client/openai_chat_completions_token_tito_miss"] == 1.0
    assert len(calls) == 1
    assert calls[0]["prompt"] == prompt


@pytest.mark.asyncio
async def test_get_native_response_uses_token_route_when_prompt_ids_available(
    monkeypatch: pytest.MonkeyPatch,
):
    recording_client = _RecordingClient()
    client = OpenAIChatCompletionsTokenClient(recording_client)

    async def fake_get_prompt_ids(  # noqa: ANN001
        self,
        state,
        prompt_messages,
        oai_tools,
        chat_template_kwargs=None,
        member_id=None,
        prefix_candidate_indices=None,
        model=None,
    ):
        assert model == "routed-member-model"
        return [10, 20]

    monkeypatch.setattr(
        OpenAIChatCompletionsTokenClient, "get_prompt_ids", fake_get_prompt_ids
    )

    state = cast(
        State,
        {
            "model": "test-model",
            "trajectory": [
                _make_step(
                    prompt=[{"role": "user", "content": "u1"}],
                    completion=[{"role": "assistant", "content": "a1"}],
                    prompt_ids=[1],
                    completion_ids=[2],
                )
            ],
        },
    )
    prompt = cast(Any, [{"role": "user", "content": "u2"}])

    response = await client.get_native_response(
        prompt=prompt,
        model="routed-member-model",
        sampling_args={},
        tools=None,
        state=state,
    )

    assert response["ok"] is True
    assert state["metrics"]["client/openai_chat_completions_token_tito_hit"] == 1.0
    assert len(recording_client.calls) == 1
    assert recording_client.calls[0]["path"] == "/chat/completions/tokens"
    assert recording_client.calls[0]["body"]["tokens"] == [10, 20]
