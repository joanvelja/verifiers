from typing import Any, cast

import pytest
from renderers.base import ParsedResponse, RenderedTokens
from renderers.streams import CompletedResponse, StreamSet

from verifiers.clients.openai_chat_completions_token_client import (
    OpenAIChatCompletionsTokenClient,
)
from verifiers.types import (
    Response,
    ResponseMessage,
    ResponseTokens,
    State,
    Usage,
)
from verifiers.utils.rendered_streams import (
    RENDERER_STREAMS_STATE_KEY,
    commit_rendered_step,
)
from verifiers.utils.response_utils import parse_response_tokens


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
        self.calls.append(
            {
                "path": path,
                "body": body,
                "cast_to": cast_to,
                "kwargs": kwargs,
            }
        )
        return {"ok": True, "body": body}


class _FakeRenderer:
    def __init__(self) -> None:
        self.render_calls: list[dict[str, Any]] = []
        self.bridge_calls: list[dict[str, Any]] = []

    def render(self, messages, *, tools=None, add_generation_prompt=False):
        self.render_calls.append(
            {
                "messages": messages,
                "tools": tools,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        token_ids = _encode_messages(messages)
        if add_generation_prompt:
            token_ids.append(900)
        return RenderedTokens(
            token_ids=token_ids,
            message_indices=_message_indices(messages, token_ids),
        )

    def bridge_to_next_turn(
        self,
        previous_prompt_ids,
        previous_completion_ids,
        new_messages,
        *,
        tools=None,
    ):
        self.bridge_calls.append(
            {
                "previous_prompt_ids": previous_prompt_ids,
                "previous_completion_ids": previous_completion_ids,
                "new_messages": new_messages,
                "tools": tools,
            }
        )
        token_ids = (
            list(previous_prompt_ids)
            + list(previous_completion_ids)
            + _encode_messages(new_messages)
            + [900]
        )
        return RenderedTokens(
            token_ids=token_ids,
            message_indices=[-1]
            * (len(previous_prompt_ids) + len(previous_completion_ids))
            + _message_indices(new_messages, token_ids),
        )


class _RendererTokenClient(OpenAIChatCompletionsTokenClient):
    def __init__(self, recording_client: _RecordingClient, renderer: _FakeRenderer):
        super().__init__(recording_client)
        self._renderer = renderer
        self.renderer = "auto"

    async def to_native_prompt(self, messages):  # type: ignore[override]
        return cast(Any, messages), {}


class _ResponseTokenClient(_RendererTokenClient):
    async def raise_from_native_response(self, response):  # type: ignore[override]
        return None

    async def from_native_response(self, response):  # type: ignore[override]
        return Response(
            id="resp",
            created=0,
            model="test-model",
            usage=Usage(
                prompt_tokens=len(response["body"]["tokens"]),
                reasoning_tokens=0,
                completion_tokens=1,
                total_tokens=len(response["body"]["tokens"]) + 1,
            ),
            message=ResponseMessage(
                content="answer",
                finish_reason="stop",
                is_truncated=False,
                tokens=ResponseTokens(
                    prompt_ids=list(response["body"]["tokens"]),
                    prompt_mask=[0] * len(response["body"]["tokens"]),
                    completion_ids=[777],
                    completion_mask=[1],
                    completion_logprobs=[-0.2],
                ),
            ),
        )


def _encode_messages(messages):
    ids = []
    for message in messages:
        role_id = {"system": 10, "user": 20, "assistant": 30}.get(message["role"], 40)
        ids.extend([role_id, len(str(message.get("content", "")))])
    return ids


def _message_indices(messages, token_ids):
    indices = []
    for i, _ in enumerate(messages):
        indices.extend([i, i])
    if len(indices) < len(token_ids):
        indices.extend([-1] * (len(token_ids) - len(indices)))
    return indices


def _state(streams: StreamSet | None = None) -> State:
    return cast(
        State,
        {
            "model": "test-model",
            "trajectory": [],
            RENDERER_STREAMS_STATE_KEY: streams or StreamSet(),
        },
    )


@pytest.mark.asyncio
async def test_first_text_turn_uses_renderer_token_route():
    recording_client = _RecordingClient()
    renderer = _FakeRenderer()
    client = _RendererTokenClient(recording_client, renderer)
    prompt = cast(Any, [{"role": "user", "content": "u1"}])

    response = await client.get_native_response(
        prompt=prompt,
        model="test-model",
        sampling_args={},
        tools=None,
        state=_state(),
    )

    assert response["ok"] is True
    assert recording_client.calls[0]["path"] == "/chat/completions/tokens"
    assert recording_client.calls[0]["body"]["tokens"] == [20, 2, 900]
    assert renderer.render_calls == [
        {
            "messages": [{"role": "user", "content": "u1"}],
            "tools": None,
            "add_generation_prompt": True,
        }
    ]
    assert renderer.bridge_calls == []


@pytest.mark.asyncio
async def test_next_turn_uses_committed_stream_set_for_lineage():
    recording_client = _RecordingClient()
    renderer = _FakeRenderer()
    client = _RendererTokenClient(recording_client, renderer)
    prepared = StreamSet().prepare_append(
        "debater_a",
        [{"role": "user", "content": "u1"}],
        renderer,
    )
    streams = StreamSet().commit_response(
        "debater_a",
        prepared,
        CompletedResponse(
            completion_ids=[101],
            parsed=ParsedResponse(content="a1"),
        ),
    )
    renderer.render_calls.clear()
    prompt = cast(
        Any,
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ],
    )

    await client.get_native_response(
        prompt=prompt,
        model="test-model",
        sampling_args={},
        tools=None,
        state=_state(streams),
        lineage_key="debater_a",
    )

    assert recording_client.calls[0]["body"]["tokens"] == [20, 2, 900, 101, 20, 2, 900]
    assert renderer.render_calls == []
    assert renderer.bridge_calls == [
        {
            "previous_prompt_ids": [20, 2, 900],
            "previous_completion_ids": [101],
            "new_messages": [{"role": "user", "content": "u2"}],
            "tools": None,
        }
    ]


@pytest.mark.asyncio
async def test_get_response_carries_prepared_turn_for_commit():
    recording_client = _RecordingClient()
    renderer = _FakeRenderer()
    client = _ResponseTokenClient(recording_client, renderer)
    prompt = [{"role": "user", "content": "u1"}]

    response = await client.get_response(
        prompt=cast(Any, prompt),
        model="test-model",
        sampling_args={},
        state=_state(),
    )
    tokens = await parse_response_tokens(response)

    assert response.message.renderer_stream_id == "default"
    assert response.message.renderer_prepared_turn is not None
    assert response.message.tokens is not None
    assert response.message.tokens.prompt_message_indices == [0, 0, -1]
    assert tokens is not None
    assert tokens["prompt_message_indices"] == [0, 0, -1]

    streams = commit_rendered_step(
        StreamSet(),
        {
            "prompt": cast(Any, prompt),
            "completion": [{"role": "assistant", "content": "answer"}],
            "response": response,
            "tokens": tokens,
            "reward": None,
            "advantage": None,
            "is_truncated": False,
            "trajectory_id": "traj",
            "extras": {},
        },
    )
    stream = streams.get("default")
    assert stream is not None
    assert stream.prompt_message_indices == (0, 0, -1)
    assert stream.token_ids == (20, 2, 900, 777)
