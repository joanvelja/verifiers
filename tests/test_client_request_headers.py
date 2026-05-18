from typing import Any

import pytest

from verifiers.clients.client import Client, VERIFIERS_MEMBER_HEADER
from verifiers.types import (
    ClientConfig,
    Messages,
    Response,
    ResponseMessage,
    SamplingArgs,
    Tool,
)


class _HeaderRecordingClient(Client[None, Messages, dict[str, Any], Tool]):
    def __init__(self, config: ClientConfig):
        super().__init__(config)
        self.calls: list[dict[str, Any]] = []

    def setup_client(self, config: ClientConfig) -> None:
        return None

    async def to_native_tool(self, tool: Tool) -> Tool:
        return tool

    async def to_native_prompt(self, messages: Messages) -> tuple[Messages, dict]:
        return messages, {}

    async def get_native_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {}

    async def raise_from_native_response(self, response: dict[str, Any]) -> None:
        return None

    async def from_native_response(self, response: dict[str, Any]) -> Response:
        return Response(
            id="r",
            created=0,
            model="m",
            message=ResponseMessage(
                role="assistant",
                content="ok",
                finish_reason="stop",
                is_truncated=False,
            ),
        )

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_request_headers_merge_state_caller_and_member_id() -> None:
    client = _HeaderRecordingClient(
        ClientConfig(
            extra_headers_from_state={
                "X-Session-ID": "example_id",
                "X-Actor-Ref": "runtime.actor_ref",
            }
        )
    )

    await client.get_response(
        prompt=[],
        model="m",
        sampling_args={},
        state={
            "example_id": "ex-1",
            "runtime": {"actor_ref": "rollout-1"},
        },
        extra_headers={"X-Session-ID": "caller", "X-Caller": "yes"},
        member_id="debater_a",
    )

    headers = client.calls[0]["extra_headers"]
    assert headers == {
        "X-Session-ID": "caller",
        "X-Actor-Ref": "rollout-1",
        "X-Caller": "yes",
        VERIFIERS_MEMBER_HEADER: "debater_a",
    }
