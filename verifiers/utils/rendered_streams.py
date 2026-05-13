from __future__ import annotations

from typing import cast

from renderers.base import ParsedResponse, ParsedToolCall, ToolCallParseStatus
from renderers.streams import CompletedResponse, PreparedTurn, StreamSet

from verifiers.types import Response, State, TrajectoryStep

RENDERER_STREAMS_STATE_KEY = "_renderer_streams"
DEFAULT_RENDERER_STREAM_ID = "default"


def get_renderer_streams(state: State) -> StreamSet:
    return cast(StreamSet, state.get(RENDERER_STREAMS_STATE_KEY, StreamSet()))


def commit_rendered_step(
    streams: StreamSet,
    step: TrajectoryStep,
    *,
    stream_id: str | None = None,
) -> StreamSet:
    response = step["response"]
    prepared = cast(PreparedTurn | None, response.message.renderer_prepared_turn)
    tokens = step["tokens"]
    if prepared is None or tokens is None:
        return streams

    return streams.commit_response(
        stream_id or response.message.renderer_stream_id or DEFAULT_RENDERER_STREAM_ID,
        prepared,
        CompletedResponse(
            completion_ids=tokens["completion_ids"],
            completion_logprobs=tokens["completion_logprobs"],
            parsed=_parsed_response(response),
        ),
    )


def _parsed_response(response: Response) -> ParsedResponse:
    message = response.message
    tool_calls = [
        ParsedToolCall(
            raw=tool_call.arguments,
            name=tool_call.name,
            arguments=tool_call.arguments,
            status=ToolCallParseStatus.OK,
            id=tool_call.id,
        )
        for tool_call in message.tool_calls or []
    ]
    return ParsedResponse(
        content=message.content if isinstance(message.content, str) else "",
        reasoning_content=message.reasoning_content,
        tool_calls=tool_calls,
    )
