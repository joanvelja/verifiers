from collections.abc import Mapping
from typing import cast

from openai.types.chat import ChatCompletion
from renderers.base import Message as RendererMessage
from renderers.base import ToolSpec as RendererToolSpec
from renderers.streams import PreparedTurn

from verifiers.api_profile import ApiProfile
from verifiers.clients.client import AUTH_ERRORS
from verifiers.clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
    OpenAIChatMessages,
    OpenAIChatResponse,
    OpenAITool,
    handle_openai_overlong_prompt,
)
from verifiers.errors import Error, ModelError
from verifiers.types import Messages, Response, SamplingArgs, State, Tool
from verifiers.utils.rendered_streams import (
    DEFAULT_RENDERER_STREAM_ID,
    get_renderer_streams,
)


def _normalize_sampling_args(sampling_args: SamplingArgs) -> SamplingArgs:
    sampling_args = dict(sampling_args)
    if "max_tokens" in sampling_args:
        sampling_args["max_completion_tokens"] = sampling_args.pop("max_tokens")
    sampling_args["logprobs"] = True
    extra_body = dict(return_token_ids=True)
    if "extra_body" in sampling_args:
        sampling_args["extra_body"] = {
            **sampling_args["extra_body"],
            **extra_body,
        }
    else:
        sampling_args["extra_body"] = extra_body
    return {k: v for k, v in sampling_args.items() if v is not None}


def _has_multimodal_content(messages) -> bool:
    """Check if any message contains multimodal content (images, audio).

    Works with both plain dicts (OpenAIChatMessages) and Pydantic models
    (Messages stored in trajectory steps) since both support .get().
    """
    for msg in messages:
        content = msg.get("content") if hasattr(msg, "get") else None
        if isinstance(content, list):
            for part in content:
                if hasattr(part, "get") and part.get("type") in (
                    "image_url",
                    "input_audio",
                ):
                    return True
    return False


def _to_renderer_tools(
    oai_tools: list[OpenAITool] | None,
) -> list[RendererToolSpec] | None:
    if oai_tools is None:
        return None
    renderer_tools: list[RendererToolSpec] = []
    for tool in oai_tools:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        if function is None:
            function = getattr(tool, "function", None)
        if function is None:
            continue
        if isinstance(function, Mapping):
            renderer_tools.append(
                cast(RendererToolSpec, {str(k): v for k, v in function.items()})
            )
        elif hasattr(function, "model_dump"):
            renderer_tools.append(
                cast(RendererToolSpec, function.model_dump(exclude_none=True))
            )
    return renderer_tools or None


class OpenAIChatCompletionsTokenClient(OpenAIChatCompletionsClient):
    """Wrapper for custom vLLM route /v1/chat/completions/tokens via AsyncOpenAI client.

    Pinned to ``ApiProfile.VLLM_PERMISSIVE`` — this subclass exists solely
    for vLLM's extended /chat/completions/tokens route, so vLLM-only kwargs
    (top_k, min_p, cache_salt, return_token_ids) must pass through. The
    plain-client parent's strip logic is a no-op under this profile.
    """

    _default_profile: ApiProfile = ApiProfile.VLLM_PERMISSIVE

    def __init__(
        self,
        client_or_config,
        *,
        profile: ApiProfile | None = None,
        renderer: str | None = None,
        renderer_tokenizer_name_or_path: str | None = None,
    ) -> None:
        super().__init__(client_or_config, profile=profile)
        if self._config is not None:
            renderer = (
                renderer
                if renderer is not None
                else getattr(self._config, "renderer", None)
            )
            renderer_tokenizer_name_or_path = (
                renderer_tokenizer_name_or_path
                if renderer_tokenizer_name_or_path is not None
                else getattr(self._config, "renderer_tokenizer_name_or_path", None)
            )
        elif renderer is None:
            renderer = "auto"
        self.renderer = renderer
        self.renderer_tokenizer_name_or_path = renderer_tokenizer_name_or_path
        self._renderer = None

    def _get_renderer(self, model: str):
        if self._renderer is not None:
            return self._renderer
        if self.renderer is None:
            raise RuntimeError(
                "TITO token client requires renderer='auto' or a concrete renderer name"
            )
        from renderers.base import create_renderer, load_tokenizer

        tokenizer = load_tokenizer(self.renderer_tokenizer_name_or_path or model)
        self._renderer = create_renderer(tokenizer, renderer=self.renderer)
        return self._renderer

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> Response:
        try:
            extra_headers = self._build_state_headers(kwargs.get("state"))
            if extra_headers:
                kwargs["extra_headers"] = extra_headers

            native_prompt, extra_kwargs = await self.to_native_prompt(prompt)
            native_tools = await self.to_native_tools(tools)
            native_response, prepared, stream_id = await self._get_native_response(
                native_prompt,
                model,
                sampling_args,
                native_tools,
                **extra_kwargs,
                **kwargs,
            )
            await self.raise_from_native_response(native_response)
            response = await self.from_native_response(native_response)
            if prepared is not None:
                response.message.renderer_prepared_turn = prepared
                response.message.renderer_stream_id = stream_id
                if response.message.tokens is not None:
                    response.message.tokens.prompt_message_indices = list(
                        prepared.message_indices
                    )
            return response
        except Error:
            raise
        except AUTH_ERRORS:
            raise
        except Exception as e:
            raise ModelError from e

    def prepare_turn(
        self,
        state: State,
        prompt_messages: OpenAIChatMessages,
        oai_tools: list[OpenAITool] | None,
        *,
        model: str,
        stream_id: str,
    ) -> PreparedTurn:
        renderer = self._get_renderer(model)
        return get_renderer_streams(state).prepare_append(
            stream_id,
            cast(list[RendererMessage], list(prompt_messages)),
            renderer,
            tools=_to_renderer_tools(oai_tools),
        )

    @handle_openai_overlong_prompt
    async def _get_native_response(
        self,
        prompt: OpenAIChatMessages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[OpenAITool] | None = None,
        **kwargs,
    ) -> tuple[OpenAIChatResponse, PreparedTurn | None, str | None]:
        sampling_args = _normalize_sampling_args(sampling_args)
        lineage_key = kwargs.pop("lineage_key", None)
        state = cast(State, kwargs.pop("state"))
        extra_headers = kwargs.pop("extra_headers", None)
        has_multimodal = _has_multimodal_content(prompt) or any(
            _has_multimodal_content(step["prompt"]) for step in state["trajectory"]
        )
        if has_multimodal:
            if len(state["trajectory"]) == 0:
                response = await super().get_native_response(
                    prompt,
                    model,
                    sampling_args,
                    tools,
                    extra_headers=extra_headers,
                )
                return response, None, None
            raise RuntimeError(
                "TITO renderer path does not yet carry multimodal bridge state"
            )

        if lineage_key is None and isinstance(state, dict):
            lineage_key = state.get("_lineage_key")
        stream_id = lineage_key or DEFAULT_RENDERER_STREAM_ID
        prepared = self.prepare_turn(
            state,
            prompt,
            tools,
            model=model,
            stream_id=stream_id,
        )

        extra_body = sampling_args.pop("extra_body", {})
        body = dict(
            model=model,
            messages=prompt,
            tools=tools,
            tokens=list(prepared.prompt_ids),
            **sampling_args,
            **extra_body,
        )

        response = await self.client.post(
            "/chat/completions/tokens",
            body=body,
            cast_to=ChatCompletion,
            options={"headers": extra_headers} if extra_headers else {},
        )
        return response, prepared, stream_id

    async def get_native_response(
        self,
        prompt: OpenAIChatMessages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[OpenAITool] | None = None,
        **kwargs,
    ) -> OpenAIChatResponse:
        response, _, _ = await self._get_native_response(
            prompt,
            model,
            sampling_args,
            tools,
            **kwargs,
        )
        return response
