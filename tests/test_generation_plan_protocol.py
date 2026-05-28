from __future__ import annotations

from typing import Any

import pytest
from datasets import Dataset
from verifiers.clients.client import Client
from verifiers.envs.environment import Environment
from verifiers.envs.request_context import ModelRequestContext
from verifiers.serve.client.env_client import EnvClient
from verifiers.serve.types import (
    HealthRequest,
    HealthResponse,
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
)
from verifiers.types import (
    ClientConfig,
    GenerationTarget,
    MemberGenerationPlan,
    RolloutInput,
    RolloutOutput,
    SamplingArgs,
    State,
)
from verifiers.utils.client_utils import resolve_client_config


def _plan(model: str = "member-model") -> MemberGenerationPlan:
    return MemberGenerationPlan(
        members={
            "debater_a": GenerationTarget(
                client=ClientConfig(api_base_url="http://localhost:8000/v1"),
                model=model,
                sampling_args={"temperature": 0.2},
            )
        }
    )


def _output() -> RolloutOutput:
    return RolloutOutput(
        example_id=0,
        prompt=None,
        completion=None,
        reward=0.0,
        timing={},
        is_completed=True,
        is_truncated=False,
        metrics={},
    )


def test_run_rollout_request_roundtrips_generation_plan() -> None:
    plan = _plan()
    request = RunRolloutRequest(
        input={"prompt": [], "example_id": 0},
        client_config=ClientConfig(api_base_url="http://localhost:8001/v1"),
        model="base",
        sampling_args={"max_tokens": 8},
        max_retries=1,
        state_columns=["member_generation"],
        generation=plan,
    )

    hydrated = RunRolloutRequest.model_validate(request.model_dump(mode="python"))

    assert hydrated.generation is not None
    assert hydrated.generation.target_for("debater_a").model == "member-model"


def test_run_group_request_roundtrips_list_generation_plan() -> None:
    plans = [_plan("a"), _plan("b")]
    request = RunGroupRequest(
        group_inputs=[
            {"prompt": [], "example_id": 0},
            {"prompt": [], "example_id": 0},
        ],
        client_config=ClientConfig(api_base_url="http://localhost:8001/v1"),
        model="base",
        sampling_args={"max_tokens": 8},
        max_retries=1,
        state_columns=["member_generation"],
        generation=plans,
    )

    hydrated = RunGroupRequest.model_validate(request.model_dump(mode="python"))

    assert isinstance(hydrated.generation, list)
    assert [plan.target_for("debater_a").model for plan in hydrated.generation] == [
        "a",
        "b",
    ]


class CapturingEnvClient(EnvClient):
    def __init__(self) -> None:
        super().__init__(address="inproc://test")
        self.rollout_request: RunRolloutRequest | None = None
        self.group_request: RunGroupRequest | None = None

    async def wait_for_server_startup(self, timeout: float | None = None) -> None:
        return None

    async def handle_health_request(
        self, request: HealthRequest, timeout: float | None
    ) -> HealthResponse:
        return HealthResponse()

    async def handle_run_rollout_request(
        self, request: RunRolloutRequest, timeout: float | None
    ) -> RunRolloutResponse:
        self.rollout_request = request
        return RunRolloutResponse(output=_output())

    async def handle_run_group_request(
        self, request: RunGroupRequest, timeout: float | None
    ) -> RunGroupResponse:
        self.group_request = request
        return RunGroupResponse(outputs=[])

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_env_client_builds_rollout_request_with_generation() -> None:
    client = CapturingEnvClient()
    plan = _plan()

    await client.run_rollout(
        input={"prompt": [], "example_id": 0},
        client_config=ClientConfig(api_base_url="http://localhost:8001/v1"),
        model="base",
        sampling_args={},
        generation=plan,
    )

    assert client.rollout_request is not None
    assert client.rollout_request.generation == plan


@pytest.mark.asyncio
async def test_env_client_builds_group_request_with_generation() -> None:
    client = CapturingEnvClient()
    plans = [_plan("a"), _plan("b")]

    await client.run_group(
        group_inputs=[
            {"prompt": [], "example_id": 0},
            {"prompt": [], "example_id": 0},
        ],
        client_config=ClientConfig(api_base_url="http://localhost:8001/v1"),
        model="base",
        sampling_args={},
        generation=plans,
    )

    assert client.group_request is not None
    assert client.group_request.generation == plans


class GenerationCaptureEnvironment(Environment):
    async def rollout(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        state = await self.init_state(input, client, model, sampling_args)
        state["is_completed"] = True
        return state


def _env() -> GenerationCaptureEnvironment:
    return GenerationCaptureEnvironment(
        dataset=Dataset.from_list([{"prompt": [], "example_id": 0}]),
        score_rollouts=False,
    )


def _cache_generation_client(
    env: Environment,
    plan: MemberGenerationPlan,
    member_id: str,
    client: Client,
) -> None:
    target = plan.target_for(member_id)
    key = resolve_client_config(target.client).model_dump_json()
    env._generation_client_cache[key] = client


@pytest.mark.asyncio
async def test_local_run_rollout_sets_member_generation_state(mock_client) -> None:
    env = _env()
    output = await env.run_rollout(
        input={"prompt": [], "example_id": 0},
        client=mock_client,
        model="base",
        sampling_args={},
        state_columns=["member_generation"],
        generation=_plan("learner"),
    )

    assert output["member_generation"].target_for("debater_a").model == "learner"


@pytest.mark.asyncio
async def test_local_run_group_sets_member_generation_per_rollout(mock_client) -> None:
    env = _env()
    outputs = await env.run_group(
        group_inputs=[
            {"prompt": [], "example_id": 0},
            {"prompt": [], "example_id": 0},
        ],
        client=mock_client,
        model="base",
        sampling_args={},
        state_columns=["member_generation"],
        generation=[_plan("a"), _plan("b")],
    )

    assert [
        output["member_generation"].target_for("debater_a").model for output in outputs
    ] == ["a", "b"]


@pytest.mark.asyncio
async def test_local_run_group_rejects_partial_generation_group(mock_client) -> None:
    env = _env()

    with pytest.raises(ValueError, match="all-or-none"):
        await env.run_group(
            group_inputs=[
                {"prompt": [], "example_id": 0},
                {"prompt": [], "example_id": 0},
            ],
            client=mock_client,
            model="base",
            sampling_args={},
            generation=[_plan("a"), None],
        )


@pytest.mark.asyncio
async def test_get_model_response_uses_member_generation_target(mock_client) -> None:
    env = _env()
    state = await env.init_state(
        {"prompt": [], "example_id": 0},
        mock_client,
        "base-model",
        {"temperature": 1.0},
    )
    plan = _plan("member-model")
    state["member_generation"] = plan
    target_client = type(mock_client)()
    _cache_generation_client(env, plan, "debater_a", target_client)

    await env.get_model_response(
        state=state,
        prompt=[],
        request_context=ModelRequestContext(
            member_id="debater_a", prefix_candidate_indices=(0,)
        ),
    )

    assert mock_client.call_count == 0
    assert target_client.call_count == 1
    assert target_client.last_call_kwargs["model"] == "member-model"
    assert target_client.last_call_kwargs["sampling_args"] == {"temperature": 0.2}
    assert target_client.last_call_kwargs["member_id"] == "debater_a"
    assert target_client.last_call_kwargs["prefix_candidate_indices"] == (0,)


@pytest.mark.asyncio
async def test_get_model_response_without_member_context_uses_base_client(
    mock_client,
) -> None:
    env = _env()
    state = await env.init_state(
        {"prompt": [], "example_id": 0},
        mock_client,
        "base-model",
        {"temperature": 1.0},
    )
    state["member_generation"] = _plan("member-model")

    await env.get_model_response(state=state, prompt=[])

    assert mock_client.call_count == 1
    assert mock_client.last_call_kwargs["model"] == "base-model"
    assert mock_client.last_call_kwargs["sampling_args"] == {"temperature": 1.0}
    assert "member_id" not in mock_client.last_call_kwargs
    assert "prefix_candidate_indices" not in mock_client.last_call_kwargs


@pytest.mark.asyncio
async def test_get_model_response_rejects_unknown_member(mock_client) -> None:
    env = _env()
    state = await env.init_state(
        {"prompt": [], "example_id": 0}, mock_client, "base-model", {}
    )
    state["member_generation"] = _plan("member-model")

    with pytest.raises(KeyError, match="judge"):
        await env.get_model_response(
            state=state,
            prompt=[],
            request_context=ModelRequestContext(member_id="judge"),
        )


@pytest.mark.asyncio
async def test_generation_target_clients_are_cached_and_torn_down(mock_client) -> None:
    env = _env()
    state = await env.init_state(
        {"prompt": [], "example_id": 0}, mock_client, "base-model", {}
    )
    plan = _plan("member-model")
    state["member_generation"] = plan
    target_client = type(mock_client)()
    _cache_generation_client(env, plan, "debater_a", target_client)

    await env.get_model_response(
        state=state,
        prompt=[],
        request_context=ModelRequestContext(member_id="debater_a"),
    )
    await env.get_model_response(
        state=state,
        prompt=[],
        request_context=ModelRequestContext(member_id="debater_a"),
    )
    await env._teardown()

    assert target_client.call_count == 2
    assert env._generation_client_cache == {}


_ = Any
