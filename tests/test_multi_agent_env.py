from typing import Any

import pytest

import verifiers as vf
from verifiers.clients import Client
from verifiers.envs.multi_agent_env import MultiAgentEnv
from verifiers.envs.multi_agent_kernel import StaticSchedule, TurnSlot
from verifiers.types import Messages, Response, ResponseMessage, SamplingArgs, State


class _RecordingClient(Client[None, Any, Any, Any]):
    def __init__(self) -> None:
        super().__init__(None)
        self.calls: list[dict[str, Any]] = []

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingArgs,
        tools=None,
        **kwargs: Any,
    ) -> Response:
        lineage_key = kwargs.get("lineage_key")
        state = kwargs["state"]
        self.calls.append(
            {
                "lineage_key": lineage_key,
                "prefix_candidate_indices": kwargs.get("prefix_candidate_indices"),
                "trajectory_len": len(state["trajectory"]),
            }
        )
        return Response(
            id=f"response-{len(self.calls)}",
            created=0,
            model=model,
            usage=None,
            message=ResponseMessage(
                content=f"response from {lineage_key}",
                reasoning_content=None,
                finish_reason="stop",
                is_truncated=False,
                tokens=None,
                tool_calls=None,
            ),
        )

    def setup_client(self, config):  # noqa: ANN001
        return None

    async def to_native_tool(self, tool):  # noqa: ANN001
        return tool

    async def to_native_prompt(self, messages):  # noqa: ANN001
        return messages, {}

    async def get_native_response(  # noqa: ANN001
        self, prompt, model, sampling_args, tools=None, **kwargs
    ):
        raise NotImplementedError

    async def raise_from_native_response(self, response):  # noqa: ANN001
        return None

    async def from_native_response(self, response):  # noqa: ANN001
        return response

    async def close(self) -> None:
        return None


class _MetricClient(_RecordingClient):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingArgs,
        tools=None,
        **kwargs: Any,
    ) -> Response:
        metrics = kwargs["state"].setdefault("metrics", {})
        metrics["client/test_metric"] = metrics.get("client/test_metric", 0.0) + 1.0
        if self.fail:
            raise vf.Error("simulated client failure")
        return await super().get_response(
            prompt, model, sampling_args, tools=tools, **kwargs
        )


class _LineageRoutingClient(_RecordingClient):
    def __init__(self, learner_by_example: dict[str, str]) -> None:
        super().__init__()
        self.learner_by_example = learner_by_example

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingArgs,
        tools=None,
        **kwargs: Any,
    ) -> Response:
        lineage_key = kwargs.get("lineage_key")
        state = kwargs["state"]
        learner = self.learner_by_example[state["example_id"]]
        route = "learner" if lineage_key == learner else "frozen"
        self.calls.append(
            {
                "lineage_key": lineage_key,
                "model": model,
                "route": route,
                "trajectory_len": len(state["trajectory"]),
            }
        )
        return Response(
            id=f"response-{len(self.calls)}",
            created=0,
            model=model,
            usage=None,
            message=ResponseMessage(
                content=f"{route} response from {lineage_key}",
                reasoning_content=None,
                finish_reason="stop",
                is_truncated=False,
                tokens=None,
                tool_calls=None,
            ),
        )


class _TwoRoundSimultaneousEnv(MultiAgentEnv):
    async def build_prompt(
        self, state: State, member_id: str, slot: TurnSlot
    ) -> Messages:
        own_turns = [
            utt for utt in state["_kernel"].transcript if utt.member_id == member_id
        ]
        messages: Messages = [{"role": "user", "content": f"{member_id}: start"}]
        for idx, utt in enumerate(own_turns):
            messages.append({"role": "assistant", "content": utt.raw_content})
            messages.append({"role": "user", "content": f"{member_id}: turn {idx + 1}"})
        return messages

    async def render_completion(self, state: State) -> None:
        state["completion"] = [
            msg for step in state["trajectory"] for msg in step["completion"]
        ]


@pytest.mark.asyncio
async def test_simultaneous_slots_pass_member_scoped_prefix_candidates() -> None:
    env = _TwoRoundSimultaneousEnv(
        schedule=StaticSchedule(
            (
                TurnSlot(slot_id=0, agents=("agent_a", "agent_b"), phase="round"),
                TurnSlot(slot_id=1, agents=("agent_a", "agent_b"), phase="round"),
            )
        ),
        members=["agent_a", "agent_b"],
        dataset=lambda: None,
    )
    client = _RecordingClient()

    await env.rollout(
        {
            "prompt": [{"role": "user", "content": "question"}],
            "answer": "answer",
            "example_id": "candidate-indices",
        },
        client,
        "test-model",
        {},
    )

    assert len(client.calls) == 4
    by_turn = {
        trajectory_len: {
            call["lineage_key"]: call["prefix_candidate_indices"]
            for call in client.calls
            if call["trajectory_len"] == trajectory_len
        }
        for trajectory_len in {call["trajectory_len"] for call in client.calls}
    }
    assert by_turn[0] == {"agent_a": (), "agent_b": ()}
    assert by_turn[2] == {"agent_a": (0,), "agent_b": (1,)}


@pytest.mark.asyncio
async def test_runtime_can_route_by_lineage_without_env_bindings() -> None:
    env = _TwoRoundSimultaneousEnv(
        schedule=StaticSchedule(
            (TurnSlot(slot_id=0, agents=("agent_a", "agent_b"), phase="round"),)
        ),
        members=["agent_a", "agent_b"],
        dataset=lambda: None,
    )
    client = _LineageRoutingClient(
        {
            "learner-a": "agent_a",
            "learner-b": "agent_b",
        }
    )

    for example_id in ("learner-a", "learner-b"):
        await env.rollout(
            {
                "prompt": [{"role": "user", "content": "question"}],
                "answer": "answer",
                "example_id": example_id,
            },
            client,
            "rollout-default-model",
            {},
        )

    assert [
        (call["lineage_key"], call["route"], call["model"]) for call in client.calls
    ] == [
        ("agent_a", "learner", "rollout-default-model"),
        ("agent_b", "frozen", "rollout-default-model"),
        ("agent_a", "frozen", "rollout-default-model"),
        ("agent_b", "learner", "rollout-default-model"),
    ]


@pytest.mark.asyncio
async def test_simultaneous_slot_merges_branch_metrics_on_publish() -> None:
    env = _TwoRoundSimultaneousEnv(
        schedule=StaticSchedule(
            (TurnSlot(slot_id=0, agents=("agent_a", "agent_b"), phase="round"),)
        ),
        members=["agent_a", "agent_b"],
        dataset=lambda: None,
    )
    client = _MetricClient()

    state = await env.rollout(
        {
            "prompt": [{"role": "user", "content": "question"}],
            "answer": "answer",
            "example_id": "metric-merge",
        },
        client,
        "test-model",
        {},
    )

    assert state["metrics"]["client/test_metric"] == 2.0
    assert len(state["trajectory"]) == 2


@pytest.mark.asyncio
async def test_failed_simultaneous_slot_does_not_leak_branch_metrics() -> None:
    env = _TwoRoundSimultaneousEnv(
        schedule=StaticSchedule(
            (TurnSlot(slot_id=0, agents=("agent_a", "agent_b"), phase="round"),)
        ),
        members=["agent_a", "agent_b"],
        dataset=lambda: None,
    )
    client = _MetricClient(fail=True)

    state = await env.rollout(
        {
            "prompt": [{"role": "user", "content": "question"}],
            "answer": "answer",
            "example_id": "metric-no-leak",
        },
        client,
        "test-model",
        {},
    )

    assert state["error"] is not None
    assert "client/test_metric" not in state["metrics"]
    assert state["trajectory"] == []
