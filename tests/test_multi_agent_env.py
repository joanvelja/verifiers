import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

import verifiers as vf
from verifiers.clients import Client
from verifiers.envs.multi_agent_env import MultiAgentEnv
from verifiers.envs.multi_agent_kernel import StaticSchedule, TurnSlot
from verifiers.types import (
    ClientConfig,
    GenerationTarget,
    MemberGenerationPlan,
    Messages,
    Response,
    ResponseMessage,
    SamplingArgs,
    State,
    Usage,
)
from verifiers.utils.client_utils import resolve_client_config


class RecordingClient(Client[None, Any, Any, Any]):
    def __init__(self) -> None:
        super().__init__(None)
        self.calls: list[dict[str, Any]] = []

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingArgs,
        tools: Any = None,
        **kwargs: Any,
    ) -> Response:
        member_id = kwargs.get("member_id")
        state = kwargs["state"]
        self.calls.append(
            {
                "member_id": member_id,
                "model": model,
                "sampling_args": dict(sampling_args),
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
                content=f"response from {member_id}",
                reasoning_content=None,
                finish_reason="stop",
                is_truncated=False,
                tokens=None,
                tool_calls=None,
            ),
        )

    def setup_client(self, config: None) -> None:
        return None

    async def to_native_tool(self, tool: Any) -> Any:
        return tool

    async def to_native_prompt(
        self, messages: Messages
    ) -> tuple[Messages, dict[str, Any]]:
        return messages, {}

    async def get_native_response(
        self,
        prompt: Any,
        model: str,
        sampling_args: SamplingArgs,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    async def raise_from_native_response(self, response: Any) -> None:
        return None

    async def from_native_response(self, response: Response) -> Response:
        return response

    async def close(self) -> None:
        return None


class MetricClient(RecordingClient):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingArgs,
        tools: Any = None,
        **kwargs: Any,
    ) -> Response:
        metrics = kwargs["state"].setdefault("metrics", {})
        metrics["client/test_metric"] = metrics.get("client/test_metric", 0.0) + 1.0
        if self.fail:
            raise vf.Error("simulated client failure")
        return await super().get_response(
            prompt, model, sampling_args, tools=tools, **kwargs
        )


class UsageClient(RecordingClient):
    def __init__(self, *, completion_tokens: int) -> None:
        super().__init__()
        self.completion_tokens = completion_tokens

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingArgs,
        tools: Any = None,
        **kwargs: Any,
    ) -> Response:
        response = await super().get_response(
            prompt, model, sampling_args, tools=tools, **kwargs
        )
        response.usage = Usage(
            prompt_tokens=1,
            reasoning_tokens=0,
            completion_tokens=self.completion_tokens,
            total_tokens=1 + self.completion_tokens,
        )
        return response


class SlowClient(RecordingClient):
    def __init__(self, *, delay_seconds: float) -> None:
        super().__init__()
        self.delay_seconds = delay_seconds
        self.cancelled_members: set[str] = set()

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingArgs,
        tools: Any = None,
        **kwargs: Any,
    ) -> Response:
        member_id = kwargs.get("member_id")
        try:
            await asyncio.sleep(self.delay_seconds)
        except asyncio.CancelledError:
            if isinstance(member_id, str):
                self.cancelled_members.add(member_id)
            raise
        return await super().get_response(
            prompt, model, sampling_args, tools=tools, **kwargs
        )


class MemberRoutingClient(RecordingClient):
    def __init__(self, learner_by_example: dict[str, str]) -> None:
        super().__init__()
        self.learner_by_example = learner_by_example

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingArgs,
        tools: Any = None,
        **kwargs: Any,
    ) -> Response:
        member_id = kwargs.get("member_id")
        state = kwargs["state"]
        learner = self.learner_by_example[state["example_id"]]
        route = "learner" if member_id == learner else "frozen"
        self.calls.append(
            {
                "member_id": member_id,
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
                content=f"{route} response from {member_id}",
                reasoning_content=None,
                finish_reason="stop",
                is_truncated=False,
                tokens=None,
                tool_calls=None,
            ),
        )


class TwoRoundSimultaneousEnv(MultiAgentEnv):
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


def _env(slots: tuple[TurnSlot, ...], **kwargs: Any) -> TwoRoundSimultaneousEnv:
    return TwoRoundSimultaneousEnv(
        schedule=StaticSchedule(slots),
        members=["agent_a", "agent_b"],
        dataset=lambda: None,
        score_rollouts=False,
        **kwargs,
    )


def _plan_for_clients(
    agent_a_url: str = "http://agent-a/v1",
    agent_b_url: str = "http://agent-b/v1",
    *,
    model_prefix: str = "model",
) -> MemberGenerationPlan:
    return MemberGenerationPlan(
        members={
            "agent_a": GenerationTarget(
                client=ClientConfig(api_base_url=agent_a_url),
                model=f"{model_prefix}-a",
                sampling_args={"temperature": 0.1, "max_tokens": 11},
            ),
            "agent_b": GenerationTarget(
                client=ClientConfig(api_base_url=agent_b_url),
                model=f"{model_prefix}-b",
                sampling_args={"temperature": 0.2, "max_tokens": 22},
            ),
        }
    )


def _cache_generation_client(
    env: vf.Environment,
    plan: MemberGenerationPlan,
    member_id: str,
    client: Client[Any, Any, Any, Any],
) -> None:
    target = plan.target_for(member_id)
    key = resolve_client_config(target.client).model_dump_json()
    env._generation_client_cache[key] = client


def _cache_generation_clients(
    env: vf.Environment,
    plan: MemberGenerationPlan,
    clients: dict[str, RecordingClient],
) -> None:
    for member_id, client in clients.items():
        _cache_generation_client(env, plan, member_id, client)


@pytest.mark.asyncio
async def test_simultaneous_slots_pass_member_scoped_prefix_candidates() -> None:
    env = _env(
        (
            TurnSlot(slot_id=0, agents=("agent_a", "agent_b"), phase="round"),
            TurnSlot(slot_id=1, agents=("agent_a", "agent_b"), phase="round"),
        )
    )
    client = RecordingClient()

    assert env.is_multi_agent is True

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
            call["member_id"]: call["prefix_candidate_indices"]
            for call in client.calls
            if call["trajectory_len"] == trajectory_len
        }
        for trajectory_len in {call["trajectory_len"] for call in client.calls}
    }
    assert by_turn[0] == {"agent_a": (), "agent_b": ()}
    assert by_turn[2] == {"agent_a": (0,), "agent_b": (1,)}


@pytest.mark.asyncio
async def test_runtime_can_route_by_member_id_without_env_bindings() -> None:
    env = _env((TurnSlot(slot_id=0, agents=("agent_a", "agent_b"), phase="round"),))
    client = MemberRoutingClient({"learner-a": "agent_a", "learner-b": "agent_b"})

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
        (call["member_id"], call["route"], call["model"]) for call in client.calls
    ] == [
        ("agent_a", "learner", "rollout-default-model"),
        ("agent_b", "frozen", "rollout-default-model"),
        ("agent_a", "frozen", "rollout-default-model"),
        ("agent_b", "learner", "rollout-default-model"),
    ]


@pytest.mark.asyncio
async def test_member_generation_plan_routes_model_client_and_sampling_args() -> None:
    env = _env((TurnSlot(slot_id=0, agents=("agent_a", "agent_b"), phase="round"),))
    global_client = RecordingClient()
    agent_clients = {"agent_a": RecordingClient(), "agent_b": RecordingClient()}
    plan = _plan_for_clients()
    _cache_generation_clients(env, plan, agent_clients)

    output = await env.run_rollout(
        {
            "prompt": [{"role": "user", "content": "question"}],
            "answer": "answer",
            "example_id": "member-generation",
        },
        client=global_client,
        model="global-model",
        sampling_args={"temperature": 9.0},
        generation=plan,
        state_columns=["trajectory"],
    )

    assert global_client.calls == []
    assert agent_clients["agent_a"].calls == [
        {
            "member_id": "agent_a",
            "model": "model-a",
            "sampling_args": {"temperature": 0.1, "max_tokens": 11},
            "prefix_candidate_indices": (),
            "trajectory_len": 0,
        }
    ]
    assert agent_clients["agent_b"].calls == [
        {
            "member_id": "agent_b",
            "model": "model-b",
            "sampling_args": {"temperature": 0.2, "max_tokens": 22},
            "prefix_candidate_indices": (),
            "trajectory_len": 0,
        }
    ]
    by_member = {
        step["extras"]["member_id"]: step["extras"]["generation"]
        for step in output["trajectory"]
    }
    assert by_member["agent_a"]["model"] == "model-a"
    assert by_member["agent_a"]["sampling_args"] == {
        "temperature": 0.1,
        "max_tokens": 11,
    }
    assert by_member["agent_b"]["model"] == "model-b"
    assert by_member["agent_b"]["sampling_args"] == {
        "temperature": 0.2,
        "max_tokens": 22,
    }


@pytest.mark.asyncio
async def test_generate_grouped_scoring_routes_per_rollout_generation() -> None:
    env = _env((TurnSlot(slot_id=0, agents=("agent_a", "agent_b"), phase="round"),))
    plans: list[MemberGenerationPlan] = []
    inputs = []
    clients: dict[tuple[int, str], RecordingClient] = {}
    for idx, example_id in enumerate(("ex-0", "ex-0", "ex-1", "ex-1")):
        inputs.append(
            {
                "prompt": [{"role": "user", "content": f"question {idx}"}],
                "answer": "answer",
                "example_id": example_id,
            }
        )
        members = {}
        for member_id in ("agent_a", "agent_b"):
            base_url = f"http://rollout-{idx}-{member_id}/v1"
            clients[(idx, member_id)] = RecordingClient()
            members[member_id] = GenerationTarget(
                client=ClientConfig(api_base_url=base_url),
                model=f"model-{idx}-{member_id}",
                sampling_args={"temperature": float(idx)},
            )
        plan = MemberGenerationPlan(members=members)
        plans.append(plan)
        for member_id in ("agent_a", "agent_b"):
            _cache_generation_client(env, plan, member_id, clients[(idx, member_id)])

    outputs = await env.generate(
        inputs,
        client=RecordingClient(),
        model="global-model",
        sampling_args={},
        independent_scoring=False,
        generation=plans,
    )

    assert {output["rollout_id"] for output in outputs["outputs"]} == {
        "ex-0:0",
        "ex-0:1",
        "ex-1:0",
        "ex-1:1",
    }
    assert all("rollout_id" not in output["task"] for output in outputs["outputs"])
    for idx in range(4):
        for member_id in ("agent_a", "agent_b"):
            assert clients[(idx, member_id)].calls == [
                {
                    "member_id": member_id,
                    "model": f"model-{idx}-{member_id}",
                    "sampling_args": {"temperature": float(idx)},
                    "prefix_candidate_indices": (),
                    "trajectory_len": 0,
                }
            ]


@pytest.mark.asyncio
async def test_generate_resume_matches_saved_rollout_ids(tmp_path: Path) -> None:
    env = _env((TurnSlot(slot_id=0, agents=("agent_a", "agent_b"), phase="round"),))
    plans: list[MemberGenerationPlan] = []
    inputs = []
    clients: dict[tuple[int, str], RecordingClient] = {}
    for idx in range(3):
        inputs.append(
            {
                "prompt": [{"role": "user", "content": f"question {idx}"}],
                "answer": "answer",
                "example_id": "resume-example",
            }
        )
        members = {}
        for member_id in ("agent_a", "agent_b"):
            base_url = f"http://rollout-{idx}-{member_id}/v1"
            clients[(idx, member_id)] = RecordingClient()
            members[member_id] = GenerationTarget(
                client=ClientConfig(api_base_url=base_url),
                model=f"model-{idx}-{member_id}",
            )
        plan = MemberGenerationPlan(members=members)
        plans.append(plan)
        for member_id in ("agent_a", "agent_b"):
            _cache_generation_client(env, plan, member_id, clients[(idx, member_id)])

    results_path = tmp_path / "resume-results"
    results_path.mkdir()
    (results_path / "metadata.json").write_text(
        json.dumps(
            {
                "env_id": "",
                "model": "global-model",
                "num_examples": 1,
                "rollouts_per_example": 3,
            }
        )
    )
    saved_output = {
        "example_id": "resume-example",
        "rollout_id": "resume-example:1",
        "task": "",
        "prompt": [],
        "completion": [],
        "reward": 0.0,
        "metrics": {},
        "timing": {},
        "is_completed": True,
        "is_truncated": False,
        "sampling_args": {},
        "trajectory_id": "saved",
    }
    (results_path / "results.jsonl").write_text(json.dumps(saved_output) + "\n")

    await env.generate(
        inputs,
        client=RecordingClient(),
        model="global-model",
        sampling_args={},
        independent_scoring=True,
        generation=plans,
        results_path=results_path,
        on_start=lambda *args, **kwargs: None,
        on_progress=lambda *args, **kwargs: None,
        on_log=lambda *args, **kwargs: None,
    )

    for member_id in ("agent_a", "agent_b"):
        assert clients[(0, member_id)].calls
        assert clients[(1, member_id)].calls == []
        assert clients[(2, member_id)].calls


@pytest.mark.asyncio
async def test_generate_resume_requires_rollout_ids_for_specific_generation_plans(
    tmp_path: Path,
) -> None:
    env = _env((TurnSlot(slot_id=0, agents=("agent_a", "agent_b"), phase="round"),))
    inputs = [
        {
            "prompt": [{"role": "user", "content": "question 0"}],
            "answer": "answer",
            "example_id": "resume-example",
        },
        {
            "prompt": [{"role": "user", "content": "question 1"}],
            "answer": "answer",
            "example_id": "resume-example",
        },
    ]
    plans = [
        _plan_for_clients(
            "http://rollout-0-a/v1",
            "http://rollout-0-b/v1",
            model_prefix="model-0",
        ),
        _plan_for_clients(
            "http://rollout-1-a/v1",
            "http://rollout-1-b/v1",
            model_prefix="model-1",
        ),
    ]

    results_path = tmp_path / "legacy-resume-results"
    results_path.mkdir()
    (results_path / "metadata.json").write_text(
        json.dumps(
            {
                "env_id": "",
                "model": "global-model",
                "num_examples": 1,
                "rollouts_per_example": 2,
            }
        )
    )
    saved_output = {
        "example_id": "resume-example",
        "task": "",
        "prompt": [],
        "completion": [],
        "reward": 0.0,
        "metrics": {},
        "timing": {},
        "is_completed": True,
        "is_truncated": False,
        "sampling_args": {},
        "trajectory_id": "legacy-saved",
    }
    (results_path / "results.jsonl").write_text(json.dumps(saved_output) + "\n")

    with pytest.raises(ValueError, match="rollout_id"):
        await env.generate(
            inputs,
            client=RecordingClient(),
            model="global-model",
            sampling_args={},
            independent_scoring=True,
            generation=plans,
            results_path=results_path,
            on_start=lambda *args, **kwargs: None,
            on_progress=lambda *args, **kwargs: None,
            on_log=lambda *args, **kwargs: None,
        )


@pytest.mark.asyncio
async def test_member_generation_plan_fails_loud_on_missing_member() -> None:
    env = _env((TurnSlot(slot_id=0, agents=("agent_b",), phase="round"),))

    with pytest.raises(KeyError, match="agent_b"):
        await env.run_rollout(
            {
                "prompt": [{"role": "user", "content": "question"}],
                "answer": "answer",
                "example_id": "missing-member",
            },
            client=RecordingClient(),
            model="global-model",
            sampling_args={},
            generation=MemberGenerationPlan(
                members={
                    "agent_a": GenerationTarget(
                        client=ClientConfig(api_base_url="http://agent-a/v1"),
                        model="model-a",
                    )
                }
            ),
        )


@pytest.mark.asyncio
async def test_simultaneous_slot_merges_branch_metrics_on_publish() -> None:
    env = _env((TurnSlot(slot_id=0, agents=("agent_a", "agent_b"), phase="round"),))
    client = MetricClient()

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
    env = _env((TurnSlot(slot_id=0, agents=("agent_a", "agent_b"), phase="round"),))
    client = MetricClient(fail=True)

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


@pytest.mark.asyncio
async def test_max_total_completion_tokens_stops_multi_agent_rollout() -> None:
    env = TwoRoundSimultaneousEnv(
        schedule=StaticSchedule(
            (
                TurnSlot(slot_id=0, agents=("agent_a",), phase="round"),
                TurnSlot(slot_id=1, agents=("agent_a",), phase="round"),
                TurnSlot(slot_id=2, agents=("agent_a",), phase="round"),
            )
        ),
        members=["agent_a"],
        dataset=lambda: None,
        score_rollouts=False,
    )
    env.set_max_total_completion_tokens(3)
    client = UsageClient(completion_tokens=2)

    state = await env.rollout(
        {
            "prompt": [{"role": "user", "content": "question"}],
            "answer": "answer",
            "example_id": "completion-token-limit",
        },
        client,
        "test-model",
        {},
    )

    assert len(client.calls) == 2
    assert len(state["trajectory"]) == 2
    assert state["is_completed"] is True
    assert state["stop_condition"] == "max_total_completion_tokens_reached"
    assert state["usage"]["output_tokens"] == 4.0


@pytest.mark.asyncio
async def test_timeout_cancels_simultaneous_slot_without_partial_commit() -> None:
    env = _env(
        (TurnSlot(slot_id=0, agents=("agent_a", "agent_b"), phase="round"),),
        timeout_seconds=0.01,
    )
    client = SlowClient(delay_seconds=1.0)

    state = await env.rollout(
        {
            "prompt": [{"role": "user", "content": "question"}],
            "answer": "answer",
            "example_id": "simultaneous-timeout",
        },
        client,
        "test-model",
        {},
    )

    assert state["timed_out"] is True
    assert state["is_completed"] is True
    assert state["stop_condition"] == "timeout_reached"
    assert state["trajectory"] == []
    assert state["completion"] == []
    assert client.cancelled_members == {"agent_a", "agent_b"}
