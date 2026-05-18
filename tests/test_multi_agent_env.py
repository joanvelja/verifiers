import json
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
)


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


class _MemberRoutingClient(_RecordingClient):
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
        score_rollouts=False,
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
    env = _TwoRoundSimultaneousEnv(
        schedule=StaticSchedule(
            (TurnSlot(slot_id=0, agents=("agent_a", "agent_b"), phase="round"),)
        ),
        members=["agent_a", "agent_b"],
        dataset=lambda: None,
        score_rollouts=False,
    )
    client = _MemberRoutingClient(
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
        (call["member_id"], call["route"], call["model"]) for call in client.calls
    ] == [
        ("agent_a", "learner", "rollout-default-model"),
        ("agent_b", "frozen", "rollout-default-model"),
        ("agent_a", "frozen", "rollout-default-model"),
        ("agent_b", "learner", "rollout-default-model"),
    ]


@pytest.mark.asyncio
async def test_member_generation_plan_routes_model_client_and_sampling_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _TwoRoundSimultaneousEnv(
        schedule=StaticSchedule(
            (TurnSlot(slot_id=0, agents=("agent_a", "agent_b"), phase="round"),)
        ),
        members=["agent_a", "agent_b"],
        dataset=lambda: None,
        score_rollouts=False,
    )
    clients = {
        "http://agent-a/v1": _RecordingClient(),
        "http://agent-b/v1": _RecordingClient(),
        "http://global/v1": _RecordingClient(),
    }

    def resolve_by_base_url(config):  # noqa: ANN001
        if isinstance(config, _RecordingClient):
            return config
        return clients[config.api_base_url]

    monkeypatch.setattr(
        "verifiers.envs.environment.resolve_client", resolve_by_base_url
    )

    output = await env.run_rollout(
        {
            "prompt": [{"role": "user", "content": "question"}],
            "answer": "answer",
            "example_id": "member-generation",
        },
        client=ClientConfig(api_base_url="http://global/v1"),
        model="global-model",
        sampling_args={"temperature": 9.0},
        generation=MemberGenerationPlan(
            members={
                "agent_a": GenerationTarget(
                    client=ClientConfig(api_base_url="http://agent-a/v1"),
                    model="model-a",
                    sampling_args={"temperature": 0.1, "max_tokens": 11},
                ),
                "agent_b": GenerationTarget(
                    client=ClientConfig(api_base_url="http://agent-b/v1"),
                    model="model-b",
                    sampling_args={"temperature": 0.2, "max_tokens": 22},
                ),
            }
        ),
        state_columns=["trajectory"],
    )

    assert clients["http://global/v1"].calls == []
    assert clients["http://agent-a/v1"].calls == [
        {
            "member_id": "agent_a",
            "model": "model-a",
            "sampling_args": {"temperature": 0.1, "max_tokens": 11},
            "prefix_candidate_indices": (),
            "trajectory_len": 0,
        }
    ]
    assert clients["http://agent-b/v1"].calls == [
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
async def test_generate_grouped_scoring_routes_per_rollout_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _TwoRoundSimultaneousEnv(
        schedule=StaticSchedule(
            (TurnSlot(slot_id=0, agents=("agent_a", "agent_b"), phase="round"),)
        ),
        members=["agent_a", "agent_b"],
        dataset=lambda: None,
        score_rollouts=False,
    )
    clients = {"http://global/v1": _RecordingClient()}
    plans: list[MemberGenerationPlan] = []
    inputs = []
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
            clients[base_url] = _RecordingClient()
            members[member_id] = GenerationTarget(
                client=ClientConfig(api_base_url=base_url),
                model=f"model-{idx}-{member_id}",
                sampling_args={"temperature": float(idx)},
            )
        plans.append(MemberGenerationPlan(members=members))

    def resolve_by_base_url(config):  # noqa: ANN001
        if isinstance(config, _RecordingClient):
            return config
        return clients[config.api_base_url]

    monkeypatch.setattr(
        "verifiers.envs.environment.resolve_client", resolve_by_base_url
    )

    outputs = await env.generate(
        inputs,
        client=ClientConfig(api_base_url="http://global/v1"),
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
    assert clients["http://global/v1"].calls == []
    for idx in range(4):
        for member_id in ("agent_a", "agent_b"):
            calls = clients[f"http://rollout-{idx}-{member_id}/v1"].calls
            assert calls == [
                {
                    "member_id": member_id,
                    "model": f"model-{idx}-{member_id}",
                    "sampling_args": {"temperature": float(idx)},
                    "prefix_candidate_indices": (),
                    "trajectory_len": 0,
                }
            ]


@pytest.mark.asyncio
async def test_generate_resume_matches_saved_rollout_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    env = _TwoRoundSimultaneousEnv(
        schedule=StaticSchedule(
            (TurnSlot(slot_id=0, agents=("agent_a", "agent_b"), phase="round"),)
        ),
        members=["agent_a", "agent_b"],
        dataset=lambda: None,
        score_rollouts=False,
    )
    clients = {"http://global/v1": _RecordingClient()}
    plans: list[MemberGenerationPlan] = []
    inputs = []
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
            clients[base_url] = _RecordingClient()
            members[member_id] = GenerationTarget(
                client=ClientConfig(api_base_url=base_url),
                model=f"model-{idx}-{member_id}",
            )
        plans.append(MemberGenerationPlan(members=members))

    def resolve_by_base_url(config):  # noqa: ANN001
        if isinstance(config, _RecordingClient):
            return config
        return clients[config.api_base_url]

    monkeypatch.setattr(
        "verifiers.envs.environment.resolve_client", resolve_by_base_url
    )

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
        client=ClientConfig(api_base_url="http://global/v1"),
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
        assert clients[f"http://rollout-0-{member_id}/v1"].calls
        assert clients[f"http://rollout-1-{member_id}/v1"].calls == []
        assert clients[f"http://rollout-2-{member_id}/v1"].calls


@pytest.mark.asyncio
async def test_generate_resume_requires_rollout_ids_for_specific_generation_plans(
    tmp_path,
) -> None:
    env = _TwoRoundSimultaneousEnv(
        schedule=StaticSchedule(
            (TurnSlot(slot_id=0, agents=("agent_a", "agent_b"), phase="round"),)
        ),
        members=["agent_a", "agent_b"],
        dataset=lambda: None,
        score_rollouts=False,
    )
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
        MemberGenerationPlan(
            members={
                "agent_a": GenerationTarget(
                    client=ClientConfig(api_base_url="http://rollout-0-a/v1"),
                    model="model-0-a",
                ),
                "agent_b": GenerationTarget(
                    client=ClientConfig(api_base_url="http://rollout-0-b/v1"),
                    model="model-0-b",
                ),
            }
        ),
        MemberGenerationPlan(
            members={
                "agent_a": GenerationTarget(
                    client=ClientConfig(api_base_url="http://rollout-1-a/v1"),
                    model="model-1-a",
                ),
                "agent_b": GenerationTarget(
                    client=ClientConfig(api_base_url="http://rollout-1-b/v1"),
                    model="model-1-b",
                ),
            }
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
            client=_RecordingClient(),
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
async def test_member_generation_plan_fails_loud_on_missing_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _TwoRoundSimultaneousEnv(
        schedule=StaticSchedule(
            (TurnSlot(slot_id=0, agents=("agent_a", "agent_b"), phase="round"),)
        ),
        members=["agent_a", "agent_b"],
        dataset=lambda: None,
    )
    monkeypatch.setattr(
        "verifiers.envs.environment.resolve_client",
        lambda config: (
            config if isinstance(config, _RecordingClient) else _RecordingClient()
        ),
    )

    with pytest.raises(KeyError, match="agent_b"):
        await env.run_rollout(
            {
                "prompt": [{"role": "user", "content": "question"}],
                "answer": "answer",
                "example_id": "missing-member",
            },
            client=ClientConfig(api_base_url="http://global/v1"),
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
