import os
from pathlib import Path

import pytest

from verifiers.envs.multi_agent_env import MultiAgentEnv
from verifiers.envs.multi_agent_kernel import StaticSchedule, TurnSlot
from verifiers.types import (
    ClientConfig,
    GenerationTarget,
    MemberGenerationPlan,
    Messages,
    RolloutOutput,
    State,
    UserMessage,
)


class LiveTwoMemberEnv(MultiAgentEnv):
    async def build_prompt(
        self, state: State, member_id: str, slot: TurnSlot
    ) -> Messages:
        return [UserMessage(content=f"{member_id}: return exactly the word ok.")]

    async def render_completion(self, state: State) -> None:
        state["completion"] = [
            msg for step in state["trajectory"] for msg in step["completion"]
        ]


def _load_live_env() -> None:
    env_file = os.getenv("VF_LIVE_API_ENV_FILE")
    if not env_file:
        return
    from dotenv import load_dotenv

    load_dotenv(Path(env_file))


def _live_client_config(client_type: str) -> ClientConfig:
    api_key_var = os.getenv("VF_LIVE_API_KEY_VAR")
    api_base_url = os.getenv("VF_LIVE_API_BASE_URL")
    if api_key_var is None:
        api_key_var = (
            "OPENAI_API_KEY" if os.getenv("OPENAI_API_KEY") else "PRIME_API_KEY"
        )
    if api_base_url is None and api_key_var == "OPENAI_API_KEY":
        api_base_url = "https://api.openai.com/v1"
    return ClientConfig(
        client_type=client_type,
        api_key_var=api_key_var,
        api_base_url=api_base_url or "https://api.pinference.ai/api/v1",
        timeout=float(os.getenv("VF_LIVE_API_TIMEOUT", "60")),
        max_retries=int(os.getenv("VF_LIVE_API_MAX_RETRIES", "1")),
    )


def _assert_member_route(
    output: RolloutOutput,
    member_id: str,
    *,
    client_type: str,
    model: str,
) -> None:
    step = next(
        step
        for step in output["trajectory"]
        if step["extras"]["member_id"] == member_id
    )
    generation = step["extras"]["generation"]
    assert generation["client"]["client_type"] == client_type
    assert generation["model"] == model
    assert step["response"]["usage"]["total_tokens"] > 0


@pytest.mark.asyncio
async def test_live_member_generation_routes_real_api_clients() -> None:
    if os.getenv("VF_LIVE_API_SMOKE") != "1":
        pytest.skip("Set VF_LIVE_API_SMOKE=1 to run the live API smoke.")

    _load_live_env()
    model = os.getenv("VF_LIVE_API_MODEL", "gpt-4o-mini")
    api_key_var = os.getenv("VF_LIVE_API_KEY_VAR")
    if api_key_var is None:
        api_key_var = (
            "OPENAI_API_KEY" if os.getenv("OPENAI_API_KEY") else "PRIME_API_KEY"
        )
    if not os.getenv(api_key_var):
        pytest.skip(f"{api_key_var} is not set.")

    env = LiveTwoMemberEnv(
        schedule=StaticSchedule(
            (
                TurnSlot(slot_id=0, agents=("debater_a",), phase="argue"),
                TurnSlot(slot_id=1, agents=("judge",), phase="judge"),
            )
        ),
        members=["debater_a", "judge"],
        dataset=lambda: None,
        score_rollouts=False,
    )
    generation = MemberGenerationPlan(
        members={
            "debater_a": GenerationTarget(
                client=_live_client_config("openai_chat_completions"),
                model=model,
                sampling_args={"max_tokens": 64},
            ),
            "judge": GenerationTarget(
                client=_live_client_config("openai_responses"),
                model=model,
                sampling_args={"max_tokens": 64},
            ),
        }
    )

    try:
        output = await env.run_rollout(
            {
                "prompt": [{"role": "user", "content": "live smoke"}],
                "example_id": "live-smoke",
            },
            client=_live_client_config("openai_chat_completions"),
            model="invalid-base-model-should-not-be-called",
            sampling_args={"max_tokens": 64},
            generation=generation,
            state_columns=["trajectory"],
        )
    finally:
        await env._teardown()

    assert output.get("error") is None
    assert len(output["trajectory"]) == 2
    _assert_member_route(
        output,
        "debater_a",
        client_type="openai_chat_completions",
        model=model,
    )
    _assert_member_route(
        output,
        "judge",
        client_type="openai_responses",
        model=model,
    )
