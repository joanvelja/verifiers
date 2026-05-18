import importlib.util
import random
from pathlib import Path

import pytest

import verifiers as vf
from verifiers.clients import Client
from verifiers.types import Messages, Response, ResponseMessage, SamplingArgs


ROOT = Path(__file__).resolve().parents[1]


def load_env_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GPQA_ROW = {
    "Question": "Which option is correct?",
    "Correct Answer": "correct",
    "Incorrect Answer 1": "wrong 1",
    "Incorrect Answer 2": "wrong 2",
    "Incorrect Answer 3": "wrong 3",
    "Record ID": "gpqa-test-row",
}


@pytest.mark.parametrize(
    ("module_name", "rel_path", "env_id"),
    [
        ("gpqa_rlvr_test", "environments/gpqa_rlvr/gpqa_rlvr.py", "gpqa_rlvr"),
        (
            "gpqa_consultancy_test",
            "environments/gpqa_consultancy/gpqa_consultancy.py",
            "gpqa_consultancy",
        ),
        (
            "gpqa_debate_test",
            "environments/gpqa_debate/gpqa_debate.py",
            "gpqa_debate",
        ),
    ],
)
def test_gpqa_rows_use_info_env_id_not_legacy_task(
    module_name: str, rel_path: str, env_id: str, monkeypatch: pytest.MonkeyPatch
):
    module = load_env_module(module_name, rel_path)
    if env_id == "gpqa_consultancy":
        row = module._format_row(GPQA_ROW, random.Random(0), 0.5)
    elif env_id == "gpqa_debate":
        monkeypatch.setattr(module.vf, "ensure_keys", lambda keys: None)
        monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: [GPQA_ROW])
        env = module.load_environment(num_train_examples=1, num_eval_examples=0)
        row = env.get_dataset()[0]
    else:
        row = module._format_row(GPQA_ROW, random.Random(0))

    assert "task" not in row
    assert row["info"]["env_id"] == env_id
    if env_id == "gpqa_debate":
        assert "learner_seat" not in row["info"]


class FakeJudgeClient:
    def __init__(self):
        self.prompt = None

    async def get_response(self, *, prompt, model, sampling_args):
        self.prompt = prompt
        return Response(
            id="judge",
            created=0,
            model=model,
            message=ResponseMessage(
                role="assistant",
                content="B",
                finish_reason="stop",
                is_truncated=False,
            ),
        )


@pytest.mark.asyncio
async def test_gpqa_consultancy_judge_uses_typed_messages():
    module = load_env_module(
        "gpqa_consultancy_judge_test",
        "environments/gpqa_consultancy/gpqa_consultancy.py",
    )
    judge_client = FakeJudgeClient()
    rubric = module._ConsultancyRubric(
        judge_client=judge_client,
        judge_model="judge-model",
        eval_reward_truth=True,
    )

    letter = await rubric._ask_judge(
        prompt=[vf.UserMessage(content="Q?\n\nA) no\nB) yes\n\nYou are a consultant.")],
        completion=[
            vf.AssistantMessage(content="<argument>B is best.</argument>"),
        ],
    )

    assert letter == "B"
    assert isinstance(judge_client.prompt[0], vf.UserMessage)


class TranscriptRecordingClient(Client[None, object, object, object]):
    def __init__(self):
        super().__init__(None)
        self.calls = []
        self.member_counts = {}

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingArgs,
        tools=None,
        **kwargs,
    ):
        member_id = kwargs["member_id"]
        count = self.member_counts.get(member_id, 0)
        self.member_counts[member_id] = count + 1
        self.calls.append({"member_id": member_id, "prompt": prompt})

        if member_id == "judge":
            content = "<reasoning>judge sees both critiques</reasoning><decision>debater_a</decision>"
        elif count == 0:
            answer = "A" if member_id == "debater_a" else "B"
            tag = "OPEN_A" if member_id == "debater_a" else "OPEN_B"
            content = f"{tag} <answer>{answer}</answer>"
        else:
            answer = "A" if member_id == "debater_a" else "B"
            tag = "CRITIQUE_A" if member_id == "debater_a" else "CRITIQUE_B"
            content = (
                f"{tag} "
                "<opponent_error>none found</opponent_error>"
                "<rebuttal>same answer</rebuttal>"
                f"<answer>{answer}</answer>"
            )

        return Response(
            id=f"response-{len(self.calls)}",
            created=0,
            model=model,
            message=ResponseMessage(
                role="assistant",
                content=content,
                finish_reason="stop",
                is_truncated=False,
            ),
        )

    def setup_client(self, config):
        return None

    async def to_native_tool(self, tool):
        return tool

    async def to_native_prompt(self, messages):
        return messages, {}

    async def get_native_response(
        self, prompt, model, sampling_args, tools=None, **kwargs
    ):
        raise NotImplementedError

    async def raise_from_native_response(self, response):
        return None

    async def from_native_response(self, response):
        return response

    async def close(self) -> None:
        return None


def _prompt_text(prompt: Messages) -> str:
    return "\n".join(str(message.content) for message in prompt)


@pytest.mark.asyncio
async def test_gpqa_debate_default_schedule_has_simultaneous_critique_barrier(
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_env_module(
        "gpqa_debate_schedule_test",
        "environments/gpqa_debate/gpqa_debate.py",
    )
    monkeypatch.setattr(module.vf, "ensure_keys", lambda keys: None)
    monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: [GPQA_ROW])

    env = module.load_environment(num_train_examples=1, num_eval_examples=0)
    slots = env.schedule._slots
    assert [(slot.slot_id, slot.agents, slot.phase) for slot in slots] == [
        (0, ("debater_a", "debater_b"), "propose"),
        (1, ("debater_a", "debater_b"), "critique"),
        (2, ("judge",), "final"),
    ]

    client = TranscriptRecordingClient()
    state = await env.run_rollout(
        env.get_dataset()[0],
        client,
        "test-model",
        {},
        state_columns=["trajectory"],
    )

    a_critique_prompt = next(
        call["prompt"]
        for call in client.calls
        if call["member_id"] == "debater_a" and "OPEN_B" in _prompt_text(call["prompt"])
    )
    b_critique_prompt = next(
        call["prompt"]
        for call in client.calls
        if call["member_id"] == "debater_b" and "OPEN_A" in _prompt_text(call["prompt"])
    )

    assert "CRITIQUE_B" not in _prompt_text(a_critique_prompt)
    assert "CRITIQUE_A" not in _prompt_text(b_critique_prompt)

    judge_prompt_text = _prompt_text(
        next(call["prompt"] for call in client.calls if call["member_id"] == "judge")
    )
    assert "CRITIQUE_A" in judge_prompt_text
    assert "CRITIQUE_B" in judge_prompt_text
    assert [step["extras"]["member_id"] for step in state["trajectory"]] == [
        "debater_a",
        "debater_b",
        "debater_a",
        "debater_b",
        "judge",
    ]
