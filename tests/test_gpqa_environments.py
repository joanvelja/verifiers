import importlib.util
import random
from pathlib import Path

import pytest
from datasets import Dataset

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
        monkeypatch.setattr(
            module,
            "load_dataset",
            lambda *args, **kwargs: Dataset.from_list([GPQA_ROW]),
        )
        env = module.load_environment(num_train_examples=1, num_eval_examples=0)
        row = env.get_dataset()[0]
    else:
        row = module._format_row(GPQA_ROW, random.Random(0))

    assert "task" not in row
    assert row["info"]["env_id"] == env_id
    if env_id == "gpqa_debate":
        assert "learner_seat" not in row["info"]


PCD4_SCHEDULE = [
    {"slot_id": 0, "agents": ["debater_a"], "phase": "propose"},
    {"slot_id": 1, "agents": ["debater_b"], "phase": "critique"},
    {"slot_id": 2, "agents": ["debater_a"], "phase": "rebuttal"},
    {"slot_id": 3, "agents": ["debater_b"], "phase": "counter_proposal"},
    {"slot_id": 4, "agents": ["judge"], "phase": "final"},
]


PCD4_FINAL_SCHEDULE = [
    {"slot_id": 0, "agents": ["debater_a"], "phase": "propose"},
    {"slot_id": 1, "agents": ["debater_b"], "phase": "critique"},
    {"slot_id": 2, "agents": ["debater_a"], "phase": "rebuttal"},
    {"slot_id": 3, "agents": ["debater_b"], "phase": "counter_critique"},
    {"slot_id": 4, "agents": ["debater_a", "debater_b"], "phase": "final"},
    {"slot_id": 5, "agents": ["judge"], "phase": "final"},
]

SEQUENTIAL4_SCHEDULE = [
    {"slot_id": 0, "agents": ["debater_a"], "phase": "propose"},
    {"slot_id": 1, "agents": ["debater_b"], "phase": "propose"},
    {"slot_id": 2, "agents": ["debater_a"], "phase": "critique"},
    {"slot_id": 3, "agents": ["debater_b"], "phase": "critique"},
    {"slot_id": 4, "agents": ["judge"], "phase": "final"},
]


@pytest.mark.parametrize(
    ("prompts_ref", "schedule"),
    [
        ("selfplay_oe_pcd4", PCD4_SCHEDULE),
        ("selfplay_oe_pcd4_final", PCD4_FINAL_SCHEDULE),
        ("selfplay_oe_sequential4", SEQUENTIAL4_SCHEDULE),
    ],
)
def test_gpqa_open_ended_debate_accepts_caller_prompts_ref(prompts_ref, schedule):
    """Caller-supplied prompts_ref must win over the wrapper default,
    for every pack the prime-rl kvsmoke configs actually pass.

    Regression: the wrapper used to hardcode prompts_ref="selfplay_oe" while
    forwarding **kwargs, so any caller passing prompts_ref hit
    ``TypeError: got multiple values for keyword argument 'prompts_ref'``.
    Construction is offline-safe: datasets are lazy builder closures and the
    grader client only needs its API key at the first grading request.
    """
    module = load_env_module(
        f"gpqa_open_ended_debate_prompts_ref_test_{prompts_ref}",
        "environments/gpqa_open_ended_debate/gpqa_open_ended_debate.py",
    )

    env = module.load_environment(prompts_ref=prompts_ref, schedule=schedule)

    assert env.prompts.source_ref.endswith(f"{prompts_ref}.yaml")


def _synthetic_gpqa_rows(n: int) -> Dataset:
    return Dataset.from_list(
        [
            {
                "Question": f"Question {i}?",
                "Correct Answer": f"correct {i}",
                "Incorrect Answer 1": f"wrong {i}.1",
                "Incorrect Answer 2": f"wrong {i}.2",
                "Incorrect Answer 3": f"wrong {i}.3",
                "Record ID": f"rec-{i}",
            }
            for i in range(n)
        ]
    )


def test_gpqa_debate_eval_holdout_is_disjoint_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
):
    """Eval must be a disjoint holdout of the train split, stable given seed.

    Regression: build_split used to slice raw[:n] of the same source split for
    both train and eval, so eval was always a strict prefix of train.
    """
    module = load_env_module(
        "gpqa_debate_holdout_test",
        "environments/gpqa_debate/gpqa_debate.py",
    )
    monkeypatch.setattr(module.vf, "ensure_keys", lambda keys: None)
    monkeypatch.setattr(
        module, "load_dataset", lambda *args, **kwargs: _synthetic_gpqa_rows(10)
    )

    def split_ids(seed: int) -> tuple[list[str], list[str]]:
        env = module.load_environment(
            num_train_examples=6, num_eval_examples=3, seed=seed
        )
        # _format_dataset relocates string ids to src_id and makes example_id
        # positional, so row identity lives in src_id (the GPQA Record ID).
        return (
            list(env.get_dataset()["src_id"]),
            list(env.get_eval_dataset()["src_id"]),
        )

    train_ids, eval_ids = split_ids(seed=7)
    assert len(train_ids) == 6
    assert len(eval_ids) == 3
    assert set(train_ids).isdisjoint(eval_ids)

    assert split_ids(seed=7) == (train_ids, eval_ids)
    assert split_ids(seed=8) != (train_ids, eval_ids)


def test_gpqa_debate_rejects_oversubscribed_holdout(monkeypatch: pytest.MonkeyPatch):
    module = load_env_module(
        "gpqa_debate_oversubscribed_test",
        "environments/gpqa_debate/gpqa_debate.py",
    )
    monkeypatch.setattr(module.vf, "ensure_keys", lambda keys: None)
    monkeypatch.setattr(
        module, "load_dataset", lambda *args, **kwargs: _synthetic_gpqa_rows(10)
    )

    env = module.load_environment(num_train_examples=8, num_eval_examples=3)
    with pytest.raises(ValueError, match="exceeds the 10 rows"):
        env.get_dataset()


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
    monkeypatch.setattr(
        module,
        "load_dataset",
        lambda *args, **kwargs: Dataset.from_list([GPQA_ROW]),
    )

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
