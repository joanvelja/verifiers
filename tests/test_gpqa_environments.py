import importlib.util
import random
from pathlib import Path

import pytest

import verifiers as vf
from verifiers.types import Response, ResponseMessage


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
    module_name: str, rel_path: str, env_id: str
):
    module = load_env_module(module_name, rel_path)
    if env_id == "gpqa_consultancy":
        row = module._format_row(GPQA_ROW, random.Random(0), 0.5)
    elif env_id == "gpqa_debate":
        row = module._format_row(
            GPQA_ROW,
            random.Random(0),
            example_idx=0,
            seat_mode=None,
            pin=None,
            seat_rng=None,
        )
    else:
        row = module._format_row(GPQA_ROW, random.Random(0))

    assert "task" not in row
    assert row["info"]["env_id"] == env_id


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
