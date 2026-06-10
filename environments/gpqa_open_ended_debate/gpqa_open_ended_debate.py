"""GPQA open-ended (free-form) two-debater + judge — a preset over the generic
HF-dataset debate loader.

Same debate machinery as the MCQ ``gpqa_debate``, but the questions are
open-ended (no A-D choices), so accuracy is checked by the LLM grader instead of
the MCQ exact-match short-circuit. This is just ``hf_debate`` pinned to
``joanvelja/gpqa-open-ended`` + the free-form ``selfplay_oe`` pack + the
OmniMath-style ``gpqa_oe`` grader (gpt-5.4-mini, measurement-only, air-gapped
from the judge's winner decision — ground truth never enters the loop).
"""

from verifiers.protocols.debate.env import DebateEnv
from verifiers.protocols.debate.hf import load_hf_debate_environment


def load_environment(
    subset: str = "diamond",
    judge_model: str = "gpt-5.4-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
    **kwargs: object,
) -> DebateEnv:
    return load_hf_debate_environment(
        dataset_name="joanvelja/gpqa-open-ended",
        dataset_subset=subset,
        eval_dataset_split="train",
        task_type="open_ended",
        question_key="question",
        answer_key="answer",
        example_id_key="record_id",
        info_keys=["domain", "subdomain"],
        task_name="gpqa_open_ended_debate",
        prompts_ref="selfplay_oe",
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        judge_api_key_var=judge_api_key_var,
        **kwargs,
    )
