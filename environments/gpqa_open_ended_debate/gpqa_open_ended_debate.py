"""GPQA open-ended (free-form) two-debater + judge — a preset over the generic
HF-dataset debate loader.

Same debate machinery as the MCQ ``gpqa_debate``, but the questions are
open-ended (no A-D choices), so accuracy is checked by the LLM grader instead of
the MCQ exact-match short-circuit. This is just ``hf_debate`` pinned to
``joanvelja/gpqa-open-ended`` + a free-form prompt pack (``selfplay_oe`` by
default; any ``selfplay_oe_*`` variant via ``prompts_ref``) + the OmniMath-style
``gpqa_oe`` grader (DeepSeek V4 Flash via OpenRouter, measurement-only,
air-gapped from the judge's winner decision — ground truth never enters the
loop).
"""

from verifiers.protocols.debate.env import DebateEnv
from verifiers.protocols.debate.hf import load_hf_debate_environment


def load_environment(
    subset: str = "diamond",
    prompts_ref: str = "selfplay_oe",
    judge_model: str = "deepseek/deepseek-v4-flash",
    judge_base_url: str = "https://openrouter.ai/api/v1",
    judge_api_key_var: str = "OPENROUTER_API_KEY",
    **kwargs: object,
) -> DebateEnv:
    return load_hf_debate_environment(
        dataset_name="joanvelja/gpqa-open-ended",
        dataset_subset=subset,
        task_type="open_ended",
        question_key="question",
        answer_key="answer",
        example_id_key="record_id",
        info_keys=["domain", "subdomain"],
        task_name="gpqa_open_ended_debate",
        prompts_ref=prompts_ref,
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        judge_api_key_var=judge_api_key_var,
        **kwargs,
    )
