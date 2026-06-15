"""hf_debate — generic Hugging Face dataset, two-debater + judge environment.

Thin CLI surface over ``verifiers.protocols.debate.hf.load_hf_debate_environment``,
where the full parameter surface (dataset spec, column-map, ``task_type``, judge
wiring) and behaviour are documented. The dataset-adapter logic lives once in the
library so task-specific presets (e.g. ``gpqa_open_ended_debate``) can share it
without one env package importing another.

Examples::

    # MCQ (no judge key needed — enum exact-match):
    vf-eval hf_debate --env-args '{"dataset_name": "Idavidrein/gpqa",
        "dataset_subset": "gpqa_diamond", "task_type": "mcq",
        "choices_key": "choices", "prompts_ref": "selfplay"}'

    # Open-ended (LLM grader — GPQA preset defaults use OPENROUTER_API_KEY):
    vf-eval hf_debate --env-args '{"dataset_name": "joanvelja/gpqa-open-ended",
        "task_type": "open_ended", "question_key": "question",
        "answer_key": "answer", "prompts_ref": "selfplay_oe",
        "judge_model": "deepseek/deepseek-v4-flash",
        "judge_base_url": "https://openrouter.ai/api/v1",
        "judge_api_key_var": "OPENROUTER_API_KEY"}'
"""

from verifiers.protocols.debate.hf import (  # noqa: F401
    load_hf_debate_environment as load_environment,
)
