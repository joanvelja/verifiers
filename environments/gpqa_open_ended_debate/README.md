# gpqa-open-ended-debate

GPQA **open-ended** (free-form answers, no A-D choices) under the two-debater +
judge protocol (`verifiers.protocols.debate`).

- **Dataset:** `joanvelja/gpqa-open-ended` (config `diamond`; the question is
  free-form and the reference `answer` is a free-form derivation).
- **Pack:** `selfplay_oe` by default (override with the `prompts_ref` env arg,
  e.g. any `selfplay_oe_*` variant plus its matching `schedule`) — identical
  structure to `selfplay`, but the `<answer>` field is free-form `str` (no
  enum), so `DebateRubric` routes accuracy through the **LLM grader** instead
  of the MCQ exact-match short-circuit.
- **Grader:** `gpqa_oe` judge pack (`_grader`/`_matcher`) — OmniMath-style:
  question-aware, instructed **not to solve the problem itself**, with few-shots
  spanning short atomic *and* long-form (fluent-but-wrong, concise-but-right)
  shapes. Defaults to `gpt-5.4-mini`.
- **Reward:** judge winner decision only (`debater_a` / `debater_b` / `tie`).
  The grader is a **measurement-only** accuracy diagnostic, air-gapped from
  reward — ground truth never enters the training loop.

```bash
uv run vf-eval gpqa_open_ended_debate \
  -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY \
  --api-client-type openai_chat_completions -n 2 -r 1 -t 4096 -s
```

`judge_model` / `judge_base_url` / `judge_api_key_var` are env args (default the
OpenAI `gpt-5.4-mini` grader).
