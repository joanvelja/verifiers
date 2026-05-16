# gpqa-debate

### Overview
- **Environment ID**: `gpqa-debate`
- **Short description**: GPQA Diamond / Main under a two-debater plus judge debate protocol.
- **Tags**: gpqa, debate, multi-agent, eval

### Datasets
- **Primary dataset(s)**: `Idavidrein/gpqa`
- **Source links**: [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa)
- **Splits**: Uses the Hugging Face `train` split for both train and eval builders. Set `subset` to `gpqa_diamond` or `gpqa_main`.

### Task
- **Type**: multi-agent debate
- **Protocol**: two debaters answer and critique; a neutral judge emits the final decision.
- **Rubric overview**: `DebateRubric` produces member-attributed rewards through `MARScore`: winning debater +1, losing debater -1, judge 0, tie/no decision 0.

### Requirements
- `HF_TOKEN` is required for the gated GPQA dataset.
- This environment depends on the multi-agent runtime in `verifiers[renderers]>=0.1.15.dev7`. Use a local `verifiers` checkout or a published/pinned version that includes `MultiAgentEnv`, `DebateEnv`, `MultiAgentRubric`, `MARScore`, and `RendererClient`.
- This env defaults `prime eval run` to `api_client_type="renderer"`. Use `--api-client-type openai_chat_completions_token` when running against a token-route vLLM endpoint, or `--api-client-type openai_chat_completions` for generic OpenAI-compatible chat endpoints.
- Runtime routing is intentionally outside this package. Learner/opponent endpoints, provider pinning, LoRA aliases, and seat-selection policy should be handled by the rollout runtime using per-request lineage metadata.

### Quickstart
Run a small local evaluation:

```bash
prime eval run gpqa-debate \
  -m openai/gpt-4.1-mini \
  --api-client-type openai_chat_completions \
  -n 5 -r 1 -t 1024 -T 1.0
```

Save trajectory data for multi-agent training projections:

```bash
prime eval run gpqa-debate \
  -m openai/gpt-4.1-mini \
  --api-client-type openai_chat_completions \
  -n 50 -r 4 -s -C trajectory
```

Publish privately:

```bash
prime env push --path ./environments/gpqa_debate --visibility=PRIVATE
```

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `prompts_ref` | str | `"selfplay"` | Built-in prompt pack name or path to a prompt pack YAML file |
| `subset` | str | `"gpqa_diamond"` | GPQA subset: `gpqa_diamond` or `gpqa_main` |
| `num_train_examples` | int | `-1` | Number of train examples; `-1` means all |
| `num_eval_examples` | int | `-1` | Number of eval examples; `-1` means all |
| `seed` | int | `0` | Choice-shuffling seed |
| `schedule` | list | default 5-slot schedule | Override turn schedule as a list of `{slot_id, agents, phase}` dicts |
| `truth_member` | str | `"debater_a"` | Member credited as the truth-tracking seat |
| `judge_model` | str | `"gpt-5.4-nano"` | Model used by optional open-ended grader/matcher templates |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | Episode scalar from `MARScore` |
| `reward/<member>` | Per-member training reward |
| `parse_errors/<member>` | Count of quarantined malformed outputs for that member |
| `agreement` | Debate diagnostic emitted by the rubric when available |
