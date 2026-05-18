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
- **Protocol**: two debaters answer simultaneously, critique simultaneously, then a neutral judge emits the final decision.
- **Rubric overview**: `DebateRubric` produces member-attributed rewards through `MARScore`: winning debater +1, losing debater -1, judge 0, tie/no decision 0. Ground-truth correctness is diagnostic-only unless an asymmetric pack explicitly sets `truth_member`.

### Requirements
- `HF_TOKEN` is required for the gated GPQA dataset.
- This environment depends on `verifiers>=0.1.15.dev7` plus `renderers>=0.1.8.dev2`. Use a local `verifiers` checkout or a published/pinned version that includes `MultiAgentEnv`, `DebateEnv`, `MultiAgentRubric`, `MARScore`, and `RendererClient`.
- This env defaults `prime eval run` to `api_client_type="renderer"`. Use `--api-client-type openai_chat_completions_token` when running against a token-route vLLM endpoint, or `--api-client-type openai_chat_completions` for generic OpenAI-compatible chat endpoints.
- Runtime routing is intentionally outside this package. Learner/opponent endpoints, provider pinning, LoRA aliases, and seat-selection policy should be handled by the rollout runtime using per-request `member_id` metadata.

### Quickstart
Run a small local evaluation:

```bash
prime eval run gpqa-debate \
  -m openai/gpt-4.1-mini \
  --api-client-type openai_chat_completions \
  -n 5 -r 1 -t 8192 -T 1.0
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
| `schedule` | list | default 3-slot simultaneous-propose/simultaneous-critique schedule | Override turn schedule as a list of `{slot_id, agents, phase}` dicts |
| `truth_member` | str \| null | `null` | Optional asymmetric truth side; leave unset for symmetric self-play/frozen-opponent debate |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | Episode scalar from `MARScore`; `0.0` for symmetric debate when `truth_member` is unset |
| `reward/<member>` | Per-member training reward |
| `parse_errors/<member>` | Count of quarantined malformed outputs for that member |
| `agreement` | Debate diagnostic emitted by the rubric when available |
| `any_debater_correct` / `all_debaters_correct` | Ground-truth diagnostics over debater final answers |
| `judge_selected_correct` | Ground-truth diagnostic for the selected debater, when the judge selected a debater |
