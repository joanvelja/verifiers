# gpqa-rlvr

### Overview
- **Environment ID**: `gpqa-rlvr`
- **Short description**: GPQA Diamond / Main as single-agent RLVR with verifier-graded multiple-choice answers.
- **Tags**: gpqa, rlvr, single-agent, eval

### Datasets
- **Primary dataset(s)**: `Idavidrein/gpqa`
- **Source links**: [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa)
- **Splits**: Uses the Hugging Face `train` split for train and eval builders. Set `subset` to `gpqa_diamond` or `gpqa_main`.

### Task
- **Type**: single-turn
- **Output format**: answer letter inside `<answer>...</answer>` tags
- **Rubric overview**: exact match on the first A/B/C/D letter parsed from the `<answer>` tag.

### Requirements
- `HF_TOKEN` is required for the gated GPQA dataset.

### Quickstart
```bash
prime eval run gpqa-rlvr \
  -m openai/gpt-4.1-mini \
  -n 198 -r 4 -t 2048 -T 1.0
```

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `subset` | str | `"gpqa_diamond"` | GPQA subset: `gpqa_diamond` or `gpqa_main` |
| `num_train_examples` | int | `-1` | Number of train examples; `-1` means all |
| `num_eval_examples` | int | `-1` | Number of eval examples; `-1` means all |
| `seed` | int | `0` | Choice-shuffling seed |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | 1.0 if parsed letter equals ground truth, else 0.0 |
