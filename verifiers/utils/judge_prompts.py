from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_JUDGE_PROMPTS_DIR = Path(__file__).resolve().parent / "judge_prompts"


@dataclass(frozen=True)
class JudgeTemplate:
    """Raw system/user prompt templates for an eval-time LLM judge call."""

    system: str
    user: str
    positive: str
    negative: str
    model: str | None = None
    sampling_args: dict[str, Any] | None = None
    rubric_family: str | None = None
    variant_id: str | None = None
    reward_mode: str | None = None
    threshold: float | None = None
    calibration_mode: str | None = None
    correctness_prior: float | None = None
    sensitivity: float | None = None
    false_positive_rate: float | None = None
    repeated_call_correlation: float | None = None


UTILITY_JUDGE_BLOCK_NAMES = ("_matcher", "_grader")


def normalize_verdict_token(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    token = stripped.split()[0].rstrip(string.punctuation)
    return token.upper() if token else None


def resolve_judge_prompts(ref: str) -> dict[str, JudgeTemplate]:
    ref = ref.strip()
    if "/" not in ref and "." not in ref:
        path = _JUDGE_PROMPTS_DIR / f"{ref}.yaml"
    else:
        path = Path(ref)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Judge prompt pack {path} must contain a mapping")
    validate_judge_blocks(data)
    return compile_judge_blocks(data)


def validate_judge_blocks(d: dict[str, Any]) -> None:
    ref = d.get("_judge_prompts")
    if ref is not None:
        if not isinstance(ref, str):
            raise ValueError("_judge_prompts: expected str")
        resolve_judge_prompts(ref)
    for block_name in UTILITY_JUDGE_BLOCK_NAMES:
        block = d.get(block_name)
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ValueError(
                f"{block_name}: expected mapping, got {type(block).__name__}"
            )
        required_keys = {"system", "user", "positive", "negative"}
        optional_keys = {"model", "sampling_args", "cache", "reward", "calibration"}
        missing = required_keys - set(block)
        extra = set(block) - required_keys - optional_keys
        if missing:
            raise ValueError(f"{block_name}: missing required keys {sorted(missing)}")
        if extra:
            raise ValueError(f"{block_name}: unknown keys {sorted(extra)}")
        for key in ("system", "user", "positive", "negative"):
            if not isinstance(block[key], str):
                raise ValueError(f"{block_name}.{key}: expected str")
        positive = normalize_verdict_token(block["positive"])
        negative = normalize_verdict_token(block["negative"])
        if positive is None:
            raise ValueError(f"{block_name}.positive: expected non-empty verdict token")
        if negative is None:
            raise ValueError(f"{block_name}.negative: expected non-empty verdict token")
        if positive == negative:
            raise ValueError(
                f"{block_name}: positive/negative must normalize to distinct verdicts"
            )
        if "model" in block and not isinstance(block["model"], str):
            raise ValueError(f"{block_name}.model: expected str")
        if "sampling_args" in block and not isinstance(block["sampling_args"], dict):
            raise ValueError(f"{block_name}.sampling_args: expected mapping")
        validate_optional_mapping(block, block_name, "cache")
        validate_optional_mapping(block, block_name, "reward")
        validate_optional_mapping(block, block_name, "calibration")


def validate_optional_mapping(block: dict[str, Any], block_name: str, key: str) -> None:
    value = block.get(key)
    if value is not None and not isinstance(value, dict):
        raise ValueError(f"{block_name}.{key}: expected mapping")


def compile_judge_blocks(d: dict[str, Any]) -> dict[str, JudgeTemplate]:
    ref = d.get("_judge_prompts")
    result = resolve_judge_prompts(ref) if isinstance(ref, str) else {}
    for block_name, short_name in (("_matcher", "matcher"), ("_grader", "grader")):
        block = d.get(block_name)
        if block is None:
            continue
        positive = normalize_verdict_token(block["positive"])
        negative = normalize_verdict_token(block["negative"])
        if positive is None or negative is None:
            raise ValueError(
                f"{block_name}: verdict tokens must be non-empty "
                f"(positive={block['positive']!r}, negative={block['negative']!r})"
            )
        result[short_name] = JudgeTemplate(
            system=block["system"],
            user=block["user"],
            positive=positive,
            negative=negative,
            model=block.get("model"),
            sampling_args=block.get("sampling_args"),
            rubric_family=(block.get("cache") or {}).get("rubric_family"),
            variant_id=(block.get("cache") or {}).get("variant_id"),
            reward_mode=(block.get("reward") or {}).get("mode"),
            threshold=(block.get("reward") or {}).get("threshold"),
            calibration_mode=(block.get("calibration") or {}).get("mode"),
            correctness_prior=(block.get("calibration") or {}).get("correctness_prior"),
            sensitivity=(block.get("calibration") or {}).get("sensitivity"),
            false_positive_rate=(block.get("calibration") or {}).get(
                "false_positive_rate"
            ),
            repeated_call_correlation=(block.get("calibration") or {}).get(
                "repeated_call_correlation"
            ),
        )
    return result
