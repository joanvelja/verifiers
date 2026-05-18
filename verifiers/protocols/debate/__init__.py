"""Debate protocol helpers: env wrapper, rubric, prompts, fields, parsing."""

from .env import DebateEnv, load_environment
from .fields import FieldSpec
from .channels import merge_provider_reasoning, parse_channels
from .parsing import extract_fields
from .prompts import (
    DebatePrompts,
    build_context,
    resolve_prompts,
)
from .rubric import DebateRubric
from verifiers.utils.judge_prompts import JudgeTemplate, normalize_verdict_token

__all__ = [
    "DebatePrompts",
    "DebateEnv",
    "DebateRubric",
    "FieldSpec",
    "JudgeTemplate",
    "build_context",
    "extract_fields",
    "load_environment",
    "merge_provider_reasoning",
    "normalize_verdict_token",
    "parse_channels",
    "resolve_prompts",
]
