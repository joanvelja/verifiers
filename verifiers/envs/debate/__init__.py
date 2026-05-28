"""Debate environment: prompts, fields, parsing, think-tag handling."""

from .fields import FieldSpec
from .parsing import extract_fields
from .prompts import (
    DebatePrompts,
    build_context,
    resolve_prompts,
)
from verifiers.utils.judge_prompts import JudgeTemplate, normalize_verdict_token

__all__ = [
    "DebatePrompts",
    "FieldSpec",
    "JudgeTemplate",
    "build_context",
    "extract_fields",
    "normalize_verdict_token",
    "resolve_prompts",
]
