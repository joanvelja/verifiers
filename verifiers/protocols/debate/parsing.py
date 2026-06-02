"""Core XML parser, extraction, normalization, format instructions."""

from collections.abc import Callable, Mapping
import logging
import re
from typing import Any

from .fields import FieldSpec

_log = logging.getLogger(__name__)


def _coerce_bool(v: str) -> bool | None:
    normed = v.strip().lower()
    if normed in ("true", "yes", "1"):
        return True
    if normed in ("false", "no", "0"):
        return False
    return None


_COERCE: dict[type, Callable[[str], Any]] = {
    str: lambda v: v,
    int: int,
    float: float,
    bool: _coerce_bool,
    list: lambda v: [item.strip() for item in v.split(",")],
}


def _coerce(value: str, target_type: type) -> Any:
    fn = _COERCE.get(target_type)
    if fn is None:
        _log.debug("No coercion for type %s", target_type.__name__)
        return None
    try:
        return fn(value)
    except (ValueError, TypeError) as exc:
        _log.debug("Field coerce to %s failed: %s", target_type.__name__, exc)
        return None


def parse(text: str, schema: dict[str, type]) -> dict[str, Any] | None:
    result: dict[str, Any] = {}
    for tag, target_type in schema.items():
        tag_re = re.compile(
            rf"<\s*{re.escape(tag)}(?=\s|>)[^>]*>(.*?)</\s*{re.escape(tag)}\s*>",
            re.DOTALL,
        )
        matches = tag_re.findall(text)
        if not matches:
            continue
        coerced_values = [_coerce(content.strip(), target_type) for content in matches]
        if len(matches) > 1:
            first = coerced_values[0]
            if first is None or any(value != first for value in coerced_values[1:]):
                raise ValueError(
                    f"Duplicate XML tag '{tag}' in response — ambiguous commit"
                )
        coerced = coerced_values[0]
        if coerced is not None:
            result[tag] = coerced
    return result or None


def normalize_fields(
    raw: dict[str, Any], specs: dict[str, FieldSpec]
) -> dict[str, Any]:
    result = dict(raw)
    for key, value in raw.items():
        spec = specs.get(key)
        if spec is not None and spec.normalizer is not None:
            result[key] = spec.normalizer(value)
    return result


def extract_fields(text: str, specs: dict[str, FieldSpec]) -> dict[str, Any] | None:
    type_map = {k: v.type for k, v in specs.items()}
    raw = parse(text, type_map)
    if raw is None:
        return None
    return normalize_fields(raw, specs)


def generate_format_instructions(fields: Mapping[str, FieldSpec]) -> str:
    # Describe each tag WITHOUT a fillable ``<tag>PLACEHOLDER</tag>`` exemplar.
    # Weak models copy the placeholder verbatim (e.g. echoing "YOUR DECISION"),
    # and the placeholder's own tags get duplicated — both then fail the strict
    # parser. Naming the tag + its allowed content conveys the format without
    # the copy-bait.
    from .fields import BinaryScoring, EnumScoring, NumericScoring

    parts: list[str] = []
    for name, spec in fields.items():
        scoring = spec.scoring
        if isinstance(scoring, BinaryScoring):
            parts.append(
                f"a <{name}> tag containing either {scoring.true_value} or {scoring.false_value}"
            )
        elif isinstance(scoring, EnumScoring):
            parts.append(
                f"a <{name}> tag containing exactly one of: {', '.join(scoring.values)}"
            )
        elif isinstance(scoring, NumericScoring):
            parts.append(
                f"a <{name}> tag containing a number between {scoring.min_val} and {scoring.max_val}"
            )
        elif spec.description:
            parts.append(f"a <{name}> tag containing {spec.description}")
        else:
            parts.append(f"a <{name}> tag containing your {name}")
    if not parts:
        return ""
    body = parts[0] if len(parts) == 1 else ", then ".join(parts)
    return f"End your response with {body}. Write nothing after the final tag."
