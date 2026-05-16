from __future__ import annotations

import pytest

from verifiers.protocols.debate.fields import FieldSpec
from verifiers.protocols.debate.parsing import extract_fields


def test_extract_fields_finds_schema_tags_nested_in_other_schema_tags() -> None:
    fields = {
        "reasoning": FieldSpec(type=str),
        "decision": FieldSpec(type=str),
    }

    parsed = extract_fields(
        "<reasoning>Debater A is better supported.\n"
        "<decision>debater_a</decision>\n"
        "</reasoning>",
        fields,
    )

    assert parsed == {
        "reasoning": "Debater A is better supported.\n<decision>debater_a</decision>",
        "decision": "debater_a",
    }


def test_extract_fields_rejects_duplicate_nested_and_top_level_tags() -> None:
    fields = {"decision": FieldSpec(type=str)}

    with pytest.raises(ValueError, match="Duplicate XML tag 'decision'"):
        extract_fields(
            "<reasoning><decision>debater_a</decision></reasoning>\n"
            "<decision>debater_b</decision>",
            fields,
        )


def test_extract_fields_accepts_repeated_identical_commit() -> None:
    fields = {"answer": FieldSpec(type=str)}

    parsed = extract_fields(
        "Final verdict: <answer>D</answer>\n"
        "<opponent_error>none found</opponent_error>\n"
        "<answer>D</answer>",
        fields,
    )

    assert parsed == {"answer": "D"}
