from verifiers.api_profile import ApiProfile, filter_sampling_args_for_profile


def test_openai_strict_strips_vllm_structured_output_keys() -> None:
    sampling_args = {
        "temperature": 0.2,
        "guided_json": {"type": "object"},
        "extra_body": {
            "guided_choice": ["yes", "no"],
            "guided_regex": r"\d+",
            "guided_grammar": "root ::= answer",
            "structural_tag": {"schema": {}},
            "safe": "kept",
        },
    }

    filtered, stripped = filter_sampling_args_for_profile(
        sampling_args, ApiProfile.OPENAI_STRICT
    )

    assert "guided_json" not in filtered
    assert filtered["extra_body"] == {"safe": "kept"}
    assert stripped == frozenset(
        {
            "guided_json",
            "guided_choice",
            "guided_regex",
            "guided_grammar",
            "structural_tag",
        }
    )


def test_vllm_permissive_preserves_extra_body_keys() -> None:
    sampling_args = {
        "top_k": 20,
        "extra_body": {
            "return_token_ids": True,
            "guided_choice": ["A", "B"],
        },
    }

    filtered, stripped = filter_sampling_args_for_profile(
        sampling_args, ApiProfile.VLLM_PERMISSIVE
    )

    assert filtered == sampling_args
    assert stripped == frozenset()
