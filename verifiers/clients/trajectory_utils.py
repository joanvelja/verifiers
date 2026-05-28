from collections.abc import Iterator, Mapping, Sequence
from typing import Any, TypeVar

_T = TypeVar("_T")


def get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalize_member_id(value: Any) -> str | None:
    if value is None:
        return None
    key = str(value)
    return key or None


def step_member_ids(step: Any) -> set[str]:
    keys: set[str] = set()

    extras = get_value(step, "extras")
    if isinstance(extras, Mapping):
        key = normalize_member_id(extras.get("member_id"))
        if key is not None:
            keys.add(key)

    key = normalize_member_id(get_value(step, "trajectory_id"))
    if key is not None:
        keys.add(key)

    return keys


def iter_member_candidate_steps(
    trajectory: Sequence[_T],
    *,
    member_id: str | None,
    fallback_member_id: Any = None,
    prefix_candidate_indices: tuple[int, ...] | None = None,
) -> Iterator[_T]:
    member_key = normalize_member_id(member_id)
    if member_key is None:
        member_key = normalize_member_id(fallback_member_id)

    if prefix_candidate_indices is None:
        candidate_steps = reversed(trajectory)
    else:
        candidate_steps = (
            trajectory[idx] for idx in reversed(prefix_candidate_indices)
        )

    for step in candidate_steps:
        if member_key is not None and member_key not in step_member_ids(step):
            continue
        yield step
