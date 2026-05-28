from dataclasses import dataclass

from verifiers.utils.usage_utils import StateUsageTracker


@dataclass(slots=True)
class ModelRequestContext:
    """Ephemeral metadata for one model request."""

    member_id: str | None = None
    usage_tracker: StateUsageTracker | None = None
    prefix_candidate_indices: tuple[int, ...] | None = None
