"""Metadata filters for Bedrock Knowledge Base retrieval.

Encodes the skill's field note: Bedrock's ``orAll`` needs 2+ members, so a
single condition must be the bare condition, not a one-element ``orAll``, or the
Retrieve API rejects it. :func:`build_metadata_filter` applies that rule.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

# Minimum members Bedrock requires inside an orAll/andAll group.
_MIN_GROUP_MEMBERS = 2


class EqualsCondition(BaseModel):
    """A single ``key == value`` equality condition."""

    model_config = ConfigDict(frozen=True, strict=True)

    key: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)

    def to_filter(self) -> dict[str, object]:
        return {"equals": {"key": self.key, "value": self.value}}


def build_metadata_filter(
    conditions: Sequence[EqualsCondition],
) -> dict[str, object] | None:
    """Build a Retrieve API metadata filter from equality conditions.

    - zero conditions -> ``None`` (no filter);
    - one condition -> the bare condition (NOT a one-element ``orAll``);
    - two or more -> an ``orAll`` group.
    """
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0].to_filter()
    members = [c.to_filter() for c in conditions]
    if len(members) < _MIN_GROUP_MEMBERS:  # defensive; unreachable given the guard above
        return members[0]
    return {"orAll": members}
