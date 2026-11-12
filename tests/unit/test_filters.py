from __future__ import annotations

import pytest

from rag.filters import EqualsCondition, build_metadata_filter

pytestmark = pytest.mark.unit


def test_no_conditions_returns_none() -> None:
    assert build_metadata_filter([]) is None


def test_single_condition_is_bare_not_orall() -> None:
    result = build_metadata_filter([EqualsCondition(key="tenant", value="acme")])
    assert result == {"equals": {"key": "tenant", "value": "acme"}}
    assert "orAll" not in (result or {})


def test_two_conditions_use_orall() -> None:
    result = build_metadata_filter(
        [
            EqualsCondition(key="tenant", value="acme"),
            EqualsCondition(key="tenant", value="globex"),
        ]
    )
    assert result is not None
    assert "orAll" in result
    assert len(result["orAll"]) == 2  # type: ignore[arg-type,index]
