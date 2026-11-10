"""Basic statistics helpers."""

from collections.abc import Iterable


def max_of(values: Iterable[float]) -> float:
    """Return the largest number in ``values``.

    Raises:
        ValueError: if ``values`` is empty.
    """
    items = list(values)
    if not items:
        raise ValueError("max_of() requires a non-empty list")
    return max(items)
