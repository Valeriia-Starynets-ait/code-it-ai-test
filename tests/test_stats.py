import pytest

from src.stats import max_of


def test_max_of_returns_largest_number():
    assert max_of([3, 1, 4, 1, 5, 9, 2, 6]) == 9


def test_max_of_handles_negative_numbers():
    assert max_of([-5, -2, -9]) == -2


def test_max_of_single_element():
    assert max_of([7]) == 7


def test_max_of_empty_list_raises_value_error():
    with pytest.raises(ValueError):
        max_of([])
