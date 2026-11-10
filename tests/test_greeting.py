import pytest

from src.greeting import greet


def test_greet_normal_name():
    assert greet("Alice") == "Hello, Alice!"


@pytest.mark.parametrize("name", ["", " ", "   ", "\t\n"])
def test_greet_empty_or_blank_name(name):
    assert greet(name) == "Hello, world!"
