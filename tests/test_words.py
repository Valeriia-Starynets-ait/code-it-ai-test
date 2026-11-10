from src.words import count_words


def test_normal_sentence():
    assert count_words("The quick brown fox jumps") == 5


def test_extra_spaces():
    assert count_words("  hello    world  ") == 2


def test_empty_string():
    assert count_words("") == 0
