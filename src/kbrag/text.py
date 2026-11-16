import re

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by do does for from how in is it of on or per the to was what when where which who why with".split()
)


def _stem(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(text: str) -> set[str]:
    """Lowercased, lightly stemmed tokens without stopwords."""
    return {_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS}
