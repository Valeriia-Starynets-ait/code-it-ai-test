import re

STOPWORDS = frozenset(
    {
        *("a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "has", "have"),
        *("how", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "what", "when"),
        *("where", "which", "who", "why", "will", "with", "no", "not"),
    }
)
_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MARKER = re.compile(r"\[(\d+)\]")


def _stem(word: str) -> str:
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def content_tokens(text: str) -> set[str]:
    words = _WORD.findall(_MARKER.sub(" ", text).lower())
    return {_stem(w) for w in words if w not in STOPWORDS}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def citation_markers(text: str) -> list[int]:
    seen: list[int] = []
    for match in _MARKER.finditer(text):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen
