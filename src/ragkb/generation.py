from collections.abc import Collection
from typing import Any, Protocol

from botocore.exceptions import BotoCoreError, ClientError

from ragkb.errors import CitationError, GenerationError, NoEvidenceError
from ragkb.models import Answer, Citation, Passage
from ragkb.retrieval import Retriever
from ragkb.text import citation_markers, content_tokens, split_sentences

MAX_TOKENS = 512
EXTRACTIVE_PASSAGES = 2


class Generator(Protocol):
    def generate(self, question: str, passages: list[Passage]) -> str: ...


class RuntimeClient(Protocol):
    def converse(self, **kwargs: Any) -> dict[str, Any]: ...


class ExtractiveGenerator:
    """Offline generator: quotes the best sentence from the top passages and cites each by number."""

    def generate(self, question: str, passages: list[Passage]) -> str:
        terms = content_tokens(question)
        parts: list[str] = []
        for number, passage in enumerate(passages[:EXTRACTIVE_PASSAGES], start=1):
            sentences = split_sentences(passage.text)
            best = max(sentences, key=lambda s: len(terms & content_tokens(s)), default="")
            if best:
                parts.append(f"{best.rstrip('.')} [{number}].")
        return " ".join(parts)


class BedrockGenerator:
    """Generates a cited answer with a Bedrock model through the Converse API."""

    def __init__(self, client: RuntimeClient, model_id: str) -> None:
        self._client = client
        self._model_id = model_id

    def generate(self, question: str, passages: list[Passage]) -> str:
        try:
            response = self._client.converse(
                modelId=self._model_id,
                messages=[{"role": "user", "content": [{"text": _build_prompt(question, passages)}]}],
                inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": 0.0},
            )
        except (BotoCoreError, ClientError) as exc:
            raise GenerationError(f"Bedrock converse failed: {exc}") from exc
        try:
            return str(response["output"]["message"]["content"][0]["text"])
        except (KeyError, IndexError, TypeError) as exc:
            raise GenerationError("malformed Bedrock response") from exc


def _build_prompt(question: str, passages: list[Passage]) -> str:
    context = "\n".join(f"[{n}] {p.text}" for n, p in enumerate(passages, start=1))
    return (
        "Answer the question using only the numbered passages. Cite every claim with its passage number "
        "in square brackets, like [1]. If the passages do not contain the answer, say so.\n\n"
        f"Passages:\n{context}\n\nQuestion: {question}"
    )


def _build_citations(text: str, passages: list[Passage], known_sources: Collection[str] | None) -> list[Citation]:
    markers = citation_markers(text)
    if not markers:
        raise CitationError("answer has no citations")
    citations: list[Citation] = []
    for marker in markers:
        if not 1 <= marker <= len(passages):
            raise CitationError(f"answer cites [{marker}] but only {len(passages)} passages were retrieved")
        passage = passages[marker - 1]
        if known_sources is not None and passage.source not in known_sources:
            raise CitationError(f"citation [{marker}] points to unknown source {passage.source}")
        citations.append(Citation(marker=marker, source=passage.source, chunk_id=passage.chunk_id))
    return citations


def answer_question(
    question: str,
    retriever: Retriever,
    generator: Generator,
    top_k: int,
    known_sources: Collection[str] | None = None,
) -> Answer:
    passages = retriever.retrieve(question, top_k)
    if not passages:
        raise NoEvidenceError("no relevant passages found")
    text = generator.generate(question, passages)
    citations = _build_citations(text, passages, known_sources)
    return Answer(question=question, text=text, citations=citations, passages=passages)
