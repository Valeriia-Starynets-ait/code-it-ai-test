import logging
import re
from typing import TYPE_CHECKING, Protocol

from kbrag.errors import CitationError, NoEvidenceError
from kbrag.models import Answer, Citation, Passage
from kbrag.retrieval import Retriever
from kbrag.text import content_tokens

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

logger = logging.getLogger(__name__)

_MARKER = re.compile(r"\[(\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
MAX_EXTRACTIVE_PASSAGES = 2
MIN_RELATIVE_SCORE = 0.5
QUOTE_CHARS = 200
MAX_TOKENS = 512

PROMPT = """Answer the question using only the numbered context passages below.
Cite every claim with its passage number in square brackets, for example [1].
If the context does not contain the answer, say so.

Context:
{context}

Question: {question}"""


class Generator(Protocol):
    def generate(self, question: str, passages: list[Passage]) -> str: ...


class ExtractiveGenerator:
    """Offline generator: returns the best matching sentence of each top passage with its marker."""

    def generate(self, question: str, passages: list[Passage]) -> str:
        query = content_tokens(question)
        top_score = passages[0].score if passages else 0.0
        parts = []
        for number, passage in enumerate(passages[:MAX_EXTRACTIVE_PASSAGES], start=1):
            if passage.score < top_score * MIN_RELATIVE_SCORE:
                continue
            sentences = _sentences(passage.text)
            if not sentences:
                continue
            best = max(sentences, key=lambda s: len(query & content_tokens(s)))
            parts.append(f"{best.rstrip('.')} [{number}].")
        return " ".join(parts)


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE.split(text.strip()) if s]


class BedrockGenerator:
    """Generates a cited answer with a Bedrock foundation model through the Converse API."""

    def __init__(self, client: "BedrockRuntimeClient", model_id: str) -> None:
        self._client = client
        self._model_id = model_id

    def generate(self, question: str, passages: list[Passage]) -> str:
        context = "\n".join(f"[{i}] ({p.source}) {p.text}" for i, p in enumerate(passages, start=1))
        prompt = PROMPT.format(context=context, question=question)
        response = self._client.converse(
            modelId=self._model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": MAX_TOKENS, "temperature": 0.0},
        )
        try:
            return str(response["output"]["message"]["content"][0]["text"])
        except (KeyError, IndexError) as exc:
            raise CitationError("Model response had no text content") from exc


def answer_question(question: str, retriever: Retriever, generator: Generator, top_k: int = 5) -> Answer:
    passages = retriever.retrieve(question, top_k)
    if not passages:
        raise NoEvidenceError(f"No supporting passages found for: {question}")
    text = generator.generate(question, passages)
    markers = sorted({int(m) for m in _MARKER.findall(text)})
    if not markers:
        raise CitationError("Generated answer has no citations")
    invalid = [m for m in markers if not 1 <= m <= len(passages)]
    if invalid:
        raise CitationError(f"Answer cites passages that do not exist: {invalid}")
    citations = [
        Citation(
            marker=m,
            source=passages[m - 1].source,
            chunk_id=passages[m - 1].chunk_id,
            quote=passages[m - 1].text[:QUOTE_CHARS],
        )
        for m in markers
    ]
    return Answer(question=question, text=text, citations=citations, passages=passages)


def verify_citations(answer: Answer, known_sources: set[str]) -> None:
    unknown = sorted({c.source for c in answer.citations} - known_sources)
    if unknown:
        raise CitationError(f"Citations point to unknown sources: {unknown}")
