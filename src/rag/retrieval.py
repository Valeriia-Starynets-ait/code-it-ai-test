"""In-memory fake knowledge base client for ingestion and retrieval.

This stands in for the Bedrock ``bedrock-agent-runtime`` retrieve API so the
whole pipeline runs with no AWS credentials and no network. Scoring is a
deterministic Jaccard token overlap, normalised to ``[0, 1]`` to mirror the
``score`` field a real vector search returns. The public shape matches the
:class:`rag.ports.Ingestor` and :class:`rag.ports.Retriever` protocols, so a
real Bedrock adapter can replace it without touching the pipeline.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from rag.exceptions import IngestionError, RetrievalError
from rag.models import Document, RetrievedPassage

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    intersection = query_tokens & doc_tokens
    union = query_tokens | doc_tokens
    return len(intersection) / len(union)


class FakeKnowledgeBaseClient:
    """A credential-free stand-in for a Bedrock Knowledge Base data source."""

    def __init__(self, min_score: float = 0.0) -> None:
        if not 0.0 <= min_score <= 1.0:
            raise RetrievalError(f"min_score must be in [0, 1], got {min_score}")
        self._min_score = min_score
        self._documents: list[Document] = []

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def ingest(self, documents: Sequence[Document]) -> int:
        """Index documents. Rejects duplicate ids so ingestion stays idempotent-safe."""
        seen = {doc.doc_id for doc in self._documents}
        indexed = 0
        for doc in documents:
            if doc.doc_id in seen:
                raise IngestionError(doc.doc_id, "duplicate document id")
            self._documents.append(doc)
            seen.add(doc.doc_id)
            indexed += 1
        logger.info("ingested %d document(s); total indexed=%d", indexed, len(self._documents))
        return indexed

    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedPassage]:
        """Return the top_k passages above ``min_score``, highest score first."""
        if not query.strip():
            raise RetrievalError("query must not be empty")
        if top_k < 1:
            raise RetrievalError(f"top_k must be >= 1, got {top_k}")
        if not self._documents:
            # A KB with zero indexed docs returns empty, not an error.
            logger.warning("retrieve called on an empty knowledge base")
            return ()

        query_tokens = _tokenize(query)
        scored: list[RetrievedPassage] = []
        for doc in self._documents:
            score = _jaccard(query_tokens, _tokenize(doc.text))
            if score < self._min_score:
                continue
            scored.append(
                RetrievedPassage(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    source=doc.source,
                    score=score,
                    page=doc.page,
                )
            )
        # Sort by score desc, then doc_id asc for a stable, deterministic order.
        scored.sort(key=lambda p: (-p.score, p.doc_id))
        return tuple(scored[:top_k])
