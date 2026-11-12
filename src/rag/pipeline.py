"""The RAG pipeline orchestrator.

Wires ingestion, retrieval and cited generation together. Dependencies are
injected (retriever and generator ports), never constructed inside the class, so
the pipeline is testable with a fake client and swappable for a real Bedrock one.
"""

from __future__ import annotations

import logging

from rag.exceptions import RetrievalError
from rag.models import Answer
from rag.ports import Generator, Retriever

logger = logging.getLogger(__name__)


class RagPipeline:
    """Retrieve-then-generate over an injected retriever and generator."""

    def __init__(self, retriever: Retriever, generator: Generator, top_k: int = 5) -> None:
        if top_k < 1:
            raise RetrievalError(f"top_k must be >= 1, got {top_k}")
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k

    def answer(self, question: str) -> Answer:
        """Retrieve passages for the question, then generate a cited answer."""
        if not question.strip():
            raise RetrievalError("question must not be empty")
        passages = tuple(self._retriever.retrieve(question, self._top_k))
        logger.info("retrieved %d passage(s) for question", len(passages))
        generated = self._generator.generate(question, passages)
        return Answer(question=question, answer=generated, passages=passages)
