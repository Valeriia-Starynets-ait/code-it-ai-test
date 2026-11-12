"""Answer generation with inline citations.

The :class:`TemplateGenerator` composes a deterministic, grounded answer from the
retrieved passages and attaches a :class:`rag.models.Citation` for each one, with
inline ``[n]`` markers. Citations are the trust signal: every claim in the answer
points at the passage that supports it. A real Bedrock ``retrieve_and_generate``
adapter would implement the same :class:`rag.ports.Generator` port.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from rag.exceptions import GenerationError
from rag.models import Citation, GeneratedAnswer, RetrievedPassage

logger = logging.getLogger(__name__)


class TemplateGenerator:
    """Builds a cited answer by stitching retrieved passages together.

    This is intentionally deterministic (no model call) so tests and the eval
    harness are reproducible. Swapping in a real generator means implementing the
    same ``generate`` signature.
    """

    def __init__(self, max_passages: int | None = None) -> None:
        if max_passages is not None and max_passages < 1:
            raise GenerationError(f"max_passages must be >= 1, got {max_passages}")
        self._max_passages = max_passages

    def generate(self, query: str, passages: Sequence[RetrievedPassage]) -> GeneratedAnswer:
        if not query.strip():
            raise GenerationError("query must not be empty")
        if not passages:
            # No grounding means we refuse to answer rather than hallucinate.
            raise GenerationError("no passages retrieved; cannot ground an answer")

        used = list(passages)
        if self._max_passages is not None:
            used = used[: self._max_passages]

        sentences: list[str] = []
        citations: list[Citation] = []
        for marker, passage in enumerate(used, start=1):
            sentences.append(f"{passage.text.rstrip('.')} [{marker}].")
            citations.append(
                Citation(
                    marker=marker,
                    doc_id=passage.doc_id,
                    source=passage.source,
                    page=passage.page,
                )
            )

        text = " ".join(sentences)
        logger.info("generated answer with %d citation(s)", len(citations))
        return GeneratedAnswer(text=text, citations=tuple(citations))
