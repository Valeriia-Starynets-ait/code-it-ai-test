"""Project exception hierarchy.

Every custom exception inherits from :class:`RagError` and carries context about
what failed so callers (and logs) have a real cause, not a bare message.
"""

from __future__ import annotations


class RagError(Exception):
    """Base class for every error raised by the RAG pipeline."""


class IngestionError(RagError):
    """Raised when a document cannot be ingested into the knowledge base."""

    def __init__(self, document_id: str, reason: str) -> None:
        self.document_id = document_id
        self.reason = reason
        super().__init__(f"failed to ingest document {document_id!r}: {reason}")


class RetrievalError(RagError):
    """Raised when retrieval against the knowledge base fails or is misconfigured."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"retrieval failed: {reason}")


class GenerationError(RagError):
    """Raised when an answer cannot be generated from retrieved passages."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"generation failed: {reason}")
