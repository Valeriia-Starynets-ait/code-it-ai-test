"""Ports (interfaces) the pipeline depends on.

The domain and application code depend on these protocols, never on a concrete
SDK client. Adapters (the fake client, a future real Bedrock client) implement
them and are injected into the pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag.models import Document, GeneratedAnswer, RetrievedPassage


class Retriever(Protocol):
    """Returns ranked passages for a query (the Bedrock ``retrieve`` shape)."""

    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedPassage]: ...


class Ingestor(Protocol):
    """Ingests documents into the knowledge base and reports how many landed."""

    def ingest(self, documents: Sequence[Document]) -> int: ...


class Generator(Protocol):
    """Composes a cited answer from a query and its retrieved passages."""

    def generate(self, query: str, passages: Sequence[RetrievedPassage]) -> GeneratedAnswer: ...
