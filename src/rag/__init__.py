"""Bedrock Knowledge Base RAG pipeline.

Public surface: configuration, domain models, exceptions, and the pipeline
orchestrator. Infrastructure clients (ingestion, retrieval, generation) live in
their own modules and are injected into the pipeline.
"""

from rag.config import Settings
from rag.exceptions import (
    GenerationError,
    IngestionError,
    RagError,
    RetrievalError,
)
from rag.models import (
    Answer,
    Citation,
    Document,
    GeneratedAnswer,
    RetrievedPassage,
)
from rag.pipeline import RagPipeline

__all__ = [
    "Answer",
    "Citation",
    "Document",
    "GeneratedAnswer",
    "GenerationError",
    "IngestionError",
    "RagError",
    "RagPipeline",
    "RetrievalError",
    "RetrievedPassage",
    "Settings",
]
