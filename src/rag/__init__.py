"""Bedrock Knowledge Base RAG pipeline.

Public surface: configuration, domain models, exceptions, and the pipeline
orchestrator. Infrastructure clients (ingestion, retrieval, generation) live in
their own modules and are injected into the pipeline.
"""

from rag.bedrock import BedrockKnowledgeBase
from rag.config import Settings
from rag.exceptions import (
    GenerationError,
    IngestionError,
    RagError,
    RetrievalError,
)
from rag.filters import EqualsCondition, build_metadata_filter
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
    "BedrockKnowledgeBase",
    "Citation",
    "Document",
    "EqualsCondition",
    "GeneratedAnswer",
    "GenerationError",
    "IngestionError",
    "RagError",
    "RagPipeline",
    "RetrievalError",
    "RetrievedPassage",
    "Settings",
    "build_metadata_filter",
]
