"""Domain models for the RAG pipeline.

All DTOs are frozen Pydantic v2 value objects. They carry no framework or SDK
imports so the domain stays independent of infrastructure (ports-and-adapters).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """A source document to be ingested into the knowledge base."""

    model_config = ConfigDict(frozen=True, strict=True)

    doc_id: str = Field(..., min_length=1, max_length=255)
    text: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1, description="Origin, e.g. an S3 URI.")
    page: int | None = Field(default=None, ge=1)


class RetrievedPassage(BaseModel):
    """A single ranked passage returned by retrieval.

    ``score`` mirrors the vector-search relevance score; ``page`` mirrors the
    native ``x-amz-bedrock-kb-document-page-number`` citation field.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    doc_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)
    page: int | None = Field(default=None, ge=1)


class Citation(BaseModel):
    """A citation linking an answer marker to the passage that grounds it."""

    model_config = ConfigDict(frozen=True, strict=True)

    marker: int = Field(..., ge=1, description="Inline marker, e.g. 1 for [1].")
    doc_id: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    page: int | None = Field(default=None, ge=1)


class GeneratedAnswer(BaseModel):
    """An answer plus the citations that ground each of its claims."""

    model_config = ConfigDict(frozen=True, strict=True)

    text: str = Field(..., min_length=1)
    citations: tuple[Citation, ...] = Field(default_factory=tuple)


class Answer(BaseModel):
    """The full pipeline result: question, answer, citations and the passages used."""

    model_config = ConfigDict(frozen=True, strict=True)

    question: str = Field(..., min_length=1)
    answer: GeneratedAnswer
    passages: tuple[RetrievedPassage, ...] = Field(default_factory=tuple)
