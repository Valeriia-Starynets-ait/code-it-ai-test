from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag.exceptions import GenerationError, RetrievalError
from rag.generation import TemplateGenerator
from rag.models import GeneratedAnswer, RetrievedPassage
from rag.pipeline import RagPipeline
from rag.retrieval import FakeKnowledgeBaseClient

pytestmark = pytest.mark.unit


class _EmptyRetriever:
    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievedPassage]:
        return ()


class _RecordingGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def generate(self, query: str, passages: Sequence[RetrievedPassage]) -> GeneratedAnswer:
        self.calls.append((query, len(passages)))
        return GeneratedAnswer(text="ok [1].")


def test_pipeline_returns_answer_with_passages(client: FakeKnowledgeBaseClient) -> None:
    pipeline = RagPipeline(client, TemplateGenerator(), top_k=3)
    result = pipeline.answer("which embedding model and dimensions")
    assert result.question == "which embedding model and dimensions"
    assert result.passages
    assert result.answer.citations


def test_pipeline_passes_retrieved_passages_to_generator(
    client: FakeKnowledgeBaseClient,
) -> None:
    generator = _RecordingGenerator()
    pipeline = RagPipeline(client, generator, top_k=2)
    pipeline.answer("bedrock knowledge base retrieval")
    assert generator.calls
    _, passage_count = generator.calls[0]
    assert passage_count >= 1


def test_pipeline_empty_question_raises(client: FakeKnowledgeBaseClient) -> None:
    pipeline = RagPipeline(client, TemplateGenerator(), top_k=3)
    with pytest.raises(RetrievalError):
        pipeline.answer("   ")


def test_pipeline_bad_top_k_raises(client: FakeKnowledgeBaseClient) -> None:
    with pytest.raises(RetrievalError):
        RagPipeline(client, TemplateGenerator(), top_k=0)


def test_pipeline_propagates_generation_error_on_no_passages() -> None:
    pipeline = RagPipeline(_EmptyRetriever(), TemplateGenerator(), top_k=3)
    with pytest.raises(GenerationError):
        pipeline.answer("anything")
