"""End-to-end tests: ingest -> retrieve -> generate -> eval, fake client only."""

from __future__ import annotations

import pytest

from rag.config import Settings
from rag.eval import EvalSample, evaluate
from rag.generation import TemplateGenerator
from rag.ingestion import sample_corpus
from rag.pipeline import RagPipeline
from rag.retrieval import FakeKnowledgeBaseClient

pytestmark = pytest.mark.e2e


def _pipeline() -> RagPipeline:
    client = FakeKnowledgeBaseClient()
    client.ingest(sample_corpus())
    return RagPipeline(client, TemplateGenerator(), top_k=Settings().top_k)


def test_end_to_end_question_is_answered_with_citations() -> None:
    result = _pipeline().answer("which embedding model is default and how many dimensions")
    assert result.passages[0].doc_id == "kb-002"
    assert result.answer.citations
    assert result.answer.citations[0].source.startswith("s3://")
    assert "[1]" in result.answer.text


def test_end_to_end_eval_meets_thresholds() -> None:
    pipeline = _pipeline()
    samples = (
        EvalSample(
            question="What does a Bedrock Knowledge Base manage?",
            expected_answer="ingestion embedding chunking and vector search",
            relevant_doc_ids=("kb-001",),
            expected_pages=(1,),
        ),
        EvalSample(
            question="Which embedding model is the default and how many dimensions?",
            expected_answer="titan text embeddings v2 1024 dimensions",
            relevant_doc_ids=("kb-002",),
            expected_pages=(2,),
        ),
        EvalSample(
            question="How can an answer be traced back to its source?",
            expected_answer="ranked passages with relevance scores and page citations",
            relevant_doc_ids=("kb-003",),
            expected_pages=(3,),
        ),
    )
    answers = [pipeline.answer(s.question) for s in samples]
    report = evaluate(answers, samples)
    assert report.sample_count == 3
    assert report.context_recall == 1.0
    assert report.meets_thresholds()
