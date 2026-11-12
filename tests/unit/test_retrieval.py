from __future__ import annotations

import pytest

from rag.exceptions import RetrievalError
from rag.ingestion import sample_corpus
from rag.retrieval import FakeKnowledgeBaseClient

pytestmark = pytest.mark.unit


def test_retrieve_ranks_relevant_passage_first(client: FakeKnowledgeBaseClient) -> None:
    results = client.retrieve("which embedding model and dimensions", top_k=3)
    assert results
    assert results[0].doc_id == "kb-002"


def test_retrieve_respects_top_k(client: FakeKnowledgeBaseClient) -> None:
    results = client.retrieve("bedrock knowledge base retrieval embedding", top_k=1)
    assert len(results) == 1


def test_retrieve_scores_in_unit_interval(client: FakeKnowledgeBaseClient) -> None:
    results = client.retrieve("bedrock knowledge base", top_k=5)
    assert all(0.0 <= p.score <= 1.0 for p in results)


def test_retrieve_is_deterministic(client: FakeKnowledgeBaseClient) -> None:
    first = client.retrieve("bedrock retrieval citations", top_k=3)
    second = client.retrieve("bedrock retrieval citations", top_k=3)
    assert [p.doc_id for p in first] == [p.doc_id for p in second]


def test_retrieve_empty_kb_returns_empty() -> None:
    client = FakeKnowledgeBaseClient()
    assert client.retrieve("anything", top_k=5) == ()


def test_retrieve_empty_query_raises(client: FakeKnowledgeBaseClient) -> None:
    with pytest.raises(RetrievalError):
        client.retrieve("   ", top_k=5)


def test_retrieve_bad_top_k_raises(client: FakeKnowledgeBaseClient) -> None:
    with pytest.raises(RetrievalError):
        client.retrieve("bedrock", top_k=0)


def test_min_score_filters_low_scoring_passages() -> None:
    client = FakeKnowledgeBaseClient(min_score=0.99)
    client.ingest(sample_corpus())
    assert client.retrieve("bedrock knowledge base", top_k=5) == ()


def test_bad_min_score_raises() -> None:
    with pytest.raises(RetrievalError):
        FakeKnowledgeBaseClient(min_score=1.5)
