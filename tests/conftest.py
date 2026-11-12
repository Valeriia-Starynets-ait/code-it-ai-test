"""Shared fixtures: a sample corpus and a pre-ingested fake client.

No fixture performs a network call or needs AWS credentials.
"""

from __future__ import annotations

import pytest

from rag.ingestion import sample_corpus
from rag.models import Document
from rag.retrieval import FakeKnowledgeBaseClient


@pytest.fixture
def documents() -> list[Document]:
    return list(sample_corpus())


@pytest.fixture
def client(documents: list[Document]) -> FakeKnowledgeBaseClient:
    kb = FakeKnowledgeBaseClient()
    kb.ingest(documents)
    return kb
