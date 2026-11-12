from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.exceptions import IngestionError
from rag.ingestion import load_documents, load_documents_from_jsonl, sample_corpus
from rag.retrieval import FakeKnowledgeBaseClient

pytestmark = pytest.mark.unit


def test_load_documents_validates_records() -> None:
    records = [{"doc_id": "a", "text": "hello", "source": "s3://x", "page": 1}]
    docs = load_documents(records)
    assert len(docs) == 1
    assert docs[0].doc_id == "a"


def test_load_documents_raises_on_invalid_record() -> None:
    with pytest.raises(IngestionError) as exc:
        load_documents([{"doc_id": "a", "text": "", "source": "s3://x"}])
    assert exc.value.document_id == "a"


def test_load_documents_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    rows = [
        {"doc_id": "a", "text": "alpha", "source": "s3://x"},
        {"doc_id": "b", "text": "beta", "source": "s3://y"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    docs = load_documents_from_jsonl(path)
    assert [d.doc_id for d in docs] == ["a", "b"]


def test_load_documents_from_jsonl_missing_file(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        load_documents_from_jsonl(tmp_path / "nope.jsonl")


def test_load_documents_from_jsonl_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text("{not json}", encoding="utf-8")
    with pytest.raises(IngestionError):
        load_documents_from_jsonl(path)


def test_load_documents_from_jsonl_non_object_line(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text('["a list"]', encoding="utf-8")
    with pytest.raises(IngestionError):
        load_documents_from_jsonl(path)


def test_ingest_counts_documents() -> None:
    client = FakeKnowledgeBaseClient()
    indexed = client.ingest(sample_corpus())
    assert indexed == 3
    assert client.document_count == 3


def test_ingest_rejects_duplicate_ids() -> None:
    client = FakeKnowledgeBaseClient()
    client.ingest(sample_corpus())
    with pytest.raises(IngestionError):
        client.ingest(sample_corpus())
