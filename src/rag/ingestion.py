"""Document loading for ingestion.

Turns raw records into validated :class:`rag.models.Document` value objects
before they reach the knowledge base client. Keeping this separate from the
client lets a real Bedrock adapter reuse the same loaders.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from rag.exceptions import IngestionError
from rag.models import Document

logger = logging.getLogger(__name__)


def load_documents(records: Iterable[dict[str, object]]) -> list[Document]:
    """Validate raw dict records into Documents, raising on the first bad record."""
    documents: list[Document] = []
    for index, record in enumerate(records):
        try:
            documents.append(Document.model_validate(record))
        except ValueError as exc:
            doc_id = str(record.get("doc_id", f"<index {index}>"))
            raise IngestionError(doc_id, f"invalid document record: {exc}") from exc
    logger.info("loaded %d document(s) for ingestion", len(documents))
    return documents


def load_documents_from_jsonl(path: Path) -> list[Document]:
    """Load documents from a JSON Lines file (one JSON object per line)."""
    if not path.is_file():
        raise IngestionError(str(path), "corpus file does not exist")
    records: list[dict[str, object]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise IngestionError(str(path), f"invalid JSON on line {line_no}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise IngestionError(str(path), f"line {line_no} is not a JSON object")
        records.append(parsed)
    return load_documents(records)


def sample_corpus() -> Sequence[Document]:
    """A small built-in corpus so the CLI demo runs with no external files."""
    return (
        Document(
            doc_id="kb-001",
            text=(
                "Amazon Bedrock Knowledge Bases manage ingestion, embedding, chunking "
                "and vector search for retrieval augmented generation."
            ),
            source="s3://corpus/bedrock-overview.txt",
            page=1,
        ),
        Document(
            doc_id="kb-002",
            text=(
                "Titan Text Embeddings v2 produces 1024 dimensional vectors and is the "
                "default embedding model for a Bedrock Knowledge Base."
            ),
            source="s3://corpus/embeddings.txt",
            page=2,
        ),
        Document(
            doc_id="kb-003",
            text=(
                "Retrieval returns ranked passages with relevance scores and native page "
                "citations so an answer can be traced back to its source."
            ),
            source="s3://corpus/retrieval.txt",
            page=3,
        ),
    )
