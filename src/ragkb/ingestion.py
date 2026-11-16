import json
import logging
import re
import shutil
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ragkb.errors import IngestionError
from ragkb.models import Chunk, Document

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = frozenset({".md", ".txt"})
DATA_DIR = "data"
MANIFEST_NAME = "manifest.jsonl"
_HEADING = re.compile(r"^#+\s*(.+?)\s*$", re.MULTILINE)


class IngestResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_count: int
    chunk_count: int
    output_dir: Path


def _normalise(path: Path, raw: str) -> str:
    if path.suffix == ".md":
        return _HEADING.sub(r"\1.", raw)
    return raw


def load_documents(folder: Path) -> list[Document]:
    if not folder.is_dir():
        raise IngestionError(f"documents folder not found: {folder}")
    paths = sorted(
        (p for p in folder.rglob("*") if p.is_file() and p.suffix in SUPPORTED_SUFFIXES),
        key=lambda p: p.relative_to(folder).as_posix(),
    )
    if not paths:
        raise IngestionError(f"no supported documents ({', '.join(sorted(SUPPORTED_SUFFIXES))}) in {folder}")
    return [
        Document(source=p.relative_to(folder).as_posix(), text=_normalise(p, p.read_text(encoding="utf-8")))
        for p in paths
    ]


def chunk_document(doc: Document, chunk_words: int, overlap_words: int) -> list[Chunk]:
    if chunk_words < 1 or not 0 <= overlap_words < chunk_words:
        raise IngestionError(f"overlap ({overlap_words}) must be smaller than chunk size ({chunk_words})")
    words = doc.text.split()
    step = chunk_words - overlap_words
    chunks: list[Chunk] = []
    for index, start in enumerate(range(0, max(len(words), 1), step)):
        end = start + chunk_words
        chunks.append(
            Chunk(chunk_id=f"{doc.source}#{index}", source=doc.source, index=index, text=" ".join(words[start:end]))
        )
        if end >= len(words):
            break
    return chunks


def _chunk_filename(chunk: Chunk) -> str:
    return f"{chunk.source.replace('/', '__')}-{chunk.index}.txt"


def _write_chunk(data_dir: Path, chunk: Chunk) -> None:
    path = data_dir / _chunk_filename(chunk)
    path.write_text(chunk.text, encoding="utf-8")
    sidecar = {"metadataAttributes": {"source": chunk.source, "chunk_id": chunk.chunk_id, "chunk_index": chunk.index}}
    Path(f"{path}.metadata.json").write_text(json.dumps(sidecar), encoding="utf-8")


def ingest(docs_dir: Path, out_dir: Path, chunk_words: int, overlap_words: int) -> IngestResult:
    """Chunk documents and stage them as an S3-ready Bedrock data source (one file plus metadata per chunk)."""
    documents = load_documents(docs_dir)
    chunks = [c for d in documents for c in chunk_document(d, chunk_words, overlap_words)]
    data_dir = out_dir / DATA_DIR
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)
    for chunk in chunks:
        _write_chunk(data_dir, chunk)
    lines = (json.dumps(c.model_dump(), ensure_ascii=False) for c in chunks)
    (out_dir / MANIFEST_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("staged %s chunks from %s documents in %s", len(chunks), len(documents), out_dir)
    return IngestResult(document_count=len(documents), chunk_count=len(chunks), output_dir=out_dir)
