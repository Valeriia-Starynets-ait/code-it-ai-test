import json
import logging
from pathlib import Path

from kbrag.errors import IngestionError
from kbrag.models import Chunk, Document

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = frozenset({".md", ".txt"})
DEFAULT_CHUNK_WORDS = 120
DEFAULT_OVERLAP_WORDS = 20
MANIFEST_NAME = "manifest.json"
METADATA_SUFFIX = ".metadata.json"


def load_documents(folder: Path) -> list[Document]:
    if not folder.is_dir():
        raise IngestionError(f"Documents folder not found: {folder}")
    paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not paths:
        raise IngestionError(f"No .md or .txt documents under {folder}")
    docs = [
        Document(doc_id=_doc_id(p.relative_to(folder)), source=p.relative_to(folder).as_posix(), text=_read(p))
        for p in paths
    ]
    logger.info("Loaded %d documents from %s", len(docs), folder)
    return docs


def chunk_document(
    doc: Document, chunk_words: int = DEFAULT_CHUNK_WORDS, overlap_words: int = DEFAULT_OVERLAP_WORDS
) -> list[Chunk]:
    if chunk_words < 1 or not 0 <= overlap_words < chunk_words:
        raise IngestionError(f"Invalid chunking: chunk_words={chunk_words}, overlap_words={overlap_words}")
    words = doc.text.split()
    step = chunk_words - overlap_words
    chunks: list[Chunk] = []
    for index, start in enumerate(range(0, max(len(words), 1), step)):
        window = words[start : start + chunk_words]
        if not window:
            break
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}-{index:04d}",
                doc_id=doc.doc_id,
                source=doc.source,
                index=index,
                text=" ".join(window),
            )
        )
        if start + chunk_words >= len(words):
            break
    return chunks


def prepare_data_source(chunks: list[Chunk], out_dir: Path) -> None:
    """Write chunk files with Bedrock metadata sidecars plus a manifest.

    Sync the folder to the S3 bucket behind the knowledge base data source and configure that
    data source with chunking strategy NONE so Bedrock keeps these chunk boundaries.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        file_name = f"{chunk.chunk_id}.txt"
        (out_dir / file_name).write_text(chunk.text, encoding="utf-8")
        metadata = {"metadataAttributes": {"source": chunk.source, "doc_id": chunk.doc_id, "chunk_index": chunk.index}}
        (out_dir / f"{file_name}{METADATA_SUFFIX}").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest = {
        "sources": sorted({c.source for c in chunks}),
        "chunks": [c.model_dump(exclude={"text"}) for c in chunks],
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Prepared %d chunks in %s", len(chunks), out_dir)


def load_manifest(data_dir: Path) -> dict[str, list[str]]:
    path = data_dir / MANIFEST_NAME
    if not path.is_file():
        raise IngestionError(f"No {MANIFEST_NAME} in {data_dir}; run ingest first")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {"sources": list(raw["sources"])}


def load_chunks(data_dir: Path) -> list[Chunk]:
    path = data_dir / MANIFEST_NAME
    if not path.is_file():
        raise IngestionError(f"No {MANIFEST_NAME} in {data_dir}; run ingest first")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Chunk(**entry, text=(data_dir / f"{entry['chunk_id']}.txt").read_text(encoding="utf-8"))
        for entry in raw["chunks"]
    ]


def _read(path: Path) -> str:
    try:
        return _normalize_headings(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise IngestionError(f"Cannot read {path}: {exc}") from exc


def _normalize_headings(text: str) -> str:
    """Turn markdown heading lines into standalone sentences so chunking keeps them separate."""
    lines = []
    for line in text.splitlines():
        heading = line.lstrip("#").strip() if line.startswith("#") else None
        lines.append(f"{heading.rstrip('.')}." if heading else line)
    return "\n".join(lines)


def _doc_id(relative: Path) -> str:
    return relative.with_suffix("").as_posix().replace("/", "-")
