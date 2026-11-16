import json
from pathlib import Path

import pytest

from ragkb.errors import IngestionError
from ragkb.ingestion import chunk_document, ingest, load_documents
from ragkb.models import Document


def test_load_documents_reads_supported_files_with_relative_sources(docs_dir: Path) -> None:
    (docs_dir / "nested").mkdir()
    (docs_dir / "nested" / "gamma.md").write_text("Gamma text.", encoding="utf-8")
    (docs_dir / "ignored.png").write_bytes(b"\x89PNG")

    docs = load_documents(docs_dir)

    assert [d.source for d in docs] == ["alpha.md", "beta.txt", "nested/gamma.md"]


def test_load_documents_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="not found"):
        load_documents(tmp_path / "nope")


def test_load_documents_empty_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="no supported documents"):
        load_documents(tmp_path)


def test_chunk_document_overlaps_and_keeps_source() -> None:
    doc = Document(source="a.md", text=" ".join(f"w{i}" for i in range(10)))

    chunks = chunk_document(doc, chunk_words=4, overlap_words=1)

    assert [c.chunk_id for c in chunks] == ["a.md#0", "a.md#1", "a.md#2"]
    assert chunks[0].text == "w0 w1 w2 w3"
    assert chunks[1].text.startswith("w3 ")
    assert chunks[-1].text.endswith("w9")
    assert {c.source for c in chunks} == {"a.md"}


def test_chunk_document_short_text_is_single_chunk() -> None:
    chunks = chunk_document(Document(source="a.md", text="just a few words"), chunk_words=50, overlap_words=5)
    assert len(chunks) == 1


def test_chunk_document_rejects_overlap_not_smaller_than_size() -> None:
    with pytest.raises(IngestionError, match="overlap"):
        chunk_document(Document(source="a.md", text="x y z"), chunk_words=3, overlap_words=3)


def test_ingest_writes_bedrock_data_source_files(docs_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "kb"

    result = ingest(docs_dir, out, chunk_words=50, overlap_words=5)

    assert result.document_count == 2
    assert result.chunk_count == 2
    chunk_files = sorted((out / "data").glob("*.txt"))
    assert len(chunk_files) == 2
    sidecar = json.loads(Path(f"{chunk_files[0]}.metadata.json").read_text(encoding="utf-8"))
    assert sidecar["metadataAttributes"]["source"] == "alpha.md"
    manifest = [json.loads(line) for line in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [m["source"] for m in manifest] == ["alpha.md", "beta.txt"]
    assert manifest[0]["text"].startswith("Alpha systems")


def test_ingest_is_idempotent_and_removes_stale_chunks(docs_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "kb"
    ingest(docs_dir, out, chunk_words=50, overlap_words=5)
    (docs_dir / "beta.txt").unlink()

    result = ingest(docs_dir, out, chunk_words=50, overlap_words=5)

    assert result.chunk_count == 1
    assert len(list((out / "data").glob("*.txt"))) == 1
