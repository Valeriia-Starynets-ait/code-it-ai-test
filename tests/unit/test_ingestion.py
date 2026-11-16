import json
from pathlib import Path

import pytest

from kbrag.errors import IngestionError
from kbrag.ingestion import chunk_document, load_documents, prepare_data_source
from kbrag.models import Document


def test_load_documents_reads_supported_files_with_relative_sources(docs_dir: Path) -> None:
    docs = load_documents(docs_dir)
    assert [d.source for d in docs] == ["hr/vacation.txt", "onboarding.md"]
    assert all(d.text for d in docs)


def test_load_documents_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        load_documents(tmp_path / "nope")


def test_load_documents_empty_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        load_documents(tmp_path)


def test_chunk_document_respects_size_and_overlap() -> None:
    doc = Document(doc_id="d", source="d.md", text=" ".join(f"w{i}" for i in range(25)))
    chunks = chunk_document(doc, chunk_words=10, overlap_words=2)
    assert [len(c.text.split()) for c in chunks] == [10, 10, 9]
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]
    assert [c.chunk_id for c in chunks] == ["d-0000", "d-0001", "d-0002"]
    assert {c.source for c in chunks} == {"d.md"}


def test_chunk_document_rejects_bad_overlap() -> None:
    doc = Document(doc_id="d", source="d.md", text="a b c")
    with pytest.raises(IngestionError):
        chunk_document(doc, chunk_words=5, overlap_words=5)


def test_prepare_data_source_writes_chunks_metadata_and_manifest(docs_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    chunks = [c for d in load_documents(docs_dir) for c in chunk_document(d)]
    prepare_data_source(chunks, out)

    first = chunks[0]
    assert (out / f"{first.chunk_id}.txt").read_text(encoding="utf-8") == first.text
    meta = json.loads((out / f"{first.chunk_id}.txt.metadata.json").read_text(encoding="utf-8"))
    assert meta["metadataAttributes"]["source"] == first.source
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert sorted(manifest["sources"]) == ["hr/vacation.txt", "onboarding.md"]
    assert len(manifest["chunks"]) == len(chunks)


def test_load_documents_turns_markdown_headings_into_sentences(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# Vacation policy\nTwenty days.", encoding="utf-8")
    assert load_documents(tmp_path)[0].text == "Vacation policy.\nTwenty days."
