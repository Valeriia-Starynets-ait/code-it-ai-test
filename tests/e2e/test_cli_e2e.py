"""End-to-end tests driving the CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.cli import main

pytestmark = pytest.mark.e2e


def test_cli_ingest_sample(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["ingest"])
    assert code == 0
    assert "ingested 3 document(s)" in capsys.readouterr().out


def test_cli_ingest_from_jsonl(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"doc_id": "x", "text": "hello world", "source": "s3://x"}),
        encoding="utf-8",
    )
    code = main(["ingest", "--corpus", str(corpus)])
    assert code == 0
    assert "ingested 1 document(s)" in capsys.readouterr().out


def test_cli_ask_prints_citations(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["ask", "which embedding model and how many dimensions"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Citations:" in out
    assert "[1]" in out


def test_cli_eval_gate_passes(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["eval", "--gate"])
    assert code == 0
    out = capsys.readouterr().out
    assert "gate: PASS" in out
    assert "context_recall" in out
    assert "faithfulness" in out


def test_cli_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        main([])
