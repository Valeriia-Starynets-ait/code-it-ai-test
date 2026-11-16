import json
from pathlib import Path

import pytest

from kbrag.cli import main


def _write_questions(path: Path, questions: list[str]) -> Path:
    path.write_text(json.dumps([{"question": q} for q in questions]), encoding="utf-8")
    return path


def test_ingest_then_ask_prints_answer_with_sources(
    docs_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "ds"
    assert main(["ingest", "--docs", str(docs_dir), "--out", str(out)]) == 0
    assert (out / "manifest.json").exists()

    assert main(["ask", "How many vacation days per year?", "--data", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "[1]" in printed
    assert "hr/vacation.txt" in printed


def test_ask_without_evidence_exits_nonzero(docs_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "ds"
    main(["ingest", "--docs", str(docs_dir), "--out", str(out)])
    assert main(["ask", "zzzz qqqq", "--data", str(out)]) == 2
    assert "No supporting passages" in capsys.readouterr().err


def test_ingest_missing_folder_exits_nonzero(tmp_path: Path) -> None:
    assert main(["ingest", "--docs", str(tmp_path / "x"), "--out", str(tmp_path / "o")]) == 1


def test_eval_prints_scores_and_passes(docs_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "ds"
    main(["ingest", "--docs", str(docs_dir), "--out", str(out)])
    questions = _write_questions(tmp_path / "q.json", ["How many vacation days per year?"])
    code = main(["eval", "--data", str(out), "--questions", str(questions), "--threshold", "0.5"])
    printed = capsys.readouterr().out
    assert code == 0
    assert "faithfulness" in printed and "answer_relevance" in printed


def test_eval_exits_nonzero_below_threshold(docs_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "ds"
    main(["ingest", "--docs", str(docs_dir), "--out", str(out)])
    questions = _write_questions(tmp_path / "q.json", ["zzzz qqqq"])
    assert main(["eval", "--data", str(out), "--questions", str(questions), "--threshold", "0.5"]) == 1


def test_bedrock_backend_requires_kb_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KBRAG_KNOWLEDGE_BASE_ID", raising=False)
    assert main(["ask", "q", "--data", str(tmp_path), "--backend", "bedrock"]) == 1
