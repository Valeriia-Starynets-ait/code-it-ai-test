from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ragkb import cli
from ragkb.config import Settings


def _ingest(sample_docs: Path, out: Path) -> None:
    assert cli.main(["ingest", "--docs", str(sample_docs), "--out", str(out)]) == 0


def test_ingest_command_reports_counts(sample_docs: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _ingest(sample_docs, tmp_path / "kb")

    out = capsys.readouterr().out
    assert "3 documents" in out
    assert (tmp_path / "kb" / "manifest.jsonl").exists()


def test_ask_prints_answer_and_sources(sample_docs: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _ingest(sample_docs, tmp_path / "kb")
    capsys.readouterr()

    code = cli.main(["ask", "What does faithfulness measure?", "--index", str(tmp_path / "kb")])

    out = capsys.readouterr().out
    assert code == 0
    assert "Sources:" in out
    assert "evaluation.md" in out


def test_ask_without_evidence_exits_non_zero(
    sample_docs: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _ingest(sample_docs, tmp_path / "kb")

    code = cli.main(["ask", "zebra xylophone", "--index", str(tmp_path / "kb")])

    assert code == 1
    assert "no relevant passages" in capsys.readouterr().err


def test_eval_prints_scores_and_passes(
    sample_docs: Path, sample_questions: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _ingest(sample_docs, tmp_path / "kb")

    code = cli.main(["eval", "--questions", str(sample_questions), "--index", str(tmp_path / "kb")])

    out = capsys.readouterr().out
    assert code == 0
    assert "faithfulness" in out
    assert "answer_relevance" in out
    assert "PASS" in out


def test_eval_exits_non_zero_below_threshold(
    sample_docs: Path, sample_questions: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _ingest(sample_docs, tmp_path / "kb")

    code = cli.main(
        ["eval", "--questions", str(sample_questions), "--index", str(tmp_path / "kb"), "--threshold", "1.01"]
    )

    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_missing_index_is_a_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["ask", "anything", "--index", str(tmp_path / "absent")])

    assert code == 2
    assert "manifest" in capsys.readouterr().err


def test_bedrock_backend_requires_knowledge_base_id(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["ask", "q", "--index", str(tmp_path), "--backend", "bedrock"])

    assert code == 2
    assert "RAG_KNOWLEDGE_BASE_ID" in capsys.readouterr().err


def test_bedrock_backend_builds_clients_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_boto3 = MagicMock()
    monkeypatch.setattr(cli, "boto3", fake_boto3)
    settings = Settings(backend="bedrock", knowledge_base_id="KB1", aws_region="eu-west-1")

    retriever, generator = cli.build_pipeline(settings, index=Path("unused"))

    fake_boto3.client.assert_any_call("bedrock-agent-runtime", region_name="eu-west-1")
    fake_boto3.client.assert_any_call("bedrock-runtime", region_name="eu-west-1")
    assert type(retriever).__name__ == "BedrockRetriever"
    assert type(generator).__name__ == "BedrockGenerator"
