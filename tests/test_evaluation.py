import json
from pathlib import Path

import pytest

from ragkb.errors import EvaluationError
from ragkb.evaluation import answer_relevance, faithfulness, load_cases, run_eval
from ragkb.generation import ExtractiveGenerator
from ragkb.ingestion import ingest
from ragkb.models import Passage
from ragkb.retrieval import LocalRetriever, load_chunks


def test_faithfulness_is_high_for_supported_answer(passages: list[Passage]) -> None:
    assert faithfulness("Buckets are durable [1].", passages) == 1.0


def test_faithfulness_is_low_for_unsupported_claim(passages: list[Passage]) -> None:
    score = faithfulness("Rockets orbit planets quickly [1].", passages)
    assert score == 0.0


def test_faithfulness_scores_fraction_of_supported_sentences(passages: list[Passage]) -> None:
    score = faithfulness("Buckets are durable [1]. Rockets orbit planets quickly [1].", passages)
    assert score == 0.5


def test_faithfulness_of_empty_answer_is_zero(passages: list[Passage]) -> None:
    assert faithfulness("", passages) == 0.0


def test_answer_relevance_rewards_question_coverage() -> None:
    on_topic = answer_relevance("What does faithfulness measure?", "Faithfulness measures support [1].")
    off_topic = answer_relevance("What does faithfulness measure?", "Bananas are yellow [1].")
    assert on_topic > off_topic
    assert off_topic == 0.0


def test_load_cases_validates_file(tmp_path: Path) -> None:
    bad = tmp_path / "q.json"
    bad.write_text(json.dumps([{"nope": 1}]), encoding="utf-8")
    with pytest.raises(EvaluationError, match="invalid"):
        load_cases(bad)
    with pytest.raises(EvaluationError, match="not found"):
        load_cases(tmp_path / "missing.json")


def test_run_eval_on_sample_data_meets_threshold(sample_docs: Path, sample_questions: Path, tmp_path: Path) -> None:
    ingest(sample_docs, tmp_path / "kb", chunk_words=60, overlap_words=10)
    retriever = LocalRetriever(load_chunks(tmp_path / "kb"))

    report = run_eval(load_cases(sample_questions), retriever, ExtractiveGenerator(), top_k=3)

    assert len(report.results) == 5
    assert report.mean_faithfulness >= 0.8
    assert report.mean_answer_relevance >= 0.5
    assert report.source_hit_rate == 1.0
    assert report.passes(0.5)
    assert not report.passes(1.01)
