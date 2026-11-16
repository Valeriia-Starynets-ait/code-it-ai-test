import json
from pathlib import Path

import pytest

from kbrag.errors import EvaluationError
from kbrag.evaluation import answer_relevance, faithfulness, load_eval_set, run_eval
from kbrag.generation import ExtractiveGenerator
from kbrag.models import Chunk, Passage
from kbrag.retrieval import InMemoryRetriever


def test_faithfulness_full_when_answer_comes_from_passages(passages: list[Passage]) -> None:
    answer = "Full time employees accrue twenty days of paid vacation per year [2]."
    assert faithfulness(answer, passages) == pytest.approx(1.0)


def test_faithfulness_low_for_hallucinated_claim(passages: list[Passage]) -> None:
    assert faithfulness("The cafeteria serves sushi on Fridays [1].", passages) < 0.3


def test_faithfulness_empty_answer_is_zero(passages: list[Passage]) -> None:
    assert faithfulness("", passages) == 0.0


def test_answer_relevance_high_when_answer_covers_question() -> None:
    assert answer_relevance("How many vacation days per year?", "Twenty vacation days per year are granted.") > 0.7


def test_answer_relevance_low_for_off_topic() -> None:
    assert answer_relevance("How many vacation days per year?", "Laptops ship one week early.") < 0.3


def test_load_eval_set_validates(tmp_path: Path) -> None:
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"question": "q1"}, {"question": ""}]), encoding="utf-8")
    with pytest.raises(EvaluationError):
        load_eval_set(path)


def test_run_eval_reports_means_and_threshold(chunks: list[Chunk]) -> None:
    items = [{"question": "How many vacation days per year?"}, {"question": "When does the laptop ship?"}]
    from kbrag.models import EvalItem

    report = run_eval([EvalItem(**i) for i in items], InMemoryRetriever(chunks), ExtractiveGenerator(), top_k=2)
    assert len(report.results) == 2
    assert 0.0 <= report.mean_faithfulness <= 1.0
    assert report.passed(0.0)
    assert not report.passed(1.01)


def test_run_eval_scores_unanswerable_question_as_zero(chunks: list[Chunk]) -> None:
    from kbrag.models import EvalItem

    report = run_eval([EvalItem(question="zzzz qqqq")], InMemoryRetriever(chunks), ExtractiveGenerator(), top_k=2)
    assert report.results[0].faithfulness == 0.0
    assert report.results[0].answered is False
