from __future__ import annotations

import pytest

from rag.eval import EvalReport, EvalSample, evaluate, score_sample
from rag.models import Answer, Citation, GeneratedAnswer, RetrievedPassage

pytestmark = pytest.mark.unit


def _answer(
    question: str,
    text: str,
    passages: tuple[RetrievedPassage, ...],
) -> Answer:
    citations = tuple(
        Citation(marker=i, doc_id=p.doc_id, source=p.source, page=p.page) for i, p in enumerate(passages, start=1)
    )
    return Answer(
        question=question,
        answer=GeneratedAnswer(text=text, citations=citations),
        passages=passages,
    )


def _passage(doc_id: str, text: str, page: int | None = None) -> RetrievedPassage:
    return RetrievedPassage(doc_id=doc_id, text=text, source=f"s3://{doc_id}", score=0.9, page=page)


def test_perfect_retrieval_and_answer_scores_high() -> None:
    passage = _passage("a", "titan embeddings produce 1024 dimensions", 2)
    answer = _answer("q", "titan embeddings produce 1024 dimensions [1].", (passage,))
    sample = EvalSample(
        question="q",
        expected_answer="titan embeddings produce 1024 dimensions",
        relevant_doc_ids=("a",),
        expected_pages=(2,),
    )
    score = score_sample(answer, sample)
    assert score.context_recall == 1.0
    assert score.page_match == 1.0
    assert score.faithfulness == 1.0
    assert score.answer_relevancy == 1.0
    assert score.imputed is False


def test_missing_relevant_doc_lowers_context_recall() -> None:
    passage = _passage("other", "unrelated text", 9)
    answer = _answer("q", "unrelated text [1].", (passage,))
    sample = EvalSample(
        question="q",
        expected_answer="expected answer",
        relevant_doc_ids=("a", "b"),
        expected_pages=(1,),
    )
    score = score_sample(answer, sample)
    assert score.context_recall == 0.0


def test_page_match_tolerates_off_by_one() -> None:
    passage = _passage("a", "fact", 3)
    answer = _answer("q", "fact [1].", (passage,))
    sample = EvalSample(
        question="q",
        expected_answer="fact",
        relevant_doc_ids=("a",),
        expected_pages=(2,),  # one off from the retrieved page 3
    )
    score = score_sample(answer, sample)
    assert score.page_match == 1.0


def test_missing_expected_pages_imputes_and_flags() -> None:
    passage = _passage("a", "fact", 1)
    answer = _answer("q", "fact [1].", (passage,))
    sample = EvalSample(
        question="q",
        expected_answer="fact",
        relevant_doc_ids=("a",),
        expected_pages=(),  # not computable -> imputed to 1.0
    )
    score = score_sample(answer, sample)
    assert score.page_match == 1.0
    assert score.imputed is True


def test_evaluate_aggregates_and_counts_imputed() -> None:
    passage = _passage("a", "grounded fact about titan", 1)
    answer = _answer("q", "grounded fact about titan [1].", (passage,))
    samples = (
        EvalSample(
            question="q",
            expected_answer="grounded fact about titan",
            relevant_doc_ids=("a",),
            expected_pages=(),  # imputed
        ),
    )
    report = evaluate([answer], samples)
    assert isinstance(report, EvalReport)
    assert report.sample_count == 1
    assert report.imputed_count == 1
    assert report.meets_thresholds() is True


def test_evaluate_rejects_misaligned_lengths() -> None:
    with pytest.raises(ValueError):
        evaluate(
            [],
            (
                EvalSample(
                    question="q",
                    expected_answer="a",
                    relevant_doc_ids=("a",),
                ),
            ),
        )


def test_meets_thresholds_fails_below_gate() -> None:
    report = EvalReport(
        sample_count=1,
        imputed_count=0,
        context_recall=0.5,
        page_match=1.0,
        faithfulness=1.0,
        answer_relevancy=1.0,
    )
    assert report.meets_thresholds() is False
