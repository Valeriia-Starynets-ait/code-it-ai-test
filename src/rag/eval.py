"""RAGAS-style evaluation harness.

A credential-free, deterministic stand-in for a RAGAS + Langfuse run. It keeps
the shape the rag-eval-ragas-langfuse skill prescribes:

- retrieval and answer metrics are reported **separately** (a high answer score
  on poor retrieval usually means the model answered from its own weights);
- a +/-1 page tolerance absorbs off-by-one chunk boundaries;
- rows whose metric cannot be computed are **imputed to 1.0 and counted**, so an
  imputed row never silently reads as a genuine pass.

The metrics are token-overlap heuristics rather than LLM judgements, so the
harness runs with no network and no model. A real run would wrap the app's own
Bedrock model as the judge and write scores back onto Langfuse traces.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from rag.models import Answer

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CITATION_MARKER_RE = re.compile(r"\[\d+\]")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _answer_tokens(text: str) -> set[str]:
    """Tokenize an answer, dropping inline ``[n]`` citation markers.

    Markers are formatting, not claims, so they must not count against
    faithfulness or relevancy.
    """
    return _tokenize(_CITATION_MARKER_RE.sub(" ", text))


class EvalSample(BaseModel):
    """One frozen eval item: a question with its ground-truth expectations."""

    model_config = ConfigDict(frozen=True, strict=True)

    question: str = Field(..., min_length=1)
    expected_answer: str = Field(..., min_length=1)
    relevant_doc_ids: tuple[str, ...] = Field(..., min_length=1)
    expected_pages: tuple[int, ...] = Field(default_factory=tuple)


class SampleScore(BaseModel):
    """Per-question metrics, so a regression is visible per item, not just in the mean."""

    model_config = ConfigDict(frozen=True, strict=True)

    question: str
    context_recall: float = Field(..., ge=0.0, le=1.0)
    page_match: float = Field(..., ge=0.0, le=1.0)
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    answer_relevancy: float = Field(..., ge=0.0, le=1.0)
    imputed: bool = False


class EvalReport(BaseModel):
    """Aggregated eval result with retrieval and answer metrics kept apart."""

    model_config = ConfigDict(frozen=True, strict=True)

    sample_count: int = Field(..., ge=0)
    imputed_count: int = Field(..., ge=0)
    # Retrieval metrics.
    context_recall: float = Field(..., ge=0.0, le=1.0)
    page_match: float = Field(..., ge=0.0, le=1.0)
    # Answer metrics.
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    answer_relevancy: float = Field(..., ge=0.0, le=1.0)
    per_sample: tuple[SampleScore, ...] = Field(default_factory=tuple)

    def meets_thresholds(
        self,
        *,
        min_context_recall: float = 0.8,
        min_faithfulness: float = 0.8,
        min_answer_relevancy: float = 0.8,
    ) -> bool:
        """Gate: true only when every configured threshold is met."""
        return (
            self.context_recall >= min_context_recall
            and self.faithfulness >= min_faithfulness
            and self.answer_relevancy >= min_answer_relevancy
        )


def _context_recall(answer: Answer, relevant_doc_ids: Sequence[str]) -> float:
    retrieved_ids = {p.doc_id for p in answer.passages}
    hits = sum(1 for doc_id in relevant_doc_ids if doc_id in retrieved_ids)
    return hits / len(relevant_doc_ids)


def _page_match(answer: Answer, expected_pages: Sequence[int]) -> float | None:
    if not expected_pages:
        return None  # not computable -> caller imputes
    retrieved_pages = {p.page for p in answer.passages if p.page is not None}
    if not retrieved_pages:
        return 0.0
    hits = sum(1 for page in expected_pages if any(abs(page - rp) <= 1 for rp in retrieved_pages))
    return hits / len(expected_pages)


def _faithfulness(answer: Answer) -> float | None:
    """Fraction of answer tokens that are grounded in the retrieved passages."""
    answer_tokens = _answer_tokens(answer.answer.text)
    if not answer_tokens:
        return None
    context_tokens: set[str] = set()
    for passage in answer.passages:
        context_tokens |= _tokenize(passage.text)
    if not context_tokens:
        return 0.0
    grounded = answer_tokens & context_tokens
    return len(grounded) / len(answer_tokens)


def _answer_relevancy(answer: Answer, expected_answer: str) -> float | None:
    answer_tokens = _answer_tokens(answer.answer.text)
    expected_tokens = _tokenize(expected_answer)
    if not answer_tokens or not expected_tokens:
        return None
    overlap = answer_tokens & expected_tokens
    return len(overlap) / len(expected_tokens)


def _impute(value: float | None, imputed_flag: list[bool]) -> float:
    """Impute a non-computable metric to 1.0 and record that it was imputed."""
    if value is None:
        imputed_flag.append(True)
        return 1.0
    return value


def score_sample(answer: Answer, sample: EvalSample) -> SampleScore:
    """Score one answer against its ground truth, imputing non-computable metrics."""
    imputed: list[bool] = []
    recall = _impute(_context_recall(answer, sample.relevant_doc_ids), imputed)
    page = _impute(_page_match(answer, sample.expected_pages), imputed)
    faith = _impute(_faithfulness(answer), imputed)
    relevancy = _impute(_answer_relevancy(answer, sample.expected_answer), imputed)
    return SampleScore(
        question=sample.question,
        context_recall=recall,
        page_match=page,
        faithfulness=faith,
        answer_relevancy=relevancy,
        imputed=bool(imputed),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(answers: Sequence[Answer], samples: Sequence[EvalSample]) -> EvalReport:
    """Score a batch of answers against the frozen eval set and aggregate."""
    if len(answers) != len(samples):
        raise ValueError(f"answers ({len(answers)}) and samples ({len(samples)}) must align one to one")
    scores = [score_sample(answer, sample) for answer, sample in zip(answers, samples, strict=True)]
    imputed_count = sum(1 for s in scores if s.imputed)
    if imputed_count:
        logger.warning("%d of %d sample(s) had an imputed metric", imputed_count, len(scores))
    return EvalReport(
        sample_count=len(scores),
        imputed_count=imputed_count,
        context_recall=_mean([s.context_recall for s in scores]),
        page_match=_mean([s.page_match for s in scores]),
        faithfulness=_mean([s.faithfulness for s in scores]),
        answer_relevancy=_mean([s.answer_relevancy for s in scores]),
        per_sample=tuple(scores),
    )
