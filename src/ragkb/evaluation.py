import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from ragkb.errors import CitationError, EvaluationError, NoEvidenceError
from ragkb.generation import Generator, answer_question
from ragkb.models import Passage
from ragkb.retrieval import Retriever
from ragkb.text import content_tokens, split_sentences

SENTENCE_SUPPORT = 0.8


class EvalCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    expected_source: str


class CaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    answer: str
    faithfulness: float
    answer_relevance: float
    source_hit: bool


class EvalReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    results: list[CaseResult]

    def _mean(self, values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @property
    def mean_faithfulness(self) -> float:
        return self._mean([r.faithfulness for r in self.results])

    @property
    def mean_answer_relevance(self) -> float:
        return self._mean([r.answer_relevance for r in self.results])

    @property
    def source_hit_rate(self) -> float:
        return self._mean([1.0 if r.source_hit else 0.0 for r in self.results])

    def passes(self, threshold: float) -> bool:
        return bool(self.results) and min(self.mean_faithfulness, self.mean_answer_relevance) >= threshold


def faithfulness(answer: str, passages: list[Passage]) -> float:
    """Share of answer sentences whose content terms are supported by the retrieved passages."""
    context = content_tokens(" ".join(p.text for p in passages))
    scored = [tokens for s in split_sentences(answer) if (tokens := content_tokens(s))]
    if not scored:
        return 0.0
    supported = sum(1 for tokens in scored if len(tokens & context) / len(tokens) >= SENTENCE_SUPPORT)
    return supported / len(scored)


def answer_relevance(question: str, answer: str) -> float:
    """Share of the question's content terms that the answer addresses."""
    terms = content_tokens(question)
    if not terms:
        return 0.0
    return len(terms & content_tokens(answer)) / len(terms)


def load_cases(path: Path) -> list[EvalCase]:
    if not path.is_file():
        raise EvaluationError(f"questions file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        cases = [EvalCase.model_validate(item) for item in raw]
    except (ValueError, TypeError, ValidationError) as exc:
        raise EvaluationError(f"invalid questions file {path}: {exc}") from exc
    if not cases:
        raise EvaluationError(f"questions file {path} is empty")
    return cases


def _evaluate_case(case: EvalCase, retriever: Retriever, generator: Generator, top_k: int) -> CaseResult:
    try:
        answer = answer_question(case.question, retriever, generator, top_k)
    except (NoEvidenceError, CitationError):
        return CaseResult(question=case.question, answer="", faithfulness=0.0, answer_relevance=0.0, source_hit=False)
    return CaseResult(
        question=case.question,
        answer=answer.text,
        faithfulness=faithfulness(answer.text, answer.passages),
        answer_relevance=answer_relevance(case.question, answer.text),
        source_hit=case.expected_source in {c.source for c in answer.citations},
    )


def run_eval(cases: list[EvalCase], retriever: Retriever, generator: Generator, top_k: int) -> EvalReport:
    return EvalReport(results=[_evaluate_case(c, retriever, generator, top_k) for c in cases])
