import json
import logging
import re
from pathlib import Path

from pydantic import ValidationError

from kbrag.errors import EvaluationError, NoEvidenceError
from kbrag.generation import Generator, answer_question
from kbrag.models import EvalItem, EvalReport, EvalResult, Passage
from kbrag.retrieval import Retriever
from kbrag.text import content_tokens

logger = logging.getLogger(__name__)

_MARKER = re.compile(r"\[\d+\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def load_eval_set(path: Path) -> list[EvalItem]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [EvalItem(**entry) for entry in raw]
    except (OSError, ValueError, TypeError, ValidationError) as exc:
        raise EvaluationError(f"Invalid evaluation set {path}: {exc}") from exc


def faithfulness(answer: str, passages: list[Passage]) -> float:
    """Mean share of each answer sentence's content tokens that appear in the retrieved passages."""
    context = set().union(*(content_tokens(p.text) for p in passages)) if passages else set()
    scores = []
    for sentence in _SENTENCE.split(_MARKER.sub("", answer).strip()):
        tokens = content_tokens(sentence)
        if tokens:
            scores.append(len(tokens & context) / len(tokens))
    return sum(scores) / len(scores) if scores else 0.0


def answer_relevance(question: str, answer: str) -> float:
    """Share of the question's content tokens covered by the answer."""
    wanted = content_tokens(question)
    if not wanted:
        return 0.0
    return len(wanted & content_tokens(_MARKER.sub("", answer))) / len(wanted)


def run_eval(items: list[EvalItem], retriever: Retriever, generator: Generator, top_k: int = 5) -> EvalReport:
    results = []
    for item in items:
        try:
            answer = answer_question(item.question, retriever, generator, top_k)
        except NoEvidenceError:
            logger.warning("No evidence for question: %s", item.question)
            results.append(EvalResult(question=item.question, answered=False, faithfulness=0.0, answer_relevance=0.0))
            continue
        results.append(
            EvalResult(
                question=item.question,
                answered=True,
                faithfulness=faithfulness(answer.text, answer.passages),
                answer_relevance=answer_relevance(item.question, answer.text),
            )
        )
    return EvalReport(results=results)
