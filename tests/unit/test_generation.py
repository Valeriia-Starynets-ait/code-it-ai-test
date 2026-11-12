from __future__ import annotations

import pytest

from rag.exceptions import GenerationError
from rag.generation import TemplateGenerator
from rag.models import RetrievedPassage

pytestmark = pytest.mark.unit


def _passage(doc_id: str, text: str, page: int | None = None) -> RetrievedPassage:
    return RetrievedPassage(doc_id=doc_id, text=text, source=f"s3://{doc_id}", score=0.9, page=page)


def test_generate_attaches_a_citation_per_passage() -> None:
    passages = [_passage("a", "Alpha fact", 1), _passage("b", "Beta fact", 2)]
    answer = TemplateGenerator().generate("q", passages)
    assert len(answer.citations) == 2
    assert [c.marker for c in answer.citations] == [1, 2]
    assert [c.doc_id for c in answer.citations] == ["a", "b"]


def test_generate_embeds_inline_markers() -> None:
    answer = TemplateGenerator().generate("q", [_passage("a", "Alpha fact")])
    assert "[1]" in answer.text


def test_generate_carries_page_into_citation() -> None:
    answer = TemplateGenerator().generate("q", [_passage("a", "Alpha fact", 7)])
    assert answer.citations[0].page == 7


def test_generate_respects_max_passages() -> None:
    passages = [_passage("a", "Alpha"), _passage("b", "Beta"), _passage("c", "Gamma")]
    answer = TemplateGenerator(max_passages=2).generate("q", passages)
    assert len(answer.citations) == 2


def test_generate_raises_without_passages() -> None:
    with pytest.raises(GenerationError):
        TemplateGenerator().generate("q", [])


def test_generate_raises_on_empty_query() -> None:
    with pytest.raises(GenerationError):
        TemplateGenerator().generate("  ", [_passage("a", "Alpha")])


def test_bad_max_passages_raises() -> None:
    with pytest.raises(GenerationError):
        TemplateGenerator(max_passages=0)
