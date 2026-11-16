from unittest.mock import MagicMock

import pytest

from kbrag.errors import CitationError, NoEvidenceError
from kbrag.generation import BedrockGenerator, ExtractiveGenerator, answer_question, verify_citations
from kbrag.models import Answer, Citation, Passage
from kbrag.retrieval import InMemoryRetriever


class _Fixed:
    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, question: str, passages: list[Passage]) -> str:  # noqa: ARG002
        return self._text


def test_answer_has_citations_mapping_to_sources(chunks: list) -> None:  # type: ignore[type-arg]
    answer = answer_question("How many vacation days per year?", InMemoryRetriever(chunks), ExtractiveGenerator())
    assert answer.citations
    assert "[1]" in answer.text
    assert answer.citations[0].source == "hr/vacation.txt"
    verify_citations(answer, {"onboarding.md", "hr/vacation.txt"})


def test_no_passages_raises_no_evidence(chunks: list) -> None:  # type: ignore[type-arg]
    with pytest.raises(NoEvidenceError):
        answer_question("zzzz qqqq", InMemoryRetriever(chunks), ExtractiveGenerator())


def test_generator_without_markers_raises_citation_error(passages: list[Passage]) -> None:
    retriever = MagicMock()
    retriever.retrieve.return_value = passages
    with pytest.raises(CitationError):
        answer_question("q", retriever, _Fixed("Uncited claim."))


def test_generator_with_out_of_range_marker_raises(passages: list[Passage]) -> None:
    retriever = MagicMock()
    retriever.retrieve.return_value = passages
    with pytest.raises(CitationError):
        answer_question("q", retriever, _Fixed("Claim [9]."))


def test_verify_citations_rejects_unknown_source() -> None:
    answer = Answer(
        question="q",
        text="x [1]",
        citations=[Citation(marker=1, source="ghost.md", chunk_id="g-0", quote="x")],
        passages=[],
    )
    with pytest.raises(CitationError):
        verify_citations(answer, {"real.md"})


def test_bedrock_generator_calls_converse_with_numbered_context(passages: list[Passage]) -> None:
    client = MagicMock()
    client.converse.return_value = {"output": {"message": {"content": [{"text": "Twenty days [2]."}]}}}
    text = BedrockGenerator(client, model_id="model-x").generate("How many days?", passages)

    assert text == "Twenty days [2]."
    kwargs = client.converse.call_args.kwargs
    assert kwargs["modelId"] == "model-x"
    prompt = kwargs["messages"][0]["content"][0]["text"]
    assert "[1]" in prompt and "[2]" in prompt and "How many days?" in prompt


def test_bedrock_generator_malformed_response_raises(passages: list[Passage]) -> None:
    client = MagicMock()
    client.converse.return_value = {"output": {}}
    with pytest.raises(CitationError):
        BedrockGenerator(client, model_id="m").generate("q", passages)


def test_extractive_generator_skips_weak_passages() -> None:
    strong = Passage(text="Twenty days of vacation per year.", score=0.8, source="v.md", chunk_id="v-0")
    weak = Passage(text="Laptops ship early. Another day passes.", score=0.2, source="o.md", chunk_id="o-0")
    text = ExtractiveGenerator().generate("How many vacation days per year?", [strong, weak])
    assert "[2]" not in text
    assert "[1]" in text
