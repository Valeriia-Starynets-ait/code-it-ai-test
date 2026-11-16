from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from ragkb.errors import CitationError, GenerationError, NoEvidenceError
from ragkb.generation import BedrockGenerator, ExtractiveGenerator, answer_question
from ragkb.models import Passage


def _client_error(message: str) -> ClientError:
    return ClientError({"Error": {"Code": "ThrottlingException", "Message": message}}, "Operation")


class StubRetriever:
    def __init__(self, passages: list[Passage]) -> None:
        self._passages = passages

    def retrieve(self, query: str, top_k: int) -> list[Passage]:
        return self._passages[:top_k]


class FixedGenerator:
    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, question: str, passages: list[Passage]) -> str:
        return self._text


def test_answer_has_citations_mapping_to_retrieved_sources(passages: list[Passage]) -> None:
    answer = answer_question("Where are buckets stored?", StubRetriever(passages), ExtractiveGenerator(), top_k=2)

    assert answer.citations
    assert {c.source for c in answer.citations} <= {p.source for p in passages}
    assert "[1]" in answer.text


def test_answer_citations_follow_markers_in_generated_text(passages: list[Passage]) -> None:
    answer = answer_question("q", StubRetriever(passages), FixedGenerator("Beta moves events [2]."), top_k=2)

    assert [(c.marker, c.source, c.chunk_id) for c in answer.citations] == [(2, "beta.txt", "beta.txt#0")]


def test_answer_without_citation_marker_is_rejected(passages: list[Passage]) -> None:
    with pytest.raises(CitationError, match="no citations"):
        answer_question("q", StubRetriever(passages), FixedGenerator("Uncited claim."), top_k=2)


def test_answer_with_unknown_marker_is_rejected(passages: list[Passage]) -> None:
    with pytest.raises(CitationError, match=r"\[7\]"):
        answer_question("q", StubRetriever(passages), FixedGenerator("Made up [7]."), top_k=2)


def test_answer_with_no_passages_raises_no_evidence() -> None:
    with pytest.raises(NoEvidenceError):
        answer_question("q", StubRetriever([]), ExtractiveGenerator(), top_k=3)


def test_known_sources_reject_unmapped_citation(passages: list[Passage]) -> None:
    with pytest.raises(CitationError, match="alpha.md"):
        answer_question("q", StubRetriever(passages), FixedGenerator("Claim [1]."), top_k=2, known_sources={"beta.txt"})


def test_bedrock_generator_numbers_passages_and_returns_text(passages: list[Passage]) -> None:
    client = MagicMock()
    client.converse.return_value = {"output": {"message": {"content": [{"text": "Durable [1]."}]}}}

    text = BedrockGenerator(client, model_id="model-x").generate("Are buckets durable?", passages)

    assert text == "Durable [1]."
    kwargs = client.converse.call_args.kwargs
    assert kwargs["modelId"] == "model-x"
    prompt = kwargs["messages"][0]["content"][0]["text"]
    assert "[1] Buckets are durable." in prompt
    assert "[2] Beta pipelines" in prompt


def test_bedrock_generator_wraps_client_errors(passages: list[Passage]) -> None:
    client = MagicMock()
    client.converse.side_effect = _client_error("boom")

    with pytest.raises(GenerationError, match="boom"):
        BedrockGenerator(client, model_id="m").generate("q", passages)


def test_bedrock_generator_rejects_malformed_response(passages: list[Passage]) -> None:
    client = MagicMock()
    client.converse.return_value = {"output": {}}

    with pytest.raises(GenerationError, match="malformed"):
        BedrockGenerator(client, model_id="m").generate("q", passages)
