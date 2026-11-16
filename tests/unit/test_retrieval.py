from typing import Any
from unittest.mock import MagicMock

import pytest

from kbrag.errors import RetrievalError
from kbrag.models import Chunk
from kbrag.retrieval import BedrockRetriever, InMemoryRetriever


def test_in_memory_retriever_ranks_relevant_chunk_first(chunks: list[Chunk]) -> None:
    result = InMemoryRetriever(chunks).retrieve("How many vacation days do employees get?", top_k=2)
    assert result[0].source == "hr/vacation.txt"
    assert result[0].score >= result[-1].score
    assert all(0.0 <= p.score <= 1.0 for p in result)


def test_in_memory_retriever_drops_zero_score_passages(chunks: list[Chunk]) -> None:
    assert InMemoryRetriever(chunks).retrieve("zzzz qqqq", top_k=3) == []


def test_bedrock_retriever_maps_response_to_passages() -> None:
    client = MagicMock()
    client.retrieve.return_value = {
        "retrievalResults": [
            {
                "content": {"text": "Twenty days of vacation."},
                "score": 0.82,
                "location": {"s3Location": {"uri": "s3://bucket/vacation-0000.txt"}},
                "metadata": {"source": "hr/vacation.txt"},
            },
            {
                "content": {"text": "No metadata chunk."},
                "score": 0.4,
                "location": {"s3Location": {"uri": "s3://bucket/other-0001.txt"}},
            },
        ]
    }
    result = BedrockRetriever(client, knowledge_base_id="KB123").retrieve("vacation?", top_k=5)

    kwargs: dict[str, Any] = client.retrieve.call_args.kwargs
    assert kwargs["knowledgeBaseId"] == "KB123"
    assert kwargs["retrievalConfiguration"]["vectorSearchConfiguration"]["numberOfResults"] == 5
    assert [p.source for p in result] == ["hr/vacation.txt", "s3://bucket/other-0001.txt"]
    assert result[0].score == pytest.approx(0.82)


def test_bedrock_retriever_wraps_client_errors() -> None:
    client = MagicMock()
    client.retrieve.side_effect = RuntimeError("throttled")
    with pytest.raises(RetrievalError):
        BedrockRetriever(client, knowledge_base_id="KB123").retrieve("q", top_k=3)
