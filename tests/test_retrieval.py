from pathlib import Path
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from ragkb.errors import RetrievalError
from ragkb.ingestion import ingest
from ragkb.retrieval import BedrockRetriever, LocalRetriever, load_chunks


def _client_error(message: str) -> ClientError:
    return ClientError({"Error": {"Code": "ThrottlingException", "Message": message}}, "Operation")


def test_local_retriever_ranks_matching_chunk_first(docs_dir: Path, tmp_path: Path) -> None:
    ingest(docs_dir, tmp_path / "kb", chunk_words=50, overlap_words=5)
    retriever = LocalRetriever(load_chunks(tmp_path / "kb"))

    results = retriever.retrieve("How do buckets store data?", top_k=2)

    assert results[0].source == "alpha.md"
    assert results[0].score > 0
    assert results == sorted(results, key=lambda p: p.score, reverse=True)


def test_local_retriever_drops_zero_score_results(docs_dir: Path, tmp_path: Path) -> None:
    ingest(docs_dir, tmp_path / "kb", chunk_words=50, overlap_words=5)
    retriever = LocalRetriever(load_chunks(tmp_path / "kb"))

    assert retriever.retrieve("zebra xylophone", top_k=3) == []


def test_load_chunks_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(RetrievalError, match="manifest"):
        load_chunks(tmp_path)


def _bedrock_response() -> dict[str, object]:
    return {
        "retrievalResults": [
            {
                "content": {"text": "Buckets are durable."},
                "score": 0.82,
                "location": {"type": "S3", "s3Location": {"uri": "s3://bucket/data/alpha.md-0.txt"}},
                "metadata": {"source": "alpha.md", "chunk_id": "alpha.md#0"},
            },
            {
                "content": {"text": "No metadata here."},
                "score": 0.3,
                "location": {"type": "S3", "s3Location": {"uri": "s3://bucket/data/raw.txt"}},
                "metadata": {},
            },
        ]
    }


def test_bedrock_retriever_maps_results_to_passages() -> None:
    client = MagicMock()
    client.retrieve.return_value = _bedrock_response()

    results = BedrockRetriever(client, knowledge_base_id="KB123").retrieve("durable?", top_k=5)

    client.retrieve.assert_called_once_with(
        knowledgeBaseId="KB123",
        retrievalQuery={"text": "durable?"},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 5}},
    )
    assert (results[0].source, results[0].chunk_id, results[0].score) == ("alpha.md", "alpha.md#0", 0.82)
    assert results[1].source == "s3://bucket/data/raw.txt"


def test_bedrock_retriever_wraps_client_errors() -> None:
    client = MagicMock()
    client.retrieve.side_effect = _client_error("throttled")

    with pytest.raises(RetrievalError, match="throttled"):
        BedrockRetriever(client, knowledge_base_id="KB123").retrieve("q", top_k=1)
