"""Tests for the real Bedrock adapter, driven entirely by botocore Stubber.

No network and no credentials: the stub validates the request shape we send and
returns canned responses, so these assert the adapter speaks the real
bedrock-agent-runtime API without calling AWS.
"""

from __future__ import annotations

from collections.abc import Iterator

import boto3
import pytest
from botocore.config import Config
from botocore.stub import Stubber

from rag.bedrock import PAGE_METADATA_KEY, BedrockKnowledgeBase
from rag.config import Settings
from rag.exceptions import GenerationError, IngestionError, RetrievalError
from rag.filters import EqualsCondition
from rag.models import Document

pytestmark = pytest.mark.unit


def _make_clients() -> tuple[object, object]:
    runtime = boto3.client(
        "bedrock-agent-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        config=Config(signature_version="v4"),
    )
    agent = boto3.client(
        "bedrock-agent",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        config=Config(signature_version="v4"),
    )
    return runtime, agent


@pytest.fixture
def stubbed() -> Iterator[tuple[BedrockKnowledgeBase, Stubber, Stubber, Settings]]:
    settings = Settings(
        knowledge_base_id="KB12345678",
        data_source_id="DS12345678",
        ingestion_poll_interval_seconds=0.01,
        ingestion_timeout_seconds=2,
    )
    runtime, agent = _make_clients()
    runtime_stub = Stubber(runtime)
    agent_stub = Stubber(agent)
    adapter = BedrockKnowledgeBase(runtime, agent, settings)
    with runtime_stub, agent_stub:
        yield adapter, runtime_stub, agent_stub, settings
        runtime_stub.assert_no_pending_responses()
        agent_stub.assert_no_pending_responses()


def _retrieve_result(text: str, source: str, score: float, page: int | None) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if page is not None:
        metadata[PAGE_METADATA_KEY] = page
    return {
        "content": {"text": text},
        "location": {"type": "S3", "s3Location": {"uri": source}},
        "score": score,
        "metadata": metadata,
    }


def test_retrieve_maps_passages_and_native_page_citation(
    stubbed: tuple[BedrockKnowledgeBase, Stubber, Stubber, Settings],
) -> None:
    adapter, stubber, _, _ = stubbed
    stubber.add_response(
        "retrieve",
        {
            "retrievalResults": [
                _retrieve_result("titan v2 1024 dims", "s3://corpus/embeddings.pdf", 0.91, 2),
            ]
        },
        {
            "knowledgeBaseId": "KB12345678",
            "retrievalQuery": {"text": "embedding model"},
            "retrievalConfiguration": {"vectorSearchConfiguration": {"numberOfResults": 5}},
        },
    )
    passages = adapter.retrieve("embedding model", top_k=5)
    assert len(passages) == 1
    assert passages[0].source == "s3://corpus/embeddings.pdf"
    assert passages[0].page == 2
    assert passages[0].score == pytest.approx(0.91)


def test_retrieve_single_condition_is_bare_not_oneelement_orall(
    stubbed: tuple[BedrockKnowledgeBase, Stubber, Stubber, Settings],
) -> None:
    adapter, stubber, _, _ = stubbed
    stubber.add_response(
        "retrieve",
        {"retrievalResults": []},
        {
            "knowledgeBaseId": "KB12345678",
            "retrievalQuery": {"text": "q"},
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults": 5,
                    # The bare equals condition, NOT {"orAll": [ ... ]} with one member.
                    "filter": {"equals": {"key": "tenant", "value": "acme"}},
                }
            },
        },
    )
    adapter.retrieve("q", top_k=5, conditions=[EqualsCondition(key="tenant", value="acme")])


def test_retrieve_two_conditions_use_orall(
    stubbed: tuple[BedrockKnowledgeBase, Stubber, Stubber, Settings],
) -> None:
    adapter, stubber, _, _ = stubbed
    stubber.add_response(
        "retrieve",
        {"retrievalResults": []},
        {
            "knowledgeBaseId": "KB12345678",
            "retrievalQuery": {"text": "q"},
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults": 5,
                    "filter": {
                        "orAll": [
                            {"equals": {"key": "tenant", "value": "acme"}},
                            {"equals": {"key": "tenant", "value": "globex"}},
                        ]
                    },
                }
            },
        },
    )
    adapter.retrieve(
        "q",
        top_k=5,
        conditions=[
            EqualsCondition(key="tenant", value="acme"),
            EqualsCondition(key="tenant", value="globex"),
        ],
    )


def test_retrieve_overfetches_when_reranking() -> None:
    settings = Settings(
        knowledge_base_id="KB12345678",
        data_source_id="DS12345678",
        rerank_model_arn="arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0",
    )
    runtime, agent = _make_clients()
    stubber = Stubber(runtime)
    adapter = BedrockKnowledgeBase(runtime, agent, settings)
    # top_k=5 -> over-fetch round(5 * 1.6) = 8, rerank down to 5.
    stubber.add_response(
        "retrieve",
        {"retrievalResults": []},
        {
            "knowledgeBaseId": "KB12345678",
            "retrievalQuery": {"text": "q"},
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults": 8,
                    "rerankingConfiguration": {
                        "type": "BEDROCK_RERANKING_MODEL",
                        "bedrockRerankingConfiguration": {
                            "modelConfiguration": {
                                "modelArn": settings.rerank_model_arn,
                            },
                            "numberOfRerankedResults": 5,
                        },
                    },
                },
            },
        },
    )
    with stubber:
        adapter.retrieve("q", top_k=5)
        stubber.assert_no_pending_responses()


def test_retrieve_empty_query_raises(
    stubbed: tuple[BedrockKnowledgeBase, Stubber, Stubber, Settings],
) -> None:
    adapter, _, _, _ = stubbed
    with pytest.raises(RetrievalError):
        adapter.retrieve("  ", top_k=5)


def test_retrieve_client_error_wrapped(
    stubbed: tuple[BedrockKnowledgeBase, Stubber, Stubber, Settings],
) -> None:
    adapter, stubber, _, _ = stubbed
    stubber.add_client_error("retrieve", service_error_code="ThrottlingException")
    with pytest.raises(RetrievalError):
        adapter.retrieve("q", top_k=5)


def _detail(doc_id: str, status: str) -> dict[str, object]:
    return {
        "knowledgeBaseId": "KB12345678",
        "dataSourceId": "DS12345678",
        "status": status,
        "identifier": {"dataSourceType": "CUSTOM", "custom": {"id": doc_id}},
    }


def test_ingest_polls_until_indexed(
    stubbed: tuple[BedrockKnowledgeBase, Stubber, Stubber, Settings],
) -> None:
    adapter, _, agent_stub, _ = stubbed
    agent_stub.add_response("ingest_knowledge_base_documents", {"documentDetails": []})
    # First poll: still in progress. Second poll: indexed.
    agent_stub.add_response(
        "list_knowledge_base_documents",
        {"documentDetails": [_detail("kb-001", "IN_PROGRESS")]},
    )
    agent_stub.add_response(
        "list_knowledge_base_documents",
        {"documentDetails": [_detail("kb-001", "INDEXED")]},
    )
    indexed = adapter.ingest([Document(doc_id="kb-001", text="hello", source="inline")])
    assert indexed == 1


def test_ingest_zero_indexed_is_an_error(
    stubbed: tuple[BedrockKnowledgeBase, Stubber, Stubber, Settings],
) -> None:
    adapter, _, agent_stub, _ = stubbed
    agent_stub.add_response("ingest_knowledge_base_documents", {"documentDetails": []})
    agent_stub.add_response(
        "list_knowledge_base_documents",
        {"documentDetails": [_detail("kb-001", "FAILED")]},
    )
    with pytest.raises(IngestionError):
        adapter.ingest([Document(doc_id="kb-001", text="hello", source="inline")])


def test_ingest_empty_document_list_raises(
    stubbed: tuple[BedrockKnowledgeBase, Stubber, Stubber, Settings],
) -> None:
    adapter, _, _, _ = stubbed
    with pytest.raises(IngestionError):
        adapter.ingest([])


def test_generate_returns_answer_with_citations(
    stubbed: tuple[BedrockKnowledgeBase, Stubber, Stubber, Settings],
) -> None:
    adapter, stubber, _, _ = stubbed
    stubber.add_response(
        "retrieve_and_generate",
        {
            "sessionId": "sess-1",
            "output": {"text": "Titan v2 produces 1024 dimensional vectors."},
            "citations": [
                {
                    "retrievedReferences": [
                        {
                            "location": {
                                "type": "S3",
                                "s3Location": {"uri": "s3://corpus/embeddings.pdf"},
                            },
                            "metadata": {PAGE_METADATA_KEY: 2},
                        }
                    ]
                }
            ],
        },
        {
            "input": {"text": "embedding dims"},
            "retrieveAndGenerateConfiguration": {
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": "KB12345678",
                    "modelArn": Settings().generation_model,
                },
            },
        },
    )
    answer = adapter.generate("embedding dims", passages=())
    assert "1024" in answer.text
    assert answer.citations[0].page == 2
    assert answer.citations[0].source == "s3://corpus/embeddings.pdf"


def test_generate_attaches_guardrail_to_output() -> None:
    settings = Settings(
        knowledge_base_id="KB12345678",
        data_source_id="DS12345678",
        guardrail_id="gr-1",
        guardrail_version="DRAFT",
    )
    runtime, agent = _make_clients()
    stubber = Stubber(runtime)
    adapter = BedrockKnowledgeBase(runtime, agent, settings)
    stubber.add_response(
        "retrieve_and_generate",
        {"sessionId": "sess-1", "output": {"text": "answer"}, "citations": []},
        {
            "input": {"text": "q"},
            "retrieveAndGenerateConfiguration": {
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": "KB12345678",
                    "modelArn": settings.generation_model,
                    "generationConfiguration": {
                        "guardrailConfiguration": {
                            "guardrailId": "gr-1",
                            "guardrailVersion": "DRAFT",
                        }
                    },
                },
            },
        },
    )
    with stubber:
        answer = adapter.generate("q", passages=())
        stubber.assert_no_pending_responses()
    assert answer.text == "answer"


def test_generate_empty_answer_raises(
    stubbed: tuple[BedrockKnowledgeBase, Stubber, Stubber, Settings],
) -> None:
    adapter, stubber, _, _ = stubbed
    stubber.add_response(
        "retrieve_and_generate",
        {"sessionId": "sess-1", "output": {"text": ""}, "citations": []},
        {
            "input": {"text": "q"},
            "retrieveAndGenerateConfiguration": {
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": "KB12345678",
                    "modelArn": Settings().generation_model,
                },
            },
        },
    )
    with pytest.raises(GenerationError):
        adapter.generate("q", passages=())
