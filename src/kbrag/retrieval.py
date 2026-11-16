import logging
from typing import TYPE_CHECKING, Protocol

from botocore.exceptions import BotoCoreError, ClientError

from kbrag.errors import RetrievalError
from kbrag.models import Chunk, Passage
from kbrag.text import content_tokens

if TYPE_CHECKING:
    from mypy_boto3_bedrock_agent_runtime import AgentsforBedrockRuntimeClient
    from mypy_boto3_bedrock_agent_runtime.type_defs import KnowledgeBaseRetrievalResultTypeDef

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[Passage]: ...


class InMemoryRetriever:
    """Offline lexical retriever over prepared chunks; stands in for Bedrock in tests and local runs."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str, top_k: int) -> list[Passage]:
        query_tokens = content_tokens(query)
        if not query_tokens:
            return []
        scored = [
            Passage(
                text=c.text,
                score=len(query_tokens & content_tokens(c.text)) / len(query_tokens),
                source=c.source,
                chunk_id=c.chunk_id,
            )
            for c in self._chunks
        ]
        ranked = sorted((p for p in scored if p.score > 0), key=lambda p: p.score, reverse=True)
        return ranked[:top_k]


class BedrockRetriever:
    """Queries a Bedrock Knowledge Base through the bedrock-agent-runtime Retrieve API."""

    def __init__(self, client: "AgentsforBedrockRuntimeClient", knowledge_base_id: str) -> None:
        self._client = client
        self._kb_id = knowledge_base_id

    def retrieve(self, query: str, top_k: int) -> list[Passage]:
        try:
            response = self._client.retrieve(
                knowledgeBaseId=self._kb_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": top_k}},
            )
        except (BotoCoreError, ClientError, RuntimeError) as exc:
            raise RetrievalError(f"Knowledge base {self._kb_id} query failed: {exc}") from exc
        passages = [self._to_passage(r) for r in response.get("retrievalResults", [])]
        logger.info("Retrieved %d passages", len(passages))
        return passages

    @staticmethod
    def _to_passage(result: "KnowledgeBaseRetrievalResultTypeDef") -> Passage:
        uri = result.get("location", {}).get("s3Location", {}).get("uri", "unknown")
        metadata = result.get("metadata", {})
        return Passage(
            text=result["content"]["text"],
            score=float(result.get("score", 0.0)),
            source=str(metadata.get("source", uri)),
            chunk_id=uri.rsplit("/", 1)[-1].removesuffix(".txt"),
        )
