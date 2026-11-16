import json
from pathlib import Path
from typing import Any, Protocol

from botocore.exceptions import BotoCoreError, ClientError

from ragkb.errors import RetrievalError
from ragkb.ingestion import MANIFEST_NAME
from ragkb.models import Chunk, Passage
from ragkb.text import content_tokens


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[Passage]: ...


class AgentRuntimeClient(Protocol):
    def retrieve(self, **kwargs: Any) -> dict[str, Any]: ...


def load_chunks(index_dir: Path) -> list[Chunk]:
    manifest = index_dir / MANIFEST_NAME
    if not manifest.is_file():
        raise RetrievalError(f"manifest not found: {manifest} (run `ragkb ingest` first)")
    try:
        return [Chunk.model_validate(json.loads(line)) for line in manifest.read_text(encoding="utf-8").splitlines()]
    except ValueError as exc:
        raise RetrievalError(f"manifest is corrupt: {manifest}: {exc}") from exc


class LocalRetriever:
    """Offline stand-in for the knowledge base: scores chunks by the share of query terms they contain."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str, top_k: int) -> list[Passage]:
        terms = content_tokens(query)
        if not terms:
            return []
        scored = [
            Passage(
                chunk_id=c.chunk_id,
                source=c.source,
                text=c.text,
                score=len(terms & content_tokens(c.text)) / len(terms),
            )
            for c in self._chunks
        ]
        ranked = sorted((p for p in scored if p.score > 0), key=lambda p: p.score, reverse=True)
        return ranked[:top_k]


class BedrockRetriever:
    """Queries a real Bedrock Knowledge Base through the bedrock-agent-runtime Retrieve API."""

    def __init__(self, client: AgentRuntimeClient, knowledge_base_id: str) -> None:
        self._client = client
        self._knowledge_base_id = knowledge_base_id

    def retrieve(self, query: str, top_k: int) -> list[Passage]:
        try:
            response = self._client.retrieve(
                knowledgeBaseId=self._knowledge_base_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": top_k}},
            )
        except (BotoCoreError, ClientError) as exc:
            raise RetrievalError(f"Bedrock retrieve failed: {exc}") from exc
        return [_to_passage(item) for item in response.get("retrievalResults", [])]


def _to_passage(item: dict[str, Any]) -> Passage:
    metadata = item.get("metadata") or {}
    uri = str(((item.get("location") or {}).get("s3Location") or {}).get("uri", "unknown"))
    return Passage(
        chunk_id=str(metadata.get("chunk_id", uri)),
        source=str(metadata.get("source", uri)),
        text=str(item["content"]["text"]),
        score=float(item.get("score", 0.0)),
    )
