"""Real Amazon Bedrock Knowledge Base adapter.

Implements the :class:`rag.ports.Retriever`, :class:`rag.ports.Ingestor` and
:class:`rag.ports.Generator` ports against the ``bedrock-agent-runtime`` API, so
it drops straight into :class:`rag.pipeline.RagPipeline` in place of the fake
client. Every method maps to a section of the bedrock-kb-rag skill; see the
module and method docstrings for which one.

Nothing here is exercised by a live call in the test suite - the tests drive it
with ``botocore.stub.Stubber`` so there is no network and no credentials.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any, cast

from rag.config import RERANK_OVERFETCH_FACTOR, Settings
from rag.exceptions import GenerationError, IngestionError, RetrievalError
from rag.filters import EqualsCondition, build_metadata_filter
from rag.models import Citation, Document, GeneratedAnswer, RetrievedPassage

# The ``retrieve`` / ``retrieve_and_generate`` calls live on the
# ``bedrock-agent-runtime`` client; ``ingest_knowledge_base_documents`` and
# ``list_knowledge_base_documents`` live on the ``bedrock-agent`` control-plane
# client. The adapter takes both. They are typed as ``Any`` because the request
# bodies are built dynamically (optional filter / rerank / guardrail blocks) and
# the generated TypedDicts are invariant; the request shapes are pinned by the
# Stubber tests instead.
RuntimeClient = Any
AgentClient = Any

logger = logging.getLogger(__name__)

# Native page-number metadata key on PDF-backed knowledge bases (skill field note).
PAGE_METADATA_KEY = "x-amz-bedrock-kb-document-page-number"

# Per-document ingestion statuses (from the KnowledgeBaseDocumentDetail enum).
_INGEST_SUCCESS = "INDEXED"
_INGEST_FAILED = frozenset({"FAILED", "METADATA_UPDATE_FAILED", "NOT_FOUND"})
# Statuses that mean "still working" - keep polling.
_INGEST_IN_FLIGHT = frozenset({"PENDING", "STARTING", "IN_PROGRESS", "DELETING", "DELETE_IN_PROGRESS"})


class BedrockKnowledgeBase:
    """Adapter over a Bedrock Knowledge Base via ``bedrock-agent-runtime``."""

    def __init__(
        self,
        runtime_client: RuntimeClient,
        agent_client: AgentClient,
        settings: Settings,
    ) -> None:
        self._runtime = runtime_client
        self._agent = agent_client
        self._settings = settings

    # ------------------------------------------------------------------ retrieve
    def retrieve(
        self,
        query: str,
        top_k: int,
        conditions: Sequence[EqualsCondition] | None = None,
    ) -> Sequence[RetrievedPassage]:
        """Retrieve ranked, cited passages (skill: "Retrieve skeleton", step 5-6).

        When a rerank model is configured, over-fetch ~1.6x topK then let Bedrock
        rerank down to ``top_k`` (skill step 6). Native page numbers come from the
        ``x-amz-bedrock-kb-document-page-number`` metadata key (skill field note).
        """
        if not query.strip():
            raise RetrievalError("query must not be empty")
        if top_k < 1:
            raise RetrievalError(f"top_k must be >= 1, got {top_k}")

        fetch_k = top_k
        vector_cfg: dict[str, object] = {"numberOfResults": fetch_k}

        metadata_filter = build_metadata_filter(conditions or ())
        if metadata_filter is not None:
            vector_cfg["filter"] = metadata_filter

        retrieval_cfg: dict[str, object] = {"vectorSearchConfiguration": vector_cfg}

        if self._settings.rerank_model_arn:
            fetch_k = max(top_k, round(top_k * RERANK_OVERFETCH_FACTOR))
            vector_cfg["numberOfResults"] = fetch_k
            # Reranking nests inside the vector-search config, not the top level.
            vector_cfg["rerankingConfiguration"] = {
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": {"modelArn": self._settings.rerank_model_arn},
                    "numberOfRerankedResults": top_k,
                },
            }

        try:
            response = self._runtime.retrieve(
                knowledgeBaseId=self._settings.knowledge_base_id,
                retrievalQuery={"text": query},
                retrievalConfiguration=retrieval_cfg,
            )
        except Exception as exc:  # botocore ClientError and friends
            raise RetrievalError(f"retrieve call failed: {exc}") from exc

        results = cast("list[dict[str, Any]]", response.get("retrievalResults", []))
        passages = tuple(self._to_passage(r) for r in results[:top_k])
        logger.info("retrieved %d passage(s) (fetch_k=%d)", len(passages), fetch_k)
        return passages

    @staticmethod
    def _to_passage(result: dict[str, Any]) -> RetrievedPassage:
        content = result.get("content", {})
        text = content.get("text", "")
        location = result.get("location", {})
        source = _extract_source(location)
        metadata = result.get("metadata", {})
        page = _page_of(metadata)
        score = float(result.get("score", 0.0))
        # Bedrock scores are not bounded to [0, 1]; clamp so the model stays valid.
        score = min(max(score, 0.0), 1.0)
        return RetrievedPassage(
            doc_id=source,
            text=text or "(empty)",
            source=source,
            score=score,
            page=page,
        )

    # ------------------------------------------------------------------- ingest
    def ingest(self, documents: Sequence[Document]) -> int:
        """Ingest via ``IngestKnowledgeBaseDocuments`` with inline metadata (skill step 4).

        Polls per-document status until every document is terminal. Zero indexed
        documents is raised as an error, not returned as an empty success - a KB
        with 0 indexed docs silently returns empty on retrieve (skill step 4 /
        pitfalls), so we fail loudly here instead.
        """
        if not documents:
            raise IngestionError("<none>", "no documents to ingest")

        payload = [self._document_payload(doc) for doc in documents]
        try:
            self._agent.ingest_knowledge_base_documents(
                knowledgeBaseId=self._settings.knowledge_base_id,
                dataSourceId=self._settings.data_source_id,
                documents=payload,
            )
        except Exception as exc:
            raise IngestionError(documents[0].doc_id, f"ingest call failed: {exc}") from exc

        indexed = self._poll_until_indexed({doc.doc_id for doc in documents})
        if indexed == 0:
            raise IngestionError("<all>", "0 documents indexed after ingestion")
        logger.info("ingestion complete: %d document(s) indexed", indexed)
        return indexed

    def _document_payload(self, doc: Document) -> dict[str, Any]:
        metadata_attrs: list[dict[str, Any]] = [
            {
                "key": "doc_id",
                "value": {"type": "STRING", "stringValue": doc.doc_id},
            }
        ]
        if doc.page is not None:
            metadata_attrs.append(
                {
                    "key": "page",
                    "value": {"type": "NUMBER", "numberValue": doc.page},
                }
            )
        return {
            "content": {
                "dataSourceType": "CUSTOM",
                "custom": {
                    "customDocumentIdentifier": {"id": doc.doc_id},
                    "sourceType": "IN_LINE",
                    "inlineContent": {
                        "type": "TEXT",
                        "textContent": {"data": doc.text},
                    },
                },
            },
            "metadata": {
                "type": "IN_LINE_ATTRIBUTE",
                "inlineAttributes": metadata_attrs,
            },
        }

    def _poll_until_indexed(self, expected_ids: set[str]) -> int:
        deadline = time.monotonic() + self._settings.ingestion_timeout_seconds
        while True:
            statuses = self._list_document_statuses()
            terminal = {
                doc_id: status
                for doc_id, status in statuses.items()
                if doc_id in expected_ids and status not in _INGEST_IN_FLIGHT
            }
            failed = {d: s for d, s in terminal.items() if s in _INGEST_FAILED}
            if failed:
                first = next(iter(failed))
                raise IngestionError(first, f"ingestion failed with status {failed[first]}")
            if len(terminal) >= len(expected_ids):
                return sum(1 for s in terminal.values() if s == _INGEST_SUCCESS)
            if time.monotonic() >= deadline:
                raise IngestionError(
                    "<timeout>",
                    f"ingestion did not finish within {self._settings.ingestion_timeout_seconds}s",
                )
            time.sleep(self._settings.ingestion_poll_interval_seconds)

    def _list_document_statuses(self) -> dict[str, str]:
        statuses: dict[str, str] = {}
        next_token: str | None = None
        while True:
            kwargs: dict[str, Any] = {
                "knowledgeBaseId": self._settings.knowledge_base_id,
                "dataSourceId": self._settings.data_source_id,
            }
            if next_token:
                kwargs["nextToken"] = next_token
            try:
                response = self._agent.list_knowledge_base_documents(**kwargs)
            except Exception as exc:
                raise IngestionError("<status>", f"status poll failed: {exc}") from exc
            for item in response.get("documentDetails", []):
                identifier = item.get("identifier", {})
                custom = identifier.get("custom", {})
                doc_id = custom.get("id", "")
                if doc_id:
                    statuses[doc_id] = item.get("status", "IN_PROGRESS")
            next_token = response.get("nextToken")
            if not next_token:
                return statuses

    # ----------------------------------------------------------------- generate
    def generate(
        self,
        query: str,
        passages: Sequence[RetrievedPassage],
    ) -> GeneratedAnswer:
        """Answer via ``retrieve_and_generate`` with an optional output guardrail.

        Maps to skill step 7: generate with a guardrail, and surface citations as
        the trust signal. The guardrail is attached to the *generation* step
        because a retrieved chunk is untrusted and can carry injected instructions
        (skill field note). ``passages`` is unused here because
        ``retrieve_and_generate`` retrieves internally; the pipeline still passes
        it for parity with the fake generator and the port signature.
        """
        del passages  # retrieve_and_generate retrieves internally.
        if not query.strip():
            raise GenerationError("query must not be empty")

        kb_config: dict[str, Any] = {
            "knowledgeBaseId": self._settings.knowledge_base_id,
            "modelArn": self._settings.generation_model,
        }
        gen_config: dict[str, Any] = {}
        if self._settings.guardrail_id and self._settings.guardrail_version:
            gen_config["guardrailConfiguration"] = {
                "guardrailId": self._settings.guardrail_id,
                "guardrailVersion": self._settings.guardrail_version,
            }
        if gen_config:
            kb_config["generationConfiguration"] = gen_config

        try:
            response = self._runtime.retrieve_and_generate(
                input={"text": query},
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": kb_config,
                },
            )
        except Exception as exc:
            raise GenerationError(f"retrieve_and_generate call failed: {exc}") from exc

        text = response.get("output", {}).get("text", "")
        if not text:
            raise GenerationError("retrieve_and_generate returned an empty answer")
        citations = _citations_from_response(response.get("citations", []))
        logger.info("generated answer with %d citation(s)", len(citations))
        return GeneratedAnswer(text=text, citations=tuple(citations))


def _page_of(metadata: dict[str, Any]) -> int | None:
    """Recover the native page number from chunk metadata (skill field note)."""
    raw = metadata.get(PAGE_METADATA_KEY)
    if raw is None:
        return None
    try:
        page = int(raw)
    except (TypeError, ValueError):
        return None
    return page if page >= 1 else None


def _extract_source(location: dict[str, Any]) -> str:
    for key in ("s3Location", "customDocumentLocation", "webLocation"):
        loc = location.get(key)
        if isinstance(loc, dict):
            for field in ("uri", "id", "url"):
                value = loc.get(field)
                if value:
                    return str(value)
    return "unknown"


def _citations_from_response(raw_citations: Sequence[dict[str, Any]]) -> list[Citation]:
    citations: list[Citation] = []
    marker = 0
    for citation in raw_citations:
        for ref in citation.get("retrievedReferences", []):
            marker += 1
            location = ref.get("location", {})
            metadata = ref.get("metadata", {})
            source = _extract_source(location)
            citations.append(
                Citation(
                    marker=marker,
                    doc_id=source,
                    source=source,
                    page=_page_of(metadata),
                )
            )
    return citations
