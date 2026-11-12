# code-it-ai-test

Throwaway repo for testing Code-it AI POC agents.

## Bedrock Knowledge Base RAG pipeline

A self-contained Bedrock Knowledge Base RAG pipeline. It runs end to end with
no AWS credentials and no network: retrieval is served by an in-memory fake
client that matches the Bedrock `retrieve` shape, so a real Bedrock adapter can
replace it without touching the pipeline, CLI or eval harness.

### What is here

- `src/rag/config.py` - `BaseSettings` config. Records the deliberate KB choices:
  Titan Text Embeddings v2, 1024 dimensions, `FIXED_SIZE` chunking (~300 tokens,
  20% overlap). These are configuration, not live calls.
- `src/rag/models.py` - frozen Pydantic v2 domain models (`Document`,
  `RetrievedPassage`, `Citation`, `GeneratedAnswer`, `Answer`).
- `src/rag/exceptions.py` - the `RagError` hierarchy.
- `src/rag/ports.py` - `Retriever`, `Ingestor`, `Generator` protocols (the ports).
- `src/rag/ingestion.py` - document loading (JSONL + a built-in sample corpus).
- `src/rag/retrieval.py` - `FakeKnowledgeBaseClient`, a credential-free ingest +
  retrieve adapter with deterministic Jaccard scoring.
- `src/rag/generation.py` - `TemplateGenerator`, cited answer generation with
  inline `[n]` markers. Citations are the trust signal.
- `src/rag/pipeline.py` - `RagPipeline`, the retrieve-then-generate orchestrator
  with injected dependencies.
- `src/rag/eval.py` - a RAGAS-style eval harness. Retrieval and answer metrics
  are reported separately, page matching tolerates +/-1, and non-computable
  rows are imputed to 1.0 and counted.
- `src/rag/cli.py` - the `rag` CLI (`ingest`, `ask`, `eval`).
- `terraform/` - a stub for the intended infrastructure (S3 corpus bucket,
  scoped KB service role, placeholders for the KB and OpenSearch Serverless
  vector store), tagged `code-it:ticket`. Not applied here.

### Usage

```bash
uv run --extra dev rag ingest
uv run --extra dev rag ask "which embedding model is default and how many dimensions"
uv run --extra dev rag eval --gate
```

### Quality gate

`make check` runs ruff lint, ruff format check, mypy and pytest (unit + e2e):

```bash
make check
```

### Scope

No deploy. Tests use no network and no AWS credentials. The real Bedrock
Knowledge Base, embedding and generation calls are intentionally out of scope;
the ports make them a drop-in replacement for the fake client.
