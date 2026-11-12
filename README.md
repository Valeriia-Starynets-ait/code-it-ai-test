# code-it-ai-test

Throwaway repo for testing Code-it AI POC agents.

## Bedrock Knowledge Base RAG pipeline

A Bedrock Knowledge Base RAG pipeline with two interchangeable backends behind
one set of ports:

- the real `BedrockKnowledgeBase` adapter (`src/rag/bedrock.py`) that calls the
  `bedrock-agent-runtime` and `bedrock-agent` APIs; and
- an in-memory `FakeKnowledgeBaseClient` (`src/rag/retrieval.py`) used for the
  CLI demo and the whole test suite, so everything runs with no AWS credentials
  and no network.

The pipeline, CLI and eval harness depend only on the ports, so swapping fake
for real is a one-line backend choice (`rag --backend bedrock ...`).

### Deliberate choices: vector store and chunking

Recorded together per the bedrock-kb-rag skill because they are not
interchangeable and cannot be changed after the data source exists:

- Vector store: OpenSearch Serverless (`VECTORSEARCH`). Reason: it is the
  default in our reusable KB construct, surfaces native page citations (which
  S3 Vectors does not), and needs no cluster to size for a POC. The Terraform
  stub leaves it as a placeholder.
- Embedding model: Titan Text Embeddings v2 at 1024 dimensions. Reason: it is
  the default KB embedding model and the dimension must match the model or KB
  creation fails. Embedding choice is also a cost decision (Cohere Embed 3 has
  run ~5x pricier), so the cheaper default is deliberate.
- Chunking: HIERARCHICAL (parent/child). Reason: hierarchical gives the answer
  model larger parent context while retrieval still matches on small child
  chunks, and it avoids the ~1 MB per-document limit that makes SEMANTIC
  chunking fail on larger PDFs. FIXED_SIZE (~300 tokens, 20% overlap) stays
  available in config for clean, uniform corpora. The strategy is fixed before
  the data source is created and recreating it means re-ingesting.

### What is here

- `src/rag/config.py` - `BaseSettings` config recording the choices above plus
  the KB id, data source id, generation model, optional guardrail and rerank
  ARN. Configuration only, no live calls.
- `src/rag/models.py` - frozen Pydantic v2 domain models.
- `src/rag/exceptions.py` - the `RagError` hierarchy.
- `src/rag/ports.py` - `Retriever`, `Ingestor`, `Generator` protocols.
- `src/rag/filters.py` - metadata-filter builder that applies the single-value
  rule (a bare `equals`, never a one-element `orAll`).
- `src/rag/bedrock.py` - the real `BedrockKnowledgeBase` adapter (see mapping
  below).
- `src/rag/ingestion.py` - document loading (JSONL + a built-in sample corpus).
- `src/rag/retrieval.py` - `FakeKnowledgeBaseClient` for tests and the demo.
- `src/rag/generation.py` - `TemplateGenerator`, the fake cited generator.
- `src/rag/pipeline.py` - `RagPipeline`, the retrieve-then-generate orchestrator.
- `src/rag/eval.py` - a RAGAS-style eval harness. Retrieval and answer metrics
  reported separately, +/-1 page tolerance, non-computable rows imputed to 1.0
  and counted.
- `src/rag/cli.py` - the `rag` CLI (`ingest`, `ask`, `eval`) with `--backend`.
- `terraform/` - a stub (S3 corpus bucket, scoped KB service role, KB and
  OpenSearch Serverless placeholders) tagged `code-it:ticket`. Not applied here.

### Which skill section each part of the real adapter comes from

- `retrieve` with `numberOfResults` and native page citation from
  `x-amz-bedrock-kb-document-page-number` - skill "Retrieve skeleton" and the
  field note on native page citations.
- single-condition metadata filter as a bare `equals` (no one-element `orAll`) -
  skill field note on `orAll` needing 2+ members (`src/rag/filters.py`).
- optional rerank by over-fetching ~1.6x then reranking down - skill step 6
  "Rerank only if it pays".
- ingestion through `IngestKnowledgeBaseDocuments` with inline metadata and
  status polling where 0 indexed is an error - skill step 4 "Ingest and wait"
  and the pitfall that a KB with 0 indexed docs returns empty, not an error.
- generation through `retrieve_and_generate` with an optional guardrail on the
  output - skill step 7 "Generate with a guardrail" and the guideline to apply
  guardrails to RAG outputs because retrieved chunks are untrusted.

### Usage

```bash
uv run --extra dev rag ingest
uv run --extra dev rag ask "which embedding model is default and how many dimensions"
uv run --extra dev rag eval --gate
# live Knowledge Base (needs AWS credentials and a provisioned KB):
uv run --extra dev rag --backend bedrock ask "..."
```

### Quality gate

`make check` runs ruff lint, ruff format check, mypy and pytest (unit + e2e):

```bash
make check
```

### Scope

No deploy. The test suite uses no network and no AWS credentials: the real
adapter is driven entirely by `botocore.stub.Stubber`, which validates the exact
request and response shapes against the live API model without calling AWS. The
Terraform is a stub and is not applied (applies go through `code-it-apply`, never
a raw `terraform apply`).
