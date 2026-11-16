# code-it-ai-test: Bedrock Knowledge Base RAG pipeline

A small Python proof of concept that answers questions over local documents with Amazon Bedrock Knowledge Bases.
Everything runs offline by default: tests and the default CLI backend need no network and no AWS credentials.

## Layout

| Path | Purpose |
| --- | --- |
| `src/kbrag/ingestion.py` | Load `.md`/`.txt` files, chunk them, write a Bedrock data source folder |
| `src/kbrag/retrieval.py` | `Retriever` interface, offline `InMemoryRetriever`, real `BedrockRetriever` |
| `src/kbrag/generation.py` | `Generator` interface, offline `ExtractiveGenerator`, real `BedrockGenerator`, citation checks |
| `src/kbrag/evaluation.py` | RAGAS-style faithfulness and answer relevance, offline |
| `src/kbrag/cli.py` | `kbrag ingest`, `kbrag ask`, `kbrag eval` |
| `terraform/` | Plan-only stub for the knowledge base, data source, bucket and role |
| `data/` | Sample documents and an eval question set |

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.10+ is fetched automatically).

```bash
uv sync
make check   # ruff, ruff format --check, mypy --strict, pytest
```

## Run the CLI (offline fake backend)

```bash
uv run kbrag ingest --docs data/docs --out build/ds
uv run kbrag ask "How many vacation days per year?" --data build/ds
uv run kbrag eval --data build/ds --questions data/eval_questions.json --threshold 0.7
```

- `ingest` writes `<chunk_id>.txt`, a `<chunk_id>.txt.metadata.json` sidecar (Bedrock metadata format) and `manifest.json`.
- `ask` prints the answer with `[n]` markers and a Sources list. Every citation is checked against the manifest, so it
  must map to an ingested document. With no supporting passages it exits 2; with a missing or invalid citation it exits 1.
- `eval` prints per-question and mean `faithfulness` and `answer_relevance`, then exits 1 if either mean is below
  `--threshold`. Unanswerable questions score 0 and are counted, not skipped.

The eval metrics are lexical approximations of the RAGAS definitions so they run offline. For LLM-judged RAGAS scores
against the real stack, swap in the `ragas` library with a pinned Bedrock judge model.

## Swap the fake for real Bedrock

1. Plan the infrastructure (never applied from this repo; applies go through the approved `code-it-apply` wrapper):
   ```bash
   cd terraform && terraform init -backend=false && terraform plan -var account_id=<ACCOUNT_ID> -var vector_collection_arn=<AOSS_COLLECTION_ARN>
   ```
   The stub assumes an existing OpenSearch Serverless vector collection and index (dimension 1024 for Titan Text
   Embeddings V2). Chunking is `NONE` because `kbrag ingest` already chunks.
2. Sync the `build/ds` folder to the data source bucket and start an ingestion job. Confirm the indexed document count
   before querying, since an empty knowledge base returns no results rather than an error.
3. Configure credentials through the normal AWS mechanisms (profile or role, never files in this repo) and set:
   ```bash
   export KBRAG_KNOWLEDGE_BASE_ID=<KB_ID>
   export KBRAG_AWS_REGION=us-east-1
   export KBRAG_MODEL_ID=amazon.nova-lite-v1:0   # any Converse-capable model you have access to
   ```
4. Run with `--backend bedrock`:
   ```bash
   uv run kbrag ask "How many vacation days per year?" --data build/ds --backend bedrock
   ```
   `BedrockRetriever` calls the Retrieve API and `BedrockGenerator` calls Converse with numbered passages. The same
   citation checks apply.

Settings come from `KBRAG_*` environment variables or an untracked `.env` file. Do not commit secrets.

## Tests

Tests live in `tests/unit`, use fakes and `MagicMock` Bedrock clients, and clear AWS credential variables in a fixture.
