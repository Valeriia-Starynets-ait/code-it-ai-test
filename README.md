# code-it-ai-test

Proof of concept: a small Python service that answers questions over local documents with Amazon Bedrock
Knowledge Bases. Tests run offline, with no network and no AWS credentials. Nothing is deployed to AWS.

## Layout

| Path | Purpose |
| --- | --- |
| `src/ragkb/ingestion.py` | Load `.md`/`.txt` files, chunk them, stage a Bedrock data source |
| `src/ragkb/retrieval.py` | `Retriever` interface, `LocalRetriever` (offline) and `BedrockRetriever` |
| `src/ragkb/generation.py` | `Generator` interface, `ExtractiveGenerator` (offline), `BedrockGenerator`, cited `answer_question` |
| `src/ragkb/evaluation.py` | RAGAS-style faithfulness and answer relevance, runnable offline |
| `src/ragkb/cli.py` | `ragkb ingest`, `ragkb ask`, `ragkb eval` |
| `infra/terraform/` | Knowledge base stub (plan-only) |
| `data/docs`, `data/eval` | Sample documents and sample question set |

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.10 or newer (`.python-version` pins 3.12).

```bash
uv sync
make check        # ruff check, ruff format --check, mypy, pytest
```

## Run the CLI

```bash
uv run ragkb ingest --docs data/docs --out build/kb
uv run ragkb ask "What does faithfulness measure?" --index build/kb
uv run ragkb eval --questions data/eval/questions.json --index build/kb --threshold 0.7
```

- `ingest` chunks each document (default 120 words, 20 overlap) and writes `build/kb/data/<chunk>.txt` plus a
  `<chunk>.txt.metadata.json` sidecar (Bedrock metadata format), and `build/kb/manifest.jsonl`.
- `ask` prints the answer followed by a `Sources:` list. Every answer must cite at least one retrieved passage, and
  each citation must map to a retrieved chunk and a known source document. Otherwise the call fails with an error.
  With no relevant passages it exits 1.
- `eval` prints per-question and mean `faithfulness` and `answer_relevance`, then `RESULT: PASS` or `RESULT: FAIL`.
  It exits 1 when either mean is below `--threshold`.

Exit codes: 0 success, 1 no evidence or eval below threshold, 2 configuration or input error.

### Evaluation metrics

These are lightweight, deterministic approximations of the RAGAS metrics so the harness runs offline:

- faithfulness: share of answer sentences whose content terms are covered by the retrieved passages
- answer relevance: share of the question's content terms that the answer addresses

For judge-based RAGAS scoring against a real model, replace the functions in `evaluation.py`. The harness shape
(cases in, per-case scores out, threshold gate) stays the same.

## Swap the fake for real Bedrock

The default `local` backend uses `LocalRetriever` and `ExtractiveGenerator`. The `bedrock` backend uses
`BedrockRetriever` (Retrieve API) and `BedrockGenerator` (Converse API). Both sit behind the same `Retriever` and
`Generator` interfaces, so nothing else changes.

1. Provision the knowledge base. `infra/terraform` is a stub, plan-only:
   `cd infra/terraform && terraform init -backend=false && terraform validate`. Applying is done by the owner,
   through the approved apply wrapper, not from this repository.
2. Upload `build/kb/data/` to the documents bucket under the `data/` prefix, then start an ingestion job on the
   data source. The data source uses the `NONE` chunking strategy because `ragkb ingest` already chunked.
3. Configure through environment variables or an untracked `.env` file (never commit it):

   ```bash
   export RAG_BACKEND=bedrock
   export RAG_KNOWLEDGE_BASE_ID=<knowledge_base_id from terraform output>
   export RAG_AWS_REGION=us-east-1
   export RAG_MODEL_ID=amazon.nova-lite-v1:0
   ```

   Credentials come from the standard AWS chain (profile or role). They are never stored in the repo.
4. Run `uv run ragkb ask "..."` and `uv run ragkb eval`. You can also pass `--backend bedrock` per command.

Chunk metadata (`source`, `chunk_id`) is read from the sidecar attributes, so citations map back to the original
document names. Tests use `MagicMock` clients for both Bedrock APIs and never reach the network.

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `RAG_BACKEND` | `local` | `local` or `bedrock` |
| `RAG_KNOWLEDGE_BASE_ID` | unset | Required for `bedrock` |
| `RAG_AWS_REGION` | `us-east-1` | Region for Bedrock clients |
| `RAG_MODEL_ID` | `amazon.nova-lite-v1:0` | Generation model |
| `RAG_TOP_K` | `3` | Passages retrieved per question |
| `RAG_CHUNK_WORDS`, `RAG_OVERLAP_WORDS` | `120`, `20` | Chunking for `ingest` |
