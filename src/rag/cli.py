"""Command line interface for the RAG pipeline.

Subcommands:
  ingest   load a JSONL corpus (or the built-in sample) and report indexed count
  ask      answer a question and print the cited answer
  eval     run the RAGAS-style harness over a built-in eval set and gate on it

``--backend fake`` (the default) uses the in-memory client, so no AWS
credentials or network access are required. ``--backend bedrock`` wires the real
:class:`rag.bedrock.BedrockKnowledgeBase` adapter against live AWS and is never
exercised by the test suite.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from rag.config import Settings
from rag.eval import EvalSample, evaluate
from rag.generation import TemplateGenerator
from rag.ingestion import load_documents_from_jsonl, sample_corpus
from rag.models import Answer
from rag.pipeline import RagPipeline
from rag.ports import Generator, Ingestor, Retriever
from rag.retrieval import FakeKnowledgeBaseClient

logger = logging.getLogger(__name__)

_EVAL_SET: tuple[EvalSample, ...] = (
    EvalSample(
        question="What does a Bedrock Knowledge Base manage?",
        expected_answer=("It manages ingestion, embedding, chunking and vector search for RAG."),
        relevant_doc_ids=("kb-001",),
        expected_pages=(1,),
    ),
    EvalSample(
        question="Which embedding model is the default and how many dimensions?",
        expected_answer="Titan Text Embeddings v2 with 1024 dimensions.",
        relevant_doc_ids=("kb-002",),
        expected_pages=(2,),
    ),
    EvalSample(
        question="How can an answer be traced back to its source?",
        expected_answer="Through ranked passages with relevance scores and page citations.",
        relevant_doc_ids=("kb-003",),
        expected_pages=(3,),
    ),
)


def _build_bedrock_backend(settings: Settings) -> tuple[Retriever, Ingestor, Generator]:
    # Imported lazily so the default fake path needs no boto3 import or creds.
    import boto3

    from rag.bedrock import BedrockKnowledgeBase

    runtime = boto3.client("bedrock-agent-runtime", region_name=settings.aws_region)
    agent = boto3.client("bedrock-agent", region_name=settings.aws_region)
    adapter = BedrockKnowledgeBase(runtime, agent, settings)
    return adapter, adapter, adapter


def _build_ingestor(backend: str, settings: Settings) -> Ingestor:
    """An empty ingestor (fresh fake client, or the real adapter)."""
    if backend == "bedrock":
        retriever, ingestor, _ = _build_bedrock_backend(settings)
        return ingestor
    return FakeKnowledgeBaseClient(min_score=settings.min_score)


def _build_pipeline(backend: str, settings: Settings) -> RagPipeline:
    """A query-ready pipeline. The fake backend is pre-seeded with the sample corpus."""
    if backend == "bedrock":
        retriever, _, generator = _build_bedrock_backend(settings)
        return RagPipeline(retriever, generator, top_k=settings.top_k)
    client = FakeKnowledgeBaseClient(min_score=settings.min_score)
    client.ingest(sample_corpus())
    return RagPipeline(client, TemplateGenerator(), top_k=settings.top_k)


def _format_answer(answer: Answer) -> str:
    lines = [answer.answer.text, "", "Citations:"]
    for citation in answer.answer.citations:
        page = f" p.{citation.page}" if citation.page is not None else ""
        lines.append(f"  [{citation.marker}] {citation.doc_id} ({citation.source}){page}")
    return "\n".join(lines)


def _cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    ingestor = _build_ingestor(args.backend, settings)
    documents = load_documents_from_jsonl(Path(args.corpus)) if args.corpus else list(sample_corpus())
    indexed = ingestor.ingest(documents)
    print(f"ingested {indexed} document(s)")
    return 0


def _cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    pipeline = _build_pipeline(args.backend, settings)
    answer = pipeline.answer(args.question)
    print(_format_answer(answer))
    return 0


def _cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    pipeline = _build_pipeline(args.backend, settings)
    answers = [pipeline.answer(sample.question) for sample in _EVAL_SET]
    report = evaluate(answers, _EVAL_SET)
    print("Retrieval:")
    print(f"  context_recall   = {report.context_recall:.3f}")
    print(f"  page_match(+/-1) = {report.page_match:.3f}")
    print("Answer:")
    print(f"  faithfulness     = {report.faithfulness:.3f}")
    print(f"  answer_relevancy = {report.answer_relevancy:.3f}")
    print(f"imputed rows: {report.imputed_count} of {report.sample_count}")
    passed = report.meets_thresholds()
    print(f"gate: {'PASS' if passed else 'FAIL'}")
    return 0 if passed or not args.gate else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="Bedrock KB RAG pipeline.")
    parser.add_argument(
        "--backend",
        choices=("fake", "bedrock"),
        default="fake",
        help="fake (default, no AWS) or bedrock (live Knowledge Base)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest a corpus")
    p_ingest.add_argument("--corpus", help="path to a JSONL corpus; omit for the built-in sample")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_ask = sub.add_parser("ask", help="answer a question with citations")
    p_ask.add_argument("question", help="the question to answer")
    p_ask.set_defaults(func=_cmd_ask)

    p_eval = sub.add_parser("eval", help="run the RAGAS-style eval harness")
    p_eval.add_argument(
        "--gate",
        action="store_true",
        help="exit non-zero if the eval is below threshold",
    )
    p_eval.set_defaults(func=_cmd_eval)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    settings = Settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args, settings))


if __name__ == "__main__":
    sys.exit(main())
