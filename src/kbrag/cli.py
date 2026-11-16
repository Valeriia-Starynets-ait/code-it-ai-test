import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import boto3

from kbrag.config import Settings
from kbrag.errors import ConfigError, KbragError, NoEvidenceError
from kbrag.evaluation import load_eval_set, run_eval
from kbrag.generation import BedrockGenerator, ExtractiveGenerator, Generator, answer_question, verify_citations
from kbrag.ingestion import chunk_document, load_chunks, load_documents, load_manifest, prepare_data_source
from kbrag.retrieval import BedrockRetriever, InMemoryRetriever, Retriever

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NO_EVIDENCE = 2
DEFAULT_THRESHOLD = 0.7


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except NoEvidenceError as exc:
        print(f"No supporting passages: {exc}", file=sys.stderr)
        return EXIT_NO_EVIDENCE
    except KbragError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kbrag", description="Bedrock Knowledge Base RAG proof of concept")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="chunk local documents into a data source folder")
    ingest.add_argument("--docs", type=Path, required=True)
    ingest.add_argument("--out", type=Path, required=True)
    ingest.set_defaults(handler=_ingest)

    ask = sub.add_parser("ask", help="answer a question with citations")
    ask.add_argument("question")
    _add_backend_args(ask)
    ask.set_defaults(handler=_ask)

    evaluate = sub.add_parser("eval", help="score faithfulness and answer relevance")
    evaluate.add_argument("--questions", type=Path, required=True)
    evaluate.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    _add_backend_args(evaluate)
    evaluate.set_defaults(handler=_eval)
    return parser


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, required=True, help="folder produced by ingest")
    parser.add_argument("--backend", choices=["fake", "bedrock"], default="fake")


def _ingest(args: argparse.Namespace) -> int:
    chunks = [c for doc in load_documents(args.docs) for c in chunk_document(doc)]
    prepare_data_source(chunks, args.out)
    print(f"Prepared {len(chunks)} chunks in {args.out}")
    return EXIT_OK


def _ask(args: argparse.Namespace) -> int:
    settings = Settings()
    retriever, generator = _build_backend(args.backend, args.data, settings)
    answer = answer_question(args.question, retriever, generator, settings.top_k)
    verify_citations(answer, set(load_manifest(args.data)["sources"]))
    print(answer.text)
    print("\nSources:")
    for citation in answer.citations:
        print(f"[{citation.marker}] {citation.source} ({citation.chunk_id})")
    return EXIT_OK


def _eval(args: argparse.Namespace) -> int:
    settings = Settings()
    retriever, generator = _build_backend(args.backend, args.data, settings)
    report = run_eval(load_eval_set(args.questions), retriever, generator, settings.top_k)
    for result in report.results:
        print(
            f"faithfulness={result.faithfulness:.2f} answer_relevance={result.answer_relevance:.2f} "
            f"answered={result.answered} question={result.question}"
        )
    print(f"mean faithfulness: {report.mean_faithfulness:.2f}")
    print(f"mean answer_relevance: {report.mean_answer_relevance:.2f}")
    if report.passed(args.threshold):
        print(f"PASS (threshold {args.threshold:.2f})")
        return EXIT_OK
    print(f"FAIL (threshold {args.threshold:.2f})", file=sys.stderr)
    return EXIT_ERROR


def _build_backend(backend: str, data_dir: Path, settings: Settings) -> tuple[Retriever, Generator]:
    if backend == "fake":
        return InMemoryRetriever(load_chunks(data_dir)), ExtractiveGenerator()
    if not settings.knowledge_base_id:
        raise ConfigError("Set KBRAG_KNOWLEDGE_BASE_ID to use the bedrock backend")
    session = boto3.Session(region_name=settings.aws_region)
    retriever = BedrockRetriever(session.client("bedrock-agent-runtime"), settings.knowledge_base_id)
    generator = BedrockGenerator(session.client("bedrock-runtime"), settings.model_id)
    return retriever, generator
