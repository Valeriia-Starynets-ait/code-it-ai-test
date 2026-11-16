import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import boto3

from ragkb.config import Settings
from ragkb.errors import ConfigError, NoEvidenceError, RagError
from ragkb.evaluation import EvalReport, load_cases, run_eval
from ragkb.generation import BedrockGenerator, ExtractiveGenerator, Generator, answer_question
from ragkb.ingestion import ingest
from ragkb.models import Answer
from ragkb.retrieval import BedrockRetriever, LocalRetriever, Retriever, load_chunks

DEFAULT_INDEX = Path("build/kb")
DEFAULT_DOCS = Path("data/docs")
DEFAULT_QUESTIONS = Path("data/eval/questions.json")
DEFAULT_THRESHOLD = 0.7
EXIT_FAILED = 1
EXIT_ERROR = 2


def build_pipeline(settings: Settings, index: Path) -> tuple[Retriever, Generator]:
    if settings.backend == "local":
        return LocalRetriever(load_chunks(index)), ExtractiveGenerator()
    if not settings.knowledge_base_id:
        raise ConfigError("the bedrock backend needs RAG_KNOWLEDGE_BASE_ID")
    retriever = BedrockRetriever(
        boto3.client("bedrock-agent-runtime", region_name=settings.aws_region), settings.knowledge_base_id
    )
    generator = BedrockGenerator(boto3.client("bedrock-runtime", region_name=settings.aws_region), settings.model_id)
    return retriever, generator


def _known_sources(settings: Settings, index: Path) -> set[str] | None:
    if settings.backend != "local":
        return None
    return {c.source for c in load_chunks(index)}


def _format_answer(answer: Answer) -> str:
    lines = [answer.text, "", "Sources:"]
    by_marker = dict(enumerate(answer.passages, start=1))
    for citation in answer.citations:
        passage = by_marker[citation.marker]
        lines.append(f"[{citation.marker}] {citation.source} (chunk {citation.chunk_id}, score {passage.score:.2f})")
    return "\n".join(lines)


def _format_report(report: EvalReport, threshold: float) -> str:
    lines = [f"{'question':<70} faithfulness answer_relevance source_hit"]
    for r in report.results:
        lines.append(f"{r.question[:68]:<70} {r.faithfulness:>12.2f} {r.answer_relevance:>16.2f} {r.source_hit!s:>10}")
    lines += [
        "",
        f"faithfulness: {report.mean_faithfulness:.2f}",
        f"answer_relevance: {report.mean_answer_relevance:.2f}",
        f"source_hit_rate: {report.source_hit_rate:.2f}",
        f"threshold: {threshold:.2f}",
        f"RESULT: {'PASS' if report.passes(threshold) else 'FAIL'}",
    ]
    return "\n".join(lines)


def _cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    result = ingest(args.docs, args.out, settings.chunk_words, settings.overlap_words)
    print(f"Staged {result.chunk_count} chunks from {result.document_count} documents in {result.output_dir}")
    return 0


def _cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    retriever, generator = build_pipeline(settings, args.index)
    try:
        answer = answer_question(
            args.question, retriever, generator, settings.top_k, _known_sources(settings, args.index)
        )
    except NoEvidenceError:
        print("no relevant passages found for that question", file=sys.stderr)
        return EXIT_FAILED
    print(_format_answer(answer))
    return 0


def _cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    retriever, generator = build_pipeline(settings, args.index)
    report = run_eval(load_cases(args.questions), retriever, generator, settings.top_k)
    print(_format_report(report, args.threshold))
    return 0 if report.passes(args.threshold) else EXIT_FAILED


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragkb", description="Bedrock Knowledge Base RAG proof of concept")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", choices=["local", "bedrock"], help="override RAG_BACKEND")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", parents=[common], help="chunk documents and stage a Bedrock data source")
    p_ingest.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    p_ingest.add_argument("--out", type=Path, default=DEFAULT_INDEX)
    p_ingest.set_defaults(handler=_cmd_ingest)

    p_ask = sub.add_parser("ask", parents=[common], help="answer a question with citations")
    p_ask.add_argument("question")
    p_ask.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p_ask.set_defaults(handler=_cmd_ask)

    p_eval = sub.add_parser(
        "eval", parents=[common], help="score faithfulness and answer relevance; non-zero exit below threshold"
    )
    p_eval.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p_eval.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p_eval.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p_eval.set_defaults(handler=_cmd_eval)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings()
    if args.backend:
        settings = settings.model_copy(update={"backend": args.backend})
    try:
        return int(args.handler(args, settings))
    except RagError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
