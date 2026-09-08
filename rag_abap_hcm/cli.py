"""Command line interface for the ABAP/HCM hybrid RAG pipeline."""

from __future__ import annotations

import argparse
import sys

from .config import RAGConfig
from .pipeline import ABAPHCMPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-abap-hcm",
        description="RAG ibrido per repository SAP ABAP/HCM (ChromaDB + Ollama + Qwen).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest ABAP source files into the vector store")
    ingest_parser.add_argument("path", help="Directory containing ABAP source files")

    query_parser = subparsers.add_parser("query", help="Ask a question against the ingested repository")
    query_parser.add_argument("question", help="Natural language question")
    query_parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = RAGConfig()
    pipeline = ABAPHCMPipeline(config)

    if args.command == "ingest":
        count = pipeline.ingest_directory(args.path)
        print(f"Ingested {count} unique chunk(s) from {args.path}")
        return 0

    if args.command == "query":
        result = pipeline.query(args.question, top_k=args.top_k)
        print(result.answer)
        print("\nFonti:")
        for source in result.sources:
            meta = source.metadata or {}
            print(
                f"- {meta.get('source_path', '?')} "
                f"[{meta.get('unit_type', '?')} {meta.get('name', '')}] "
                f"(score={source.combined_score:.3f})"
            )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
