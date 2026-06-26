"""Command-line interface for FieldNotes RAG Lab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config
from .loader import DocumentLoader
from .chunker import Chunker
from .index import HybridIndex
from .retriever import Retriever
from .synthesis import Synthesizer
from .evaluator import Evaluator


def _build_index(corpus_dir: str, config: Config) -> HybridIndex:
    loader = DocumentLoader(extensions=config.extensions, encoding=config.encoding)
    docs = loader.load_directory(corpus_dir)
    if not docs:
        print(f"[error] No documents found in {corpus_dir!r}", file=sys.stderr)
        sys.exit(1)

    chunker = Chunker(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    chunks = chunker.chunk_documents(docs)
    if not chunks:
        print("[error] No chunks produced — check corpus content.", file=sys.stderr)
        sys.exit(1)

    idx = HybridIndex(alpha=config.hybrid_alpha)
    idx.build(chunks)
    return idx


# -----------------------------------------------------------------------
# Subcommand handlers
# -----------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    """Load, chunk, and index a corpus directory."""
    config = Config.from_env()
    if args.chunk_size:
        config.chunk_size = args.chunk_size
    if args.chunk_overlap:
        config.chunk_overlap = args.chunk_overlap

    print(f"Loading documents from: {args.corpus_dir}")
    idx = _build_index(args.corpus_dir, config)

    index_path = args.index or config.index_path
    idx.save(index_path)
    print(
        f"Index saved to: {index_path}\n"
        f"  Documents: {idx.num_docs}\n"
        f"  Chunks:    {idx.num_chunks}\n"
        f"  Vocab:     {idx.vocab_size} terms"
    )


def cmd_ask(args: argparse.Namespace) -> None:
    """Retrieve relevant passages and synthesise a cited answer."""
    index_path = args.index or Config().index_path
    if not Path(index_path).exists():
        print(
            f"[error] Index not found: {index_path!r}. "
            "Run 'fieldnotes ingest <dir>' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    config = Config.from_env()
    idx = HybridIndex.load(index_path)
    retriever = Retriever(
        idx,
        top_k=args.top_k or config.top_k,
        min_score=args.min_score if args.min_score is not None else config.min_score,
        expand_query=not args.no_expand,
    )
    synthesizer = Synthesizer(max_context_chunks=args.top_k or config.top_k)

    chunks, citations = retriever.retrieve_with_citations(args.query)
    result = synthesizer.synthesize(args.query, chunks)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print()
        print(result.format(width=args.width or 80))


def cmd_eval(args: argparse.Namespace) -> None:
    """Run evaluation fixtures against the index."""
    index_path = args.index or Config().index_path
    if not Path(index_path).exists():
        print(f"[error] Index not found: {index_path!r}", file=sys.stderr)
        sys.exit(1)

    fixtures_path = args.fixtures
    if not Path(fixtures_path).exists():
        print(f"[error] Fixtures file not found: {fixtures_path!r}", file=sys.stderr)
        sys.exit(1)

    config = Config.from_env()
    idx = HybridIndex.load(index_path)
    retriever = Retriever(idx, top_k=args.top_k or config.top_k)
    synthesizer = Synthesizer()
    evaluator = Evaluator(retriever, synthesizer)

    queries = Evaluator.load_fixtures(fixtures_path)
    report = evaluator.run(queries)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(report.format())

    if args.output:
        Path(args.output).write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nReport saved to: {args.output}")


def cmd_stats(args: argparse.Namespace) -> None:
    """Print statistics for an existing index."""
    index_path = args.index or Config().index_path
    if not Path(index_path).exists():
        print(f"[error] Index not found: {index_path!r}", file=sys.stderr)
        sys.exit(1)

    idx = HybridIndex.load(index_path)
    print(f"Index: {index_path}")
    print(f"  Documents: {idx.num_docs}")
    print(f"  Chunks:    {idx.num_chunks}")
    print(f"  Vocab:     {idx.vocab_size} terms")
    print(f"  Alpha:     {idx.alpha} (TF-IDF weight)")
    print(f"  Buckets:   {idx.n_buckets} (hashed feature buckets)")


def cmd_export(args: argparse.Namespace) -> None:
    """Export corpus chunks as structured JSON."""
    index_path = args.index or Config().index_path
    if not Path(index_path).exists():
        print(f"[error] Index not found: {index_path!r}", file=sys.stderr)
        sys.exit(1)

    idx = HybridIndex.load(index_path)
    chunks_data = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "title": c.title,
            "source_path": c.source_path,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "text": c.text,
        }
        for c in idx._chunks
    ]
    output = args.output or "fieldnotes-export.json"
    Path(output).write_text(
        json.dumps(chunks_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Exported {len(chunks_data)} chunks to: {output}")


# -----------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fieldnotes",
        description=(
            "FieldNotes RAG Lab — local-first retrieval-augmented generation "
            "for field research notebooks."
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Load and index a corpus directory")
    p_ingest.add_argument("corpus_dir", help="Path to directory with .md/.txt files")
    p_ingest.add_argument("--index", "-i", default=None, help="Output index file path")
    p_ingest.add_argument("--chunk-size", type=int, default=None, dest="chunk_size")
    p_ingest.add_argument("--chunk-overlap", type=int, default=None, dest="chunk_overlap")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question and get a cited answer")
    p_ask.add_argument("query", help="Your question")
    p_ask.add_argument("--index", "-i", default=None, help="Index file path")
    p_ask.add_argument("--top-k", "-k", type=int, default=None, dest="top_k")
    p_ask.add_argument("--min-score", type=float, default=None, dest="min_score")
    p_ask.add_argument("--no-expand", action="store_true", dest="no_expand",
                       help="Disable query expansion")
    p_ask.add_argument("--json", action="store_true", help="Output as JSON")
    p_ask.add_argument("--width", type=int, default=80, help="Text wrap width")

    # eval
    p_eval = sub.add_parser("eval", help="Run evaluation fixtures")
    p_eval.add_argument(
        "--fixtures", "-f",
        default="examples/eval_fixtures.json",
        help="Path to evaluation fixtures JSON",
    )
    p_eval.add_argument("--index", "-i", default=None)
    p_eval.add_argument("--top-k", type=int, default=None, dest="top_k")
    p_eval.add_argument("--json", action="store_true")
    p_eval.add_argument("--output", "-o", default=None, help="Save report to file")

    # stats
    p_stats = sub.add_parser("stats", help="Print index statistics")
    p_stats.add_argument("--index", "-i", default=None)

    # export
    p_export = sub.add_parser("export", help="Export chunks as JSON")
    p_export.add_argument("--index", "-i", default=None)
    p_export.add_argument("--output", "-o", default=None)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "ingest": cmd_ingest,
        "ask": cmd_ask,
        "eval": cmd_eval,
        "stats": cmd_stats,
        "export": cmd_export,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
