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


# -----------------------------------------------------------------------
# Argparse type validators
# -----------------------------------------------------------------------

def _positive_int(value: str) -> int:
    """Argparse type: integer that must be >= 1."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value!r}"
        )
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"must be >= 1, got {n}"
        )
    return n


def _unit_float(value: str) -> float:
    """Argparse type: float that must be in [0.0, 1.0]."""
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a number between 0.0 and 1.0, got {value!r}"
        )
    if not (0.0 <= f <= 1.0):
        raise argparse.ArgumentTypeError(
            f"must be between 0.0 and 1.0, got {f}"
        )
    return f


def _load_config(args: argparse.Namespace) -> Config:
    """Resolve Config from optional --config file + env vars + CLI flags."""
    config_file = getattr(args, "config", None)
    if config_file and Path(config_file).exists():
        config = Config.from_file(config_file)
    else:
        config = Config.from_env()
    return config


def _build_index(corpus_dir: str, config: Config) -> HybridIndex:
    """Load, chunk, and index a corpus directory."""
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        print(f"[error] Corpus directory not found: {corpus_dir!r}", file=sys.stderr)
        sys.exit(1)
    if not corpus_path.is_dir():
        print(f"[error] Not a directory: {corpus_dir!r}", file=sys.stderr)
        sys.exit(1)

    loader = DocumentLoader(extensions=config.extensions, encoding=config.encoding)
    docs = loader.load_directory(corpus_dir)
    if not docs:
        print(
            f"[error] No {'/'.join(config.extensions)} files found in {corpus_dir!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    chunker = Chunker(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    chunks = chunker.chunk_documents(docs)
    if not chunks:
        print(
            "[error] No chunks produced. Check that corpus files have sufficient content.",
            file=sys.stderr,
        )
        sys.exit(1)

    idx = HybridIndex(alpha=config.hybrid_alpha)
    idx.build(chunks)
    return idx


def _load_index(index_path: str) -> HybridIndex:
    """Load an existing index, with a helpful error message if missing."""
    if not Path(index_path).exists():
        print(
            f"[error] Index not found: {index_path!r}.\n"
            "Run 'fieldnotes ingest <corpus-dir> --index <path>' to build the index.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        return HybridIndex.load(index_path)
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"[error] Index file is corrupt or incompatible: {exc}", file=sys.stderr)
        sys.exit(1)


# -----------------------------------------------------------------------
# Subcommand handlers
# -----------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    """Load, chunk, and index a corpus directory."""
    config = _load_config(args)
    if getattr(args, "chunk_size", None):
        config.chunk_size = args.chunk_size
    if getattr(args, "chunk_overlap", None):
        config.chunk_overlap = args.chunk_overlap
    if getattr(args, "alpha", None) is not None:
        config.hybrid_alpha = args.alpha

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
    if not args.query.strip():
        print(
            "[error] Query must not be empty.\n"
            "  Try: fieldnotes ask 'What birds live near fast-moving streams?'",
            file=sys.stderr,
        )
        sys.exit(1)
    config = _load_config(args)
    index_path = args.index or config.index_path
    idx = _load_index(index_path)

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
    config = _load_config(args)
    index_path = args.index or config.index_path
    idx = _load_index(index_path)

    fixtures_path = args.fixtures
    if not Path(fixtures_path).exists():
        print(f"[error] Fixtures file not found: {fixtures_path!r}", file=sys.stderr)
        sys.exit(1)

    retriever = Retriever(idx, top_k=args.top_k or config.top_k)
    synthesizer = Synthesizer()
    evaluator = Evaluator(retriever, synthesizer)

    try:
        queries = Evaluator.load_fixtures(fixtures_path)
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"[error] Invalid fixtures file: {exc}", file=sys.stderr)
        sys.exit(1)

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
    config = _load_config(args)
    index_path = args.index or config.index_path
    idx = _load_index(index_path)
    print(f"Index: {index_path}")
    print(f"  Documents: {idx.num_docs}")
    print(f"  Chunks:    {idx.num_chunks}")
    print(f"  Vocab:     {idx.vocab_size} terms")
    print(f"  Alpha:     {idx.alpha} (TF-IDF weight)")
    print(f"  Buckets:   {idx.n_buckets} (hashed feature buckets)")


def cmd_export(args: argparse.Namespace) -> None:
    """Export corpus chunks as structured JSON."""
    config = _load_config(args)
    index_path = args.index or config.index_path
    idx = _load_index(index_path)
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  fieldnotes ingest examples/field_corpus --index my-index.json\n"
            "  fieldnotes ask 'What birds live near fast streams?' --index my-index.json\n"
            "  fieldnotes eval --index my-index.json --fixtures examples/eval_fixtures.json\n"
            "  fieldnotes stats --index my-index.json\n"
        ),
    )
    parser.add_argument(
        "--config", "-C", default=None, metavar="FILE",
        help="Optional JSON config file (overrides defaults before env/CLI flags)",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Load and index a corpus directory")
    p_ingest.add_argument("corpus_dir", help="Path to directory with .md/.txt files")
    p_ingest.add_argument("--index", "-i", default=None, help="Output index file path")
    p_ingest.add_argument("--chunk-size", type=_positive_int, default=None, dest="chunk_size",
                          help="Max words per chunk, >= 1 (default: 200)")
    p_ingest.add_argument("--chunk-overlap", type=_positive_int, default=None, dest="chunk_overlap",
                          help="Overlap in words between chunks, >= 1 (default: 40)")
    p_ingest.add_argument("--alpha", type=_unit_float, default=None,
                          help="TF-IDF weight in hybrid score, 0.0–1.0 (default: 0.6)")
    p_ingest.add_argument("--config", "-C", default=None, metavar="FILE")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question and get a cited answer")
    p_ask.add_argument("query", help="Your question")
    p_ask.add_argument("--index", "-i", default=None, help="Index file path")
    p_ask.add_argument("--top-k", "-k", type=_positive_int, default=None, dest="top_k",
                       help="Number of passages to retrieve, >= 1 (default: 5)")
    p_ask.add_argument("--min-score", type=_unit_float, default=None, dest="min_score",
                       help="Minimum relevance score threshold, 0.0–1.0 (default: 0.0)")
    p_ask.add_argument("--no-expand", action="store_true", dest="no_expand",
                       help="Disable query expansion / synonym lookup")
    p_ask.add_argument("--json", action="store_true", help="Output as JSON")
    p_ask.add_argument("--width", type=_positive_int, default=80, help="Text wrap width, >= 1 (default: 80)")
    p_ask.add_argument("--config", "-C", default=None, metavar="FILE")

    # eval
    p_eval = sub.add_parser("eval", help="Run evaluation fixtures")
    p_eval.add_argument(
        "--fixtures", "-f",
        default="examples/eval_fixtures.json",
        help="Path to evaluation fixtures JSON (default: examples/eval_fixtures.json)",
    )
    p_eval.add_argument("--index", "-i", default=None)
    p_eval.add_argument("--top-k", type=_positive_int, default=None, dest="top_k",
                        help="Number of passages to retrieve, >= 1 (default: config)")
    p_eval.add_argument("--json", action="store_true", help="Output as JSON")
    p_eval.add_argument("--output", "-o", default=None, help="Save JSON report to file")
    p_eval.add_argument("--config", "-C", default=None, metavar="FILE")

    # stats
    p_stats = sub.add_parser("stats", help="Print index statistics")
    p_stats.add_argument("--index", "-i", default=None)
    p_stats.add_argument("--config", "-C", default=None, metavar="FILE")

    # export
    p_export = sub.add_parser("export", help="Export chunks as JSON")
    p_export.add_argument("--index", "-i", default=None)
    p_export.add_argument("--output", "-o", default=None, help="Output JSON file path")
    p_export.add_argument("--config", "-C", default=None, metavar="FILE")

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
