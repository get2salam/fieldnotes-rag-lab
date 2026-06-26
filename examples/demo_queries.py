#!/usr/bin/env python3
"""
Demo: run a batch of representative field-research queries through the RAG pipeline
and print cited answers with evaluation scores.

Usage:
    python examples/demo_queries.py [--index PATH] [--corpus-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the package is importable when run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from fieldnotes_rag.config import Config
from fieldnotes_rag.loader import DocumentLoader
from fieldnotes_rag.chunker import Chunker
from fieldnotes_rag.index import HybridIndex
from fieldnotes_rag.retriever import Retriever
from fieldnotes_rag.synthesis import Synthesizer

DEMO_QUERIES = [
    "What should I record after seeing a streamside bird at dusk?",
    "Is it safe to cross the stream after heavy rain?",
    "What birds are common near fast-moving streams at dusk?",
    "What is the role of Red Alder in the riparian zone?",
    "How does wind affect bird survey accuracy?",
    "What field marks distinguish the Harlequin Duck?",
    "What safety precautions are needed before stream crossings?",
    "What amphibians indicate high water quality in cold streams?",
]


def run_demo(index_path: str, corpus_dir: str | None, top_k: int = 3) -> None:
    if Path(index_path).exists():
        print(f"Loading existing index from {index_path}")
        idx = HybridIndex.load(index_path)
    elif corpus_dir:
        print(f"Building index from corpus: {corpus_dir}")
        config = Config()
        loader = DocumentLoader()
        docs = loader.load_directory(corpus_dir)
        chunker = Chunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        chunks = chunker.chunk_documents(docs)
        idx = HybridIndex(alpha=config.hybrid_alpha)
        idx.build(chunks)
        print(f"  Indexed {idx.num_docs} docs, {idx.num_chunks} chunks, {idx.vocab_size} terms")
    else:
        print("[error] Provide --index or --corpus-dir", file=sys.stderr)
        sys.exit(1)

    retriever = Retriever(idx, top_k=top_k, expand_query=True)
    synthesizer = Synthesizer(max_context_chunks=top_k)

    print(f"\n{'=' * 70}")
    print("FieldNotes RAG Lab — Demo Queries")
    print(f"{'=' * 70}\n")

    for i, query in enumerate(DEMO_QUERIES, start=1):
        chunks, citations = retriever.retrieve_with_citations(query)
        result = synthesizer.synthesize(query, chunks)
        print(f"[Query {i}] {query}")
        print("-" * 60)
        # Print first 300 chars of answer
        answer_preview = result.answer[:300]
        if len(result.answer) > 300:
            answer_preview += "…"
        print(f"Answer: {answer_preview}")
        print(f"Sources ({len(citations)}):")
        for c in citations:
            print(f"  {c.format_short()} (score={c.score:.4f})")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FieldNotes RAG Lab demo")
    parser.add_argument("--index", "-i", default=".fieldnotes-index.json")
    parser.add_argument("--corpus-dir", "-c", default="examples/field_corpus")
    parser.add_argument("--top-k", "-k", type=int, default=3)
    args = parser.parse_args()

    run_demo(args.index, args.corpus_dir, args.top_k)
