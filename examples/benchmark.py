#!/usr/bin/env python3
"""
Benchmark script: measure index build time, query latency, and eval metrics.

Usage:
    python examples/benchmark.py [--corpus-dir PATH] [--runs N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from fieldnotes_rag.config import Config
from fieldnotes_rag.loader import DocumentLoader
from fieldnotes_rag.chunker import Chunker
from fieldnotes_rag.index import HybridIndex
from fieldnotes_rag.retriever import Retriever
from fieldnotes_rag.synthesis import Synthesizer
from fieldnotes_rag.evaluator import Evaluator

BENCHMARK_QUERIES = [
    "What should I record after seeing a streamside bird at dusk?",
    "Is it safe to cross the stream after heavy rain?",
    "What birds are common near fast-moving streams?",
    "What is the role of Red Alder in the riparian zone?",
    "How does wind affect bird survey accuracy?",
]


def benchmark(corpus_dir: str, fixtures_path: str, runs: int = 5) -> None:
    config = Config()

    # 1. Ingest benchmark
    print("=" * 60)
    print("FieldNotes RAG Lab — Benchmark")
    print("=" * 60)
    print(f"\nCorpus: {corpus_dir}")
    print(f"Runs per query: {runs}")

    t0 = time.perf_counter()
    loader = DocumentLoader()
    docs = loader.load_directory(corpus_dir)
    chunker = Chunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    chunks = chunker.chunk_documents(docs)
    idx = HybridIndex(alpha=config.hybrid_alpha)
    idx.build(chunks)
    ingest_time = time.perf_counter() - t0

    print(f"\nIngest:")
    print(f"  Documents: {idx.num_docs}")
    print(f"  Chunks:    {idx.num_chunks}")
    print(f"  Vocab:     {idx.vocab_size} terms")
    print(f"  Time:      {ingest_time * 1000:.1f} ms")

    # 2. Query latency benchmark
    retriever = Retriever(idx, top_k=5, expand_query=True)
    synthesizer = Synthesizer(max_context_chunks=5)

    print(f"\nQuery latency (top-k=5, {runs} runs each):")
    total_retrieve_ms = 0.0
    total_synth_ms = 0.0

    for query in BENCHMARK_QUERIES:
        retrieve_times: List[float] = []
        synth_times: List[float] = []

        for _ in range(runs):
            t0 = time.perf_counter()
            chunks_r, _ = retriever.retrieve_with_citations(query)
            retrieve_times.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            synthesizer.synthesize(query, chunks_r)
            synth_times.append((time.perf_counter() - t0) * 1000)

        avg_r = sum(retrieve_times) / runs
        avg_s = sum(synth_times) / runs
        total_retrieve_ms += avg_r
        total_synth_ms += avg_s
        print(f"  [{avg_r:6.2f}ms retrieve | {avg_s:5.2f}ms synth] {query[:50]}")

    print(f"\n  Avg retrieve: {total_retrieve_ms / len(BENCHMARK_QUERIES):.2f} ms")
    print(f"  Avg synth:    {total_synth_ms / len(BENCHMARK_QUERIES):.2f} ms")

    # 3. Evaluation
    if Path(fixtures_path).exists():
        print(f"\nEvaluation ({fixtures_path}):")
        evaluator = Evaluator(retriever, synthesizer)
        queries = Evaluator.load_fixtures(fixtures_path)
        t0 = time.perf_counter()
        report = evaluator.run(queries)
        eval_time = (time.perf_counter() - t0) * 1000
        print(f"  Queries:           {len(queries)}")
        print(f"  Recall@5:          {report.mean_recall:.3f}")
        print(f"  MRR:               {report.mean_mrr:.3f}")
        print(f"  Citation accuracy: {report.mean_citation_accuracy:.3f}")
        print(f"  Answer coverage:   {report.mean_answer_coverage:.3f}")
        print(f"  Composite:         {report.mean_composite:.3f}")
        print(f"  Eval time:         {eval_time:.1f} ms total")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FieldNotes RAG Lab benchmark")
    parser.add_argument("--corpus-dir", "-c", default="examples/field_corpus")
    parser.add_argument("--fixtures", "-f", default="examples/eval_fixtures.json")
    parser.add_argument("--runs", "-r", type=int, default=5)
    args = parser.parse_args()
    benchmark(args.corpus_dir, args.fixtures, args.runs)
