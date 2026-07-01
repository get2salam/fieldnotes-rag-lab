#!/usr/bin/env python3
"""
Query diagnosis tool: show exactly why a query returns the results it does.

Breaks the hybrid retrieval score for each result into its TF-IDF and
hashed-cosine components and reports which expanded query terms drove each
chunk's relevance.  Useful for tuning synonyms, alpha, and corpus coverage.

Usage:
    python examples/explain_query.py \\
        --query "What salamanders live in cold mountain streams?" \\
        --corpus-dir examples/field_corpus

    # Or with a pre-built index:
    python examples/explain_query.py \\
        --query "stream crossing at dusk" \\
        --index .fieldnotes-index.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from fieldnotes_rag.chunker import Chunker
from fieldnotes_rag.config import Config
from fieldnotes_rag.index import HybridIndex, IndexedChunk, _cosine_hashed, _hash_features
from fieldnotes_rag.loader import DocumentLoader
from fieldnotes_rag.query_expansion import expand_query
from fieldnotes_rag.tokenizer import ngrams, tokenize

# (chunk, tfidf_score, cosine_score, hybrid_score, top_contributors)
_Row = Tuple[IndexedChunk, float, float, float, List[Tuple[str, float]]]


def _score_breakdown(query: str, idx: HybridIndex, top_k: int) -> List[_Row]:
    """Compute per-chunk score components and return sorted top-k rows."""
    expanded = expand_query(query)
    q_tokens = tokenize(expanded, remove_stopwords=True)
    if not q_tokens:
        return []

    q_all = q_tokens + ngrams(q_tokens, 2)
    q_counts: Dict[str, int] = {}
    for t in q_all:
        q_counts[t] = q_counts.get(t, 0) + 1
    total_q = len(q_all) or 1

    q_hashed = _hash_features(q_all, idx.n_buckets)

    rows: List[_Row] = []
    for chunk, h_vec in zip(idx._chunks, idx._hashed_vecs):
        tfidf = 0.0
        contrib: List[Tuple[str, float]] = []
        for t, c in q_counts.items():
            q_w = (c / total_q) * idx._idf_map.get(t, 1.0)
            c_w = chunk.tf_vector.get(t, 0.0)
            contribution = q_w * c_w
            tfidf += contribution
            if c_w > 0:
                contrib.append((t, contribution))

        cosine = _cosine_hashed(q_hashed, h_vec)
        hybrid = idx.alpha * tfidf + (1.0 - idx.alpha) * cosine
        contrib.sort(key=lambda x: -x[1])
        rows.append((chunk, tfidf, cosine, hybrid, contrib))

    rows.sort(key=lambda x: -x[3])
    return rows[:top_k]


def explain(query: str, idx: HybridIndex, top_k: int) -> None:
    SEP = "=" * 70

    print(SEP)
    print(f"Explain query:  {query!r}")
    print(SEP)

    # --- 1. Query expansion ---
    expanded = expand_query(query)
    if expanded != query:
        added = expanded[len(query):].strip()
        print(f"\nExpanded terms appended: {added}")
    else:
        print("\nExpansion: no synonyms matched; query used as-is.")

    q_tokens = tokenize(expanded, remove_stopwords=True)
    print(f"Effective tokens (after stopword removal): {q_tokens}")

    # --- 2. Token IDF lookup ---
    in_corpus: List[Tuple[str, float]] = []
    out_of_corpus: List[str] = []
    for t in q_tokens:
        idf = idx._idf_map.get(t)
        if idf is not None:
            in_corpus.append((t, idf))
        else:
            out_of_corpus.append(t)

    print("\nTerm IDF in this index  (higher = rarer = more discriminating):")
    for t, idf in sorted(in_corpus, key=lambda x: -x[1])[:15]:
        bar = "▪" * min(int(idf * 8), 32)
        print(f"  {t:<24} {idf:.4f}  {bar}")
    if out_of_corpus:
        print(f"  Not in corpus: {', '.join(out_of_corpus)}")

    # --- 3. Per-result score breakdown ---
    rows = _score_breakdown(query, idx, top_k)

    if not rows:
        print("\n[No results — index is empty or all query terms were stopwords.]\n")
        return

    print(
        f"\nTop-{top_k} results  "
        f"(α={idx.alpha:.2f};  hybrid = α×TF-IDF + (1-α)×Cosine):"
    )
    for rank, (chunk, tfidf, cosine, hybrid, contrib) in enumerate(rows, start=1):
        print(
            f"\n  [{rank}] doc={chunk.doc_id}  chunk={chunk.chunk_id}"
            f"  lines {chunk.start_line}–{chunk.end_line}"
        )
        print(
            f"       hybrid={hybrid:.4f}  "
            f"= {idx.alpha:.2f}×TF-IDF({tfidf:.4f}) "
            f"+ {1 - idx.alpha:.2f}×Cosine({cosine:.4f})"
        )
        if contrib:
            top_terms = "  ".join(f"{t}({v:.4f})" for t, v in contrib[:5])
            print(f"       TF-IDF driven by: {top_terms}")
        else:
            print("       TF-IDF: 0.0000  (no query tokens matched this chunk)")
        snippet = chunk.text[:100].replace("\n", " ")
        print(f"       ┄ {snippet}…")

    # --- 4. Actionable tips ---
    top_tfidf = rows[0][1]
    if top_tfidf < 0.0005:
        print(
            "\n[Tip] TF-IDF scores are near zero: query terms do not appear in the"
            " corpus.\n      Add field notes that use this vocabulary, or extend"
            " SYNONYMS in fieldnotes_rag/query_expansion.py."
        )
    elif out_of_corpus:
        missing = ", ".join(out_of_corpus[:4])
        print(
            f"\n[Tip] {len(out_of_corpus)} expanded token(s) are absent from the"
            f" index ({missing}).\n      Consider adding notes that mention them,"
            " or adding corpus-specific synonyms."
        )

    print("\n" + SEP)


def _build_index(corpus_dir: str) -> HybridIndex:
    config = Config()
    docs = DocumentLoader().load_directory(corpus_dir)
    chunks = Chunker(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
    ).chunk_documents(docs)
    idx = HybridIndex(alpha=config.hybrid_alpha)
    idx.build(chunks)
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose why a query returns specific results: "
            "shows query expansion, token IDF values, and per-result score breakdown."
        )
    )
    parser.add_argument("--query", "-q", required=True, help="Query string to diagnose")
    parser.add_argument(
        "--index", "-i", default=None, help="Pre-built index JSON file (optional)"
    )
    parser.add_argument(
        "--corpus-dir",
        "-c",
        default="examples/field_corpus",
        help="Corpus directory used to build the index when --index is not given",
    )
    parser.add_argument("--top-k", "-k", type=int, default=5)
    args = parser.parse_args()

    if args.index and Path(args.index).exists():
        print(f"Loading index: {args.index}\n")
        idx = HybridIndex.load(args.index)
    else:
        corpus_dir = Path(args.corpus_dir)
        if not corpus_dir.exists():
            print(f"[error] corpus directory not found: {corpus_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Building index from: {corpus_dir}")
        idx = _build_index(str(corpus_dir))
        print(
            f"Indexed {idx.num_docs} docs, "
            f"{idx.num_chunks} chunks, "
            f"{idx.vocab_size} vocab terms\n"
        )

    explain(args.query, idx, args.top_k)


if __name__ == "__main__":
    main()
