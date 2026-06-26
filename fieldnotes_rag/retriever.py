"""Top-k retrieval with query expansion and deduplication."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .index import HybridIndex, IndexedChunk
from .citation import Citation, build_citations
from .tokenizer import tokenize, ngrams
from .query_expansion import expand_query as _expand_query


def _deduplicate(
    scored: List[Tuple[IndexedChunk, float]],
    max_per_doc: int = 2,
) -> List[Tuple[IndexedChunk, float]]:
    """Allow at most max_per_doc chunks per source document."""
    seen: Dict[str, int] = {}
    result: List[Tuple[IndexedChunk, float]] = []
    for chunk, score in scored:
        count = seen.get(chunk.doc_id, 0)
        if count < max_per_doc:
            result.append((chunk, score))
            seen[chunk.doc_id] = count + 1
    return result


class Retriever:
    """High-level retrieval interface over a HybridIndex."""

    def __init__(
        self,
        index: HybridIndex,
        top_k: int = 5,
        min_score: float = 0.0,
        max_per_doc: int = 2,
        expand_query: bool = True,
    ) -> None:
        self.index = index
        self.top_k = top_k
        self.min_score = min_score
        self.max_per_doc = max_per_doc
        self.expand_query = expand_query

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[IndexedChunk, float]]:
        """Return top_k (chunk, score) pairs for the query."""
        k = top_k if top_k is not None else self.top_k

        effective_query = _expand_query(query) if self.expand_query else query

        # Retrieve more candidates to allow dedup filtering
        candidates = self.index.score(effective_query, top_k=k * 3)

        # Filter by minimum score
        candidates = [(c, s) for c, s in candidates if s >= self.min_score]

        # Deduplicate per document
        deduped = _deduplicate(candidates, max_per_doc=self.max_per_doc)

        return deduped[:k]

    def retrieve_with_citations(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Tuple[List[Tuple[IndexedChunk, float]], List[Citation]]:
        """Retrieve chunks and build citations in one call."""
        results = self.retrieve(query, top_k=top_k)
        citations = build_citations(results)
        return results, citations
