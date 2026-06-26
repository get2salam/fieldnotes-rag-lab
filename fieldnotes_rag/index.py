"""TF-IDF + hashed feature hybrid index — fully offline, no external models."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .chunker import Chunk
from .tokenizer import tokenize, ngrams


@dataclass
class IndexedChunk:
    """Serialisable form of a Chunk stored in the index."""

    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    text: str
    start_line: int
    end_line: int
    metadata: Dict[str, str]
    tf_vector: Dict[str, float]  # normalised term frequencies


def _idf(n_docs: int, df: int) -> float:
    """Smoothed inverse document frequency."""
    return math.log((1 + n_docs) / (1 + df)) + 1.0


def _hash_features(tokens: List[str], n_buckets: int = 1024) -> Dict[int, float]:
    """Convert tokens to a sparse hashed feature vector (MurMur-like via MD5)."""
    vec: Dict[int, float] = {}
    for token in tokens:
        bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % n_buckets
        vec[bucket] = vec.get(bucket, 0.0) + 1.0
    # L2 normalise
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def _cosine_hashed(a: Dict[int, float], b: Dict[int, float]) -> float:
    """Cosine similarity between two hashed feature vectors."""
    dot = sum(a.get(k, 0.0) * v for k, v in b.items())
    norm_a = math.sqrt(sum(v * v for v in a.values())) or 1.0
    norm_b = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (norm_a * norm_b)


class HybridIndex:
    """
    Two-stage hybrid index:
      1. TF-IDF for lexical matching
      2. Hashed feature cosine for approximate semantic overlap
    Combined as: score = alpha * tfidf_score + (1 - alpha) * hashed_cosine
    """

    def __init__(self, alpha: float = 0.6, n_buckets: int = 1024) -> None:
        self.alpha = alpha
        self.n_buckets = n_buckets
        self._chunks: List[IndexedChunk] = []
        self._idf_map: Dict[str, float] = {}
        self._hashed_vecs: List[Dict[int, float]] = []

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def build(self, chunks: List[Chunk]) -> None:
        """Index a list of Chunk objects."""
        n_docs = len(chunks)
        df_map: Dict[str, int] = {}

        raw_tfs: List[Dict[str, float]] = []
        for chunk in chunks:
            tokens = tokenize(chunk.text, remove_stopwords=True)
            # Unigrams + bigrams
            all_tokens = tokens + ngrams(tokens, 2)
            counts: Dict[str, int] = {}
            for t in all_tokens:
                counts[t] = counts.get(t, 0) + 1
            total = len(all_tokens) or 1
            tf = {t: c / total for t, c in counts.items()}
            raw_tfs.append(tf)
            for t in counts:
                df_map[t] = df_map.get(t, 0) + 1

        # Compute IDF
        self._idf_map = {t: _idf(n_docs, df) for t, df in df_map.items()}

        # Build indexed chunks with TF-IDF vectors
        self._chunks = []
        self._hashed_vecs = []
        for i, chunk in enumerate(chunks):
            tf = raw_tfs[i]
            tf_idf = {t: v * self._idf_map.get(t, 1.0) for t, v in tf.items()}
            # L2-normalise
            norm = math.sqrt(sum(v * v for v in tf_idf.values())) or 1.0
            tf_idf_norm = {t: v / norm for t, v in tf_idf.items()}

            tokens = tokenize(chunk.text, remove_stopwords=True)
            hashed = _hash_features(tokens + ngrams(tokens, 2), self.n_buckets)

            self._chunks.append(
                IndexedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_path=chunk.source_path,
                    title=chunk.title,
                    text=chunk.text,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    metadata=chunk.metadata,
                    tf_vector=tf_idf_norm,
                )
            )
            self._hashed_vecs.append(hashed)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _tfidf_score(self, query_tokens: List[str], chunk_tf: Dict[str, float]) -> float:
        """Score a chunk against a query using dot product of TF-IDF vectors."""
        q_counts: Dict[str, int] = {}
        all_q = query_tokens + ngrams(query_tokens, 2)
        for t in all_q:
            q_counts[t] = q_counts.get(t, 0) + 1
        total = len(all_q) or 1

        score = 0.0
        for t, c in q_counts.items():
            q_weight = (c / total) * self._idf_map.get(t, 1.0)
            score += q_weight * chunk_tf.get(t, 0.0)
        return score

    def score(self, query: str, top_k: int = 5) -> List[Tuple[IndexedChunk, float]]:
        """Return top_k chunks with hybrid scores, sorted descending."""
        q_tokens = tokenize(query, remove_stopwords=True)
        if not q_tokens:
            return []

        q_hashed = _hash_features(q_tokens + ngrams(q_tokens, 2), self.n_buckets)

        scored: List[Tuple[IndexedChunk, float]] = []
        for chunk, h_vec in zip(self._chunks, self._hashed_vecs):
            tfidf = self._tfidf_score(q_tokens, chunk.tf_vector)
            cosine = _cosine_hashed(q_hashed, h_vec)
            hybrid = self.alpha * tfidf + (1.0 - self.alpha) * cosine
            scored.append((chunk, hybrid))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the index to a JSON file."""
        path = Path(path)
        data = {
            "alpha": self.alpha,
            "n_buckets": self.n_buckets,
            "idf_map": self._idf_map,
            "chunks": [asdict(c) for c in self._chunks],
            "hashed_vecs": [
                {str(k): v for k, v in h.items()} for h in self._hashed_vecs
            ],
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HybridIndex":
        """Deserialise an index from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        idx = cls(alpha=data["alpha"], n_buckets=data["n_buckets"])
        idx._idf_map = data["idf_map"]
        idx._chunks = [IndexedChunk(**c) for c in data["chunks"]]
        idx._hashed_vecs = [
            {int(k): v for k, v in h.items()} for h in data["hashed_vecs"]
        ]
        return idx

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def num_docs(self) -> int:
        return len({c.doc_id for c in self._chunks})

    @property
    def vocab_size(self) -> int:
        return len(self._idf_map)
