"""Configuration dataclass for FieldNotes RAG Lab."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List


@dataclass
class Config:
    """Central configuration for the RAG pipeline."""

    # Chunking
    chunk_size: int = 200
    chunk_overlap: int = 40

    # Retrieval
    top_k: int = 5
    min_score: float = 0.01

    # Indexing
    max_features: int = 10_000
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    hybrid_alpha: float = 0.6  # weight for TF-IDF vs hashed-feature score

    # Corpus
    extensions: List[str] = field(default_factory=lambda: [".md", ".txt"])
    encoding: str = "utf-8"

    # Synthesis
    max_context_chunks: int = 5
    citation_style: str = "inline"  # "inline" or "footnote"

    # Paths
    index_path: str = ".fieldnotes-index.json"

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_env(cls) -> "Config":
        """Override config fields from environment variables prefixed FNRAG_."""
        instance = cls()
        for fname, ftype in {
            "chunk_size": int,
            "chunk_overlap": int,
            "top_k": int,
            "min_score": float,
            "max_features": int,
            "hybrid_alpha": float,
        }.items():
            env_key = f"FNRAG_{fname.upper()}"
            val = os.environ.get(env_key)
            if val is not None:
                setattr(instance, fname, ftype(val))
        return instance
