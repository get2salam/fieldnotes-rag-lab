"""Shared type aliases and data models for the RAG pipeline."""

from __future__ import annotations

from typing import List, Tuple

from .index import IndexedChunk
from .citation import Citation

# Typed aliases used across the pipeline
ScoredChunk = Tuple[IndexedChunk, float]
RankedResults = List[ScoredChunk]
CitationList = List[Citation]
