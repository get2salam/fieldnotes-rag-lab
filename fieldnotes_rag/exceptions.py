"""Custom exception types for FieldNotes RAG Lab."""

from __future__ import annotations


class FieldNotesError(Exception):
    """Base exception for the FieldNotes RAG Lab package."""


class CorpusError(FieldNotesError):
    """Raised when the corpus directory is empty or unreadable."""


class IndexError(FieldNotesError):
    """Raised when the index is missing, corrupt, or incompatible."""


class ConfigError(FieldNotesError):
    """Raised when configuration values are invalid."""


class ChunkingError(FieldNotesError):
    """Raised when chunking fails to produce any chunks."""
