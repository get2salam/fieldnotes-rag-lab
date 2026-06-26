"""FieldNotes RAG Lab — local-first retrieval-augmented generation for field research notebooks."""

__version__ = "0.1.0"
__author__ = "Abdul Salam"

from .config import Config
from .loader import DocumentLoader
from .chunker import Chunker
from .index import HybridIndex
from .retriever import Retriever
from .synthesis import Synthesizer
from .evaluator import Evaluator
from .exceptions import FieldNotesError, CorpusError, ConfigError, ChunkingError
from .query_expansion import expand_query

__all__ = [
    "Config",
    "DocumentLoader",
    "Chunker",
    "HybridIndex",
    "Retriever",
    "Synthesizer",
    "Evaluator",
    # Exceptions
    "FieldNotesError",
    "CorpusError",
    "ConfigError",
    "ChunkingError",
    # Query expansion
    "expand_query",
]
