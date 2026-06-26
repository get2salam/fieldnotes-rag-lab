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

__all__ = [
    "Config",
    "DocumentLoader",
    "Chunker",
    "HybridIndex",
    "Retriever",
    "Synthesizer",
    "Evaluator",
]
