"""Utility functions used across the FieldNotes RAG Lab package."""

from __future__ import annotations

import re
import unicodedata
from typing import List


def slugify(text: str) -> str:
    """Convert a string to a filesystem-safe slug."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")


def truncate(text: str, max_chars: int, ellipsis: str = "…") -> str:
    """Truncate text to max_chars, adding ellipsis if truncated."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(ellipsis)] + ellipsis


def word_count(text: str) -> int:
    """Return the number of whitespace-delimited tokens in text."""
    return len(text.split())


def sentence_count(text: str) -> int:
    """Approximate the number of sentences in text."""
    return max(1, len(re.findall(r"[.!?](?:\s|$)", text)))


def format_score(score: float, precision: int = 4) -> str:
    """Format a relevance score as a percentage string."""
    return f"{score * 100:.{precision - 2}f}%"


def chunk_list(items: List, size: int) -> List[List]:
    """Split a list into sublists of at most `size` items."""
    return [items[i : i + size] for i in range(0, len(items), size)]
