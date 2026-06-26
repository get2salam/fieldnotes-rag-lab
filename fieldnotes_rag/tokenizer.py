"""Lightweight tokenizer using only stdlib — no NLTK, no spaCy."""

from __future__ import annotations

import re
from typing import List, Set

_PUNCTUATION_RE = re.compile(r"[^\w\s'-]")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "be", "are", "was",
    "were", "been", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "this", "that",
    "these", "those", "i", "we", "you", "he", "she", "they", "me", "us",
    "him", "her", "them", "my", "our", "your", "his", "their", "what",
    "which", "who", "when", "where", "how", "why", "if", "then", "than",
    "not", "no", "nor", "as", "so", "yet", "both", "either", "neither",
    "each", "every", "all", "any", "few", "more", "most", "other", "some",
    "such", "also", "very", "just", "about", "above", "after", "before",
    "between", "into", "through", "during", "while", "within", "without",
    "can", "any", "only", "over", "under", "again", "further", "once",
    "here", "there", "same", "too", "s", "t", "don", "re", "ve", "ll",
}


def normalize(text: str) -> str:
    """Lower-case, strip punctuation (preserve hyphens and apostrophes), collapse whitespace."""
    text = text.lower()
    text = _PUNCTUATION_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Return a list of cleaned tokens from text."""
    tokens = normalize(text).split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using a simple boundary pattern."""
    sentences = _SENTENCE_BOUNDARY_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def ngrams(tokens: List[str], n: int) -> List[str]:
    """Generate n-gram strings from a token list."""
    if n < 1 or len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def term_frequency(tokens: List[str]) -> dict[str, float]:
    """Return relative term frequencies (count / total)."""
    if not tokens:
        return {}
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens)
    return {t: c / total for t, c in counts.items()}
