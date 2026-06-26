"""Sentence-aware sliding-window chunker that preserves source metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .loader import Document
from .tokenizer import split_sentences


@dataclass
class Chunk:
    """A text chunk derived from a source Document."""

    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    text: str
    start_line: int
    end_line: int
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.source_path} (lines {self.start_line}–{self.end_line})"

    def __repr__(self) -> str:
        return (
            f"Chunk(id={self.chunk_id!r}, doc={self.doc_id!r}, "
            f"lines={self.start_line}-{self.end_line}, chars={len(self.text)})"
        )


class Chunker:
    """Split a Document into overlapping text chunks with line-number tracking."""

    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
        min_chunk_words: int = 10,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_words = min_chunk_words

    def _line_offsets(self, text: str) -> List[int]:
        """Return the character offset at the start of each line."""
        offsets = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                offsets.append(i + 1)
        return offsets

    def _char_to_line(self, char_idx: int, offsets: List[int]) -> int:
        """Binary-search the line offsets to find the 1-based line number."""
        lo, hi = 0, len(offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offsets[mid] <= char_idx:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1  # 1-based

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Produce overlapping chunks from a Document."""
        text = doc.content
        offsets = self._line_offsets(text)
        sentences = split_sentences(text)

        chunks: List[Chunk] = []
        chunk_idx = 0
        sent_cursor = 0

        while sent_cursor < len(sentences):
            # Build a chunk by accumulating sentences up to chunk_size words
            window: List[str] = []
            word_count = 0
            i = sent_cursor

            while i < len(sentences) and word_count < self.chunk_size:
                s = sentences[i]
                words = s.split()
                window.append(s)
                word_count += len(words)
                i += 1

            if not window:
                break

            chunk_text = " ".join(window)
            word_list = chunk_text.split()

            if len(word_list) < self.min_chunk_words:
                sent_cursor += 1
                continue

            # Locate character positions in original text for line-number mapping
            # Search for the start of the first sentence's text
            start_sent = window[0]
            char_start = text.find(start_sent)
            if char_start < 0:
                char_start = 0
            char_end = min(char_start + len(chunk_text), len(text) - 1)

            start_line = self._char_to_line(char_start, offsets)
            end_line = self._char_to_line(char_end, offsets)

            chunk = Chunk(
                chunk_id=f"{doc.doc_id}_c{chunk_idx:04d}",
                doc_id=doc.doc_id,
                source_path=doc.path,
                title=doc.title,
                text=chunk_text,
                start_line=start_line,
                end_line=end_line,
                metadata=dict(doc.metadata),
            )
            chunks.append(chunk)
            chunk_idx += 1

            # Advance cursor by (chunk_size - overlap) sentences worth of words
            advance_words = max(1, self.chunk_size - self.chunk_overlap)
            words_skipped = 0
            while sent_cursor < len(sentences) and words_skipped < advance_words:
                words_skipped += len(sentences[sent_cursor].split())
                sent_cursor += 1

        return chunks

    def chunk_documents(self, docs: List[Document]) -> List[Chunk]:
        """Chunk a list of documents, returning a flat list of Chunks."""
        all_chunks: List[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
