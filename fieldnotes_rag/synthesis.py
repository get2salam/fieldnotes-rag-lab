"""Answer synthesis — builds cited answers from retrieved chunks."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import List, Tuple

from .index import IndexedChunk
from .citation import Citation, build_citations, format_sources_section


@dataclass
class SynthesisResult:
    """Holds a synthesised answer and its supporting citations."""

    query: str
    answer: str
    citations: List[Citation]
    num_chunks_used: int

    def format(self, width: int = 80) -> str:
        """Return a human-readable formatted result."""
        lines = [
            f"Query: {self.query}",
            "",
            "Answer:",
            textwrap.fill(self.answer, width=width),
            "",
            format_sources_section(self.citations),
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "num_chunks_used": self.num_chunks_used,
            "citations": [
                {
                    "ref": c.ref_num,
                    "title": c.title,
                    "source": c.source_path,
                    "lines": f"{c.start_line}-{c.end_line}",
                    "score": c.score,
                    "snippet": c.snippet,
                }
                for c in self.citations
            ],
        }


class Synthesizer:
    """
    Template-driven answer synthesizer.

    Constructs answers by:
    1. Combining key sentences from retrieved chunks
    2. Inserting inline citation markers [1], [2]…
    3. Appending a bibliography

    This approach is fully deterministic and requires no external LLM.
    """

    def __init__(
        self,
        max_context_chunks: int = 5,
        sentences_per_chunk: int = 3,
        wrap_width: int = 80,
    ) -> None:
        self.max_context_chunks = max_context_chunks
        self.sentences_per_chunk = sentences_per_chunk
        self.wrap_width = wrap_width

    def _extract_key_sentences(self, text: str, n: int) -> List[str]:
        """Return up to n sentences from the start of a chunk."""
        # Split on ". ", "! ", "? " boundaries (simple heuristic)
        buf: List[str] = []
        current: List[str] = []
        for char in text:
            current.append(char)
            if char in ".!?" and len(current) > 2:
                sentence = "".join(current).strip()
                if sentence:
                    buf.append(sentence)
                current = []
                if len(buf) >= n:
                    break
        # If remaining chars form a sentence, add them
        if current and len(buf) < n:
            leftover = "".join(current).strip()
            if leftover:
                buf.append(leftover)
        return buf[:n]

    def synthesize(
        self,
        query: str,
        scored_chunks: List[Tuple[IndexedChunk, float]],
    ) -> SynthesisResult:
        """Construct a cited answer from the top-k scored chunks."""
        chunks_to_use = scored_chunks[: self.max_context_chunks]
        citations = build_citations(chunks_to_use)

        if not chunks_to_use:
            return SynthesisResult(
                query=query,
                answer=(
                    "No relevant passages were found in the field notes corpus for "
                    "this query. Try rephrasing or ingesting additional notes."
                ),
                citations=[],
                num_chunks_used=0,
            )

        # Build answer body by stitching key sentences + citation markers
        answer_parts: List[str] = []
        for idx, (chunk, _score) in enumerate(chunks_to_use):
            ref_num = idx + 1
            key_sents = self._extract_key_sentences(chunk.text, self.sentences_per_chunk)
            if key_sents:
                passage = " ".join(key_sents)
                answer_parts.append(f"{passage} [{ref_num}]")

        answer = " ".join(answer_parts)

        # Clean up whitespace and double spaces
        answer = " ".join(answer.split())

        return SynthesisResult(
            query=query,
            answer=answer,
            citations=citations,
            num_chunks_used=len(chunks_to_use),
        )
