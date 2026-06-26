"""Citation model — maps retrieved chunks to human-readable source references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Citation:
    """A single source citation for a retrieved chunk."""

    ref_num: int
    doc_id: str
    title: str
    source_path: str
    start_line: int
    end_line: int
    score: float
    snippet: str

    @property
    def short_path(self) -> str:
        """Return a display-friendly relative-ish path."""
        p = Path(self.source_path)
        parts = p.parts
        # Show last 3 path parts for readability
        return "/".join(parts[-3:]) if len(parts) >= 3 else str(p)

    def format_inline(self) -> str:
        """Return [N] style inline citation marker."""
        return f"[{self.ref_num}]"

    def format_bibliography(self) -> str:
        """Return a full bibliography entry."""
        return (
            f"[{self.ref_num}] {self.title}\n"
            f"    {self.short_path} (lines {self.start_line}–{self.end_line})"
        )

    def format_short(self) -> str:
        """One-line citation."""
        return (
            f"[{self.ref_num}] {self.short_path}:{self.start_line}-{self.end_line}"
        )


def build_citations(
    ranked_items: List[tuple],  # List of (IndexedChunk, score)
    snippet_words: int = 25,
) -> List[Citation]:
    """Convert scored chunks into a numbered citation list."""
    citations: List[Citation] = []
    for ref_num, (chunk, score) in enumerate(ranked_items, start=1):
        words = chunk.text.split()
        snippet = " ".join(words[:snippet_words])
        if len(words) > snippet_words:
            snippet += "…"

        citations.append(
            Citation(
                ref_num=ref_num,
                doc_id=chunk.doc_id,
                title=chunk.title,
                source_path=chunk.source_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                score=round(score, 4),
                snippet=snippet,
            )
        )
    return citations


def format_sources_section(citations: List[Citation]) -> str:
    """Return a formatted sources block for appending to a synthesised answer."""
    if not citations:
        return ""
    lines = ["Sources:"]
    for c in citations:
        lines.append(c.format_bibliography())
        lines.append(f"    Relevance score: {c.score:.4f}")
        lines.append(f"    Excerpt: \"{c.snippet}\"")
    return "\n".join(lines)
