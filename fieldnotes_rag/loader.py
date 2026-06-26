"""Document loader for Markdown and plain-text field notes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional


@dataclass
class Document:
    """A loaded source document with metadata."""

    doc_id: str
    path: str
    title: str
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def num_chars(self) -> int:
        return len(self.content)

    def __repr__(self) -> str:
        return f"Document(id={self.doc_id!r}, title={self.title!r}, chars={self.num_chars})"


_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)", re.MULTILINE)
_METADATA_LINE_RE = re.compile(r"^\*\*(.+?)\*\*[:\s]+(.+)$", re.MULTILINE)


def _strip_front_matter(text: str) -> tuple[str, Dict[str, str]]:
    """Remove YAML front matter and return (body, metadata_dict)."""
    meta: Dict[str, str] = {}
    match = _FRONT_MATTER_RE.match(text)
    if match:
        for line in match.group(1).splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                meta[k.strip()] = v.strip()
        text = text[match.end():]
    return text, meta


def _extract_title(text: str, path: Path) -> str:
    """Return the first H1 heading, or the filename stem if none found."""
    match = _HEADING_RE.search(text)
    if match:
        return match.group(1).strip()
    return path.stem.replace("_", " ").replace("-", " ").title()


def _extract_inline_metadata(text: str) -> Dict[str, str]:
    """Extract bold-label metadata lines like **Location:** Ridgeline Creek."""
    result: Dict[str, str] = {}
    for match in _METADATA_LINE_RE.finditer(text):
        result[match.group(1).strip()] = match.group(2).strip()
    return result


class DocumentLoader:
    """Load .md and .txt files from a directory into Document objects."""

    def __init__(
        self,
        extensions: Optional[List[str]] = None,
        encoding: str = "utf-8",
    ) -> None:
        self.extensions = set(extensions or [".md", ".txt"])
        self.encoding = encoding

    def load_file(self, path: Path) -> Optional[Document]:
        """Load a single file and return a Document, or None if unreadable."""
        if path.suffix.lower() not in self.extensions:
            return None
        try:
            raw = path.read_text(encoding=self.encoding, errors="replace")
        except OSError:
            return None

        body, front_meta = _strip_front_matter(raw)
        inline_meta = _extract_inline_metadata(body)
        metadata = {**front_meta, **inline_meta}
        title = metadata.get("title", _extract_title(body, path))

        return Document(
            doc_id=path.stem,
            path=str(path),
            title=title,
            content=body,
            metadata=metadata,
        )

    def load_directory(self, directory: str | Path) -> List[Document]:
        """Recursively load all matching files from a directory."""
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        docs: List[Document] = []
        for p in sorted(directory.rglob("*")):
            if p.is_file() and p.suffix.lower() in self.extensions:
                doc = self.load_file(p)
                if doc is not None:
                    docs.append(doc)
        return docs

    def iter_files(self, directory: str | Path) -> Iterator[Document]:
        """Yield documents one at a time for memory-efficient processing."""
        directory = Path(directory)
        for p in sorted(directory.rglob("*")):
            if p.is_file() and p.suffix.lower() in self.extensions:
                doc = self.load_file(p)
                if doc is not None:
                    yield doc
