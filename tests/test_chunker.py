"""Tests for the document chunker."""

import pytest

from fieldnotes_rag.chunker import Chunk, Chunker
from fieldnotes_rag.loader import Document


def _make_doc(content: str, doc_id: str = "test_doc") -> Document:
    return Document(
        doc_id=doc_id,
        path=f"/corpus/{doc_id}.md",
        title="Test Document",
        content=content,
    )


class TestChunker:
    def setup_method(self):
        self.chunker = Chunker(chunk_size=50, chunk_overlap=10)

    def test_basic_chunking(self):
        doc = _make_doc(
            "The American Dipper forages in fast-moving riffles. "
            "It is the only North American songbird that walks underwater. "
            "It bobs its tail continuously, which is a distinctive behaviour. "
            "Observers should note the habitat type and water velocity. "
            "Record species, sex, age, and behaviour at time of observation."
        )
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_ids_unique(self):
        doc = _make_doc(
            " ".join([f"Sentence number {i} with some field note content." for i in range(30)])
        )
        chunks = self.chunker.chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_contains_text(self):
        doc = _make_doc("The osprey circles high above the pool. It dives to catch fish.")
        chunks = self.chunker.chunk_document(doc)
        combined = " ".join(c.text for c in chunks)
        assert "osprey" in combined.lower()

    def test_chunk_has_doc_reference(self):
        doc = _make_doc("Bird observation at dawn near the riffle habitat.", doc_id="birds")
        chunks = self.chunker.chunk_document(doc)
        for c in chunks:
            assert c.doc_id == "birds"
            assert c.source_path == "/corpus/birds.md"

    def test_line_numbers_positive(self):
        doc = _make_doc("Line one.\nLine two.\nLine three.\nLine four.\nLine five.")
        chunks = self.chunker.chunk_document(doc)
        for c in chunks:
            assert c.start_line >= 1
            assert c.end_line >= c.start_line

    def test_chunk_documents_multiple(self):
        docs = [
            _make_doc("Content for document A with birds and riparian habitat.", "doc_a"),
            _make_doc("Content for document B with plants and weather observations.", "doc_b"),
        ]
        chunks = self.chunker.chunk_documents(docs)
        doc_ids = {c.doc_id for c in chunks}
        assert "doc_a" in doc_ids
        assert "doc_b" in doc_ids

    def test_citation_format(self):
        doc = _make_doc("Field observation of Cinclus mexicanus at riffle.")
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) >= 1
        citation = chunks[0].citation
        assert "test_doc" in citation or ".md" in citation
        assert "lines" in citation.lower()

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            Chunker(chunk_size=50, chunk_overlap=50)

    def test_empty_document(self):
        doc = _make_doc("")
        chunks = self.chunker.chunk_document(doc)
        # Empty doc should produce no chunks
        assert chunks == []
