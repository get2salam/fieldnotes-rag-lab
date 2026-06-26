"""Tests for the document loader."""

import tempfile
from pathlib import Path

import pytest

from fieldnotes_rag.loader import Document, DocumentLoader, _extract_title, _strip_front_matter


class TestStripFrontMatter:
    def test_strips_yaml_front_matter(self):
        text = "---\ntitle: My Note\nauthor: Alice\n---\nBody content here."
        body, meta = _strip_front_matter(text)
        assert "Body content here." in body
        assert meta.get("title") == "My Note"
        assert meta.get("author") == "Alice"

    def test_no_front_matter_unchanged(self):
        text = "# Heading\nJust plain content."
        body, meta = _strip_front_matter(text)
        assert body == text
        assert meta == {}

    def test_partial_front_matter_not_stripped(self):
        text = "---\ntitle: Test\nBody without closing."
        body, meta = _strip_front_matter(text)
        assert body == text  # no closing ---


class TestExtractTitle:
    def test_extracts_h1(self):
        text = "# My Field Notes\nSome content."
        title = _extract_title(text, Path("anything.md"))
        assert title == "My Field Notes"

    def test_falls_back_to_filename(self):
        text = "No headings here, just text."
        title = _extract_title(text, Path("bird_sightings.md"))
        assert "Bird" in title or "bird" in title.lower()

    def test_extracts_h2_if_no_h1(self):
        text = "## Secondary Heading\nContent."
        title = _extract_title(text, Path("x.md"))
        assert title == "Secondary Heading"


class TestDocumentLoader:
    def setup_method(self):
        self.loader = DocumentLoader()

    def test_load_file_markdown(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test Title\nSome field note content.")
        doc = self.loader.load_file(md_file)
        assert doc is not None
        assert doc.title == "Test Title"
        assert "field note" in doc.content

    def test_load_file_txt(self, tmp_path):
        txt_file = tmp_path / "note.txt"
        txt_file.write_text("Observation note content.")
        doc = self.loader.load_file(txt_file)
        assert doc is not None
        assert "Observation" in doc.content

    def test_skips_non_matching_extension(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2")
        doc = self.loader.load_file(csv_file)
        assert doc is None

    def test_load_directory(self, tmp_path):
        (tmp_path / "a.md").write_text("# Note A\nContent A.")
        (tmp_path / "b.md").write_text("# Note B\nContent B.")
        (tmp_path / "skip.csv").write_text("not a note")
        docs = self.loader.load_directory(tmp_path)
        assert len(docs) == 2
        titles = {d.title for d in docs}
        assert "Note A" in titles
        assert "Note B" in titles

    def test_load_directory_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.md").write_text("# Root\nTop-level.")
        (sub / "child.md").write_text("# Child\nNested.")
        docs = self.loader.load_directory(tmp_path)
        assert len(docs) == 2

    def test_doc_has_metadata(self, tmp_path):
        md_file = tmp_path / "meta.md"
        md_file.write_text("# Title\n**Location:** Ridgeline Creek\nContent.")
        doc = self.loader.load_file(md_file)
        assert doc is not None
        assert "Location" in doc.metadata

    def test_load_nonexistent_directory_raises(self):
        with pytest.raises(ValueError):
            self.loader.load_directory("/definitely/does/not/exist")
