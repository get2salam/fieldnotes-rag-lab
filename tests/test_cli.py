"""Tests for the CLI argument parser and command dispatch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from fieldnotes_rag.cli import build_parser, main


class TestParser:
    def setup_method(self):
        self.parser = build_parser()

    def test_ingest_subcommand(self):
        args = self.parser.parse_args(["ingest", "some/dir"])
        assert args.command == "ingest"
        assert args.corpus_dir == "some/dir"

    def test_ingest_with_index_flag(self):
        args = self.parser.parse_args(["ingest", "some/dir", "--index", "my.json"])
        assert args.index == "my.json"

    def test_ask_subcommand(self):
        args = self.parser.parse_args(["ask", "What are riparian birds?"])
        assert args.command == "ask"
        assert args.query == "What are riparian birds?"

    def test_ask_with_top_k(self):
        args = self.parser.parse_args(["ask", "query", "--top-k", "3"])
        assert args.top_k == 3

    def test_ask_with_json_flag(self):
        args = self.parser.parse_args(["ask", "query", "--json"])
        assert args.json is True

    def test_ask_no_expand_flag(self):
        args = self.parser.parse_args(["ask", "query", "--no-expand"])
        assert args.no_expand is True

    def test_eval_subcommand(self):
        args = self.parser.parse_args(["eval", "--fixtures", "fixtures.json"])
        assert args.command == "eval"
        assert args.fixtures == "fixtures.json"

    def test_stats_subcommand(self):
        args = self.parser.parse_args(["stats"])
        assert args.command == "stats"

    def test_export_subcommand(self):
        args = self.parser.parse_args(["export", "--output", "out.json"])
        assert args.output == "out.json"

    def test_no_command_exits_0(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0


class TestCLIIntegration:
    """Integration tests using the real sample corpus."""

    CORPUS = str(Path(__file__).parent.parent / "examples" / "field_corpus")
    FIXTURES = str(Path(__file__).parent.parent / "examples" / "eval_fixtures.json")

    @pytest.fixture
    def index_file(self, tmp_path):
        idx_path = str(tmp_path / "test.json")
        main(["ingest", self.CORPUS, "--index", idx_path])
        return idx_path

    def test_ingest_creates_index(self, tmp_path):
        idx_path = str(tmp_path / "test.json")
        main(["ingest", self.CORPUS, "--index", idx_path])
        assert Path(idx_path).exists()

    def test_ask_returns_answer(self, index_file, capsys):
        main(["ask", "What birds live near streams?", "--index", index_file])
        out = capsys.readouterr().out
        assert "Query:" in out
        assert "Answer:" in out
        assert "Sources:" in out

    def test_ask_json_output(self, index_file, capsys):
        main(["ask", "bird observation", "--index", index_file, "--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "query" in data
        assert "answer" in data
        assert "citations" in data

    def test_stats_output(self, index_file, capsys):
        main(["stats", "--index", index_file])
        out = capsys.readouterr().out
        assert "Documents:" in out
        assert "Chunks:" in out

    def test_eval_output(self, index_file, capsys):
        if not Path(self.FIXTURES).exists():
            pytest.skip("Fixtures not available")
        main(["eval", "--index", index_file, "--fixtures", self.FIXTURES])
        out = capsys.readouterr().out
        assert "Recall" in out

    def test_export_creates_json(self, index_file, tmp_path):
        out_path = str(tmp_path / "export.json")
        main(["export", "--index", index_file, "--output", out_path])
        assert Path(out_path).exists()
        data = json.loads(Path(out_path).read_text())
        assert isinstance(data, list)
        assert len(data) > 0
        assert "chunk_id" in data[0]
        assert "text" in data[0]

    def test_ask_missing_index_exits(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            main(["ask", "query", "--index", str(tmp_path / "missing.json")])
        assert exc_info.value.code != 0
