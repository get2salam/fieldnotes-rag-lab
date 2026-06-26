"""Tests for answer synthesis and evaluation."""

import json
import tempfile
from pathlib import Path

import pytest

from fieldnotes_rag.chunker import Chunk
from fieldnotes_rag.evaluator import (
    EvalQuery,
    EvalReport,
    EvalResult,
    Evaluator,
    _recall_at_k,
    _mrr,
    _answer_coverage,
)
from fieldnotes_rag.index import HybridIndex
from fieldnotes_rag.retriever import Retriever
from fieldnotes_rag.synthesis import Synthesizer


def _chunk(chunk_id, doc_id, text):
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_path=f"/corpus/{doc_id}.md",
        title=doc_id.title(),
        text=text,
        start_line=1,
        end_line=15,
    )


CORPUS = [
    _chunk("b0", "birds", (
        "The American Dipper forages in fast riffles. "
        "It submerges into cold mountain streams to catch invertebrates. "
        "At dusk, note the light phase and bird microhabitat carefully."
    )),
    _chunk("p0", "plants", (
        "Red Alder dominates the riparian canopy and fixes atmospheric nitrogen. "
        "Native willows stabilise the bank and support warbler nesting habitat. "
        "Salmonberry flowers in early spring, attracting rufous hummingbirds."
    )),
    _chunk("s0", "safety", (
        "Stream crossing requires buddy system and loose hip belt. "
        "Check gauge reading before crossing; do not cross above 0.40 m. "
        "Emergency equipment stored at trailhead cache box and R-5 shelter."
    )),
    _chunk("w0", "weather", (
        "Wind above 20 kph reduces auditory detection probability. "
        "Heavy rain suppresses passerine foraging activity entirely. "
        "Record sky cover, wind class, and temperature at each survey point."
    )),
]


def _build_pipeline():
    idx = HybridIndex()
    idx.build(CORPUS)
    retriever = Retriever(idx, top_k=4)
    synthesizer = Synthesizer(max_context_chunks=4)
    return retriever, synthesizer


class TestRecallMetric:
    def test_full_recall(self):
        assert _recall_at_k(["a", "b"], ["a", "b", "c"]) == 1.0

    def test_partial_recall(self):
        assert _recall_at_k(["a", "b"], ["a", "c"]) == 0.5

    def test_zero_recall(self):
        assert _recall_at_k(["a"], ["b", "c"]) == 0.0

    def test_empty_expected(self):
        assert _recall_at_k([], ["a", "b"]) == 1.0


class TestMRRMetric:
    def test_first_hit(self):
        assert _mrr(["a"], ["a", "b", "c"]) == 1.0

    def test_second_hit(self):
        assert _mrr(["b"], ["a", "b", "c"]) == 0.5

    def test_third_hit(self):
        assert abs(_mrr(["c"], ["a", "b", "c"]) - 1 / 3) < 1e-9

    def test_no_hit(self):
        assert _mrr(["z"], ["a", "b", "c"]) == 0.0


class TestAnswerCoverage:
    def test_full_coverage(self):
        answer = "The bird was observed foraging in the stream at dusk."
        assert _answer_coverage(answer, ["bird", "foraging", "stream", "dusk"]) == 1.0

    def test_partial_coverage(self):
        answer = "The bird was foraging."
        score = _answer_coverage(answer, ["bird", "foraging", "stream"])
        assert abs(score - 2 / 3) < 1e-9

    def test_empty_keywords(self):
        assert _answer_coverage("anything", []) == 1.0

    def test_case_insensitive(self):
        score = _answer_coverage("The Bird was seen.", ["bird"])
        assert score == 1.0


class TestEvaluator:
    def setup_method(self):
        self.retriever, self.synthesizer = _build_pipeline()
        self.evaluator = Evaluator(self.retriever, self.synthesizer)

    def test_evaluate_single_query(self):
        eq = EvalQuery(
            query_id="q1",
            query="stream crossing safety buddy system",
            expected_doc_ids=["safety"],
            expected_keywords=["stream", "crossing"],
        )
        result = self.evaluator.evaluate_query(eq)
        assert isinstance(result, EvalResult)
        assert result.query_id == "q1"
        assert 0.0 <= result.recall_at_k <= 1.0
        assert 0.0 <= result.mrr <= 1.0

    def test_safety_query_retrieves_safety_doc(self):
        eq = EvalQuery(
            query_id="q_safety",
            query="stream crossing gauge buddy system emergency",
            expected_doc_ids=["safety"],
            expected_keywords=["safety"],
        )
        result = self.evaluator.evaluate_query(eq)
        assert result.recall_at_k == 1.0

    def test_run_report(self):
        queries = [
            EvalQuery("q1", "bird dipper stream", ["birds"], ["dipper"]),
            EvalQuery("q2", "plant riparian canopy alder", ["plants"], ["riparian"]),
            EvalQuery("q3", "crossing safety hazard", ["safety"], ["crossing"]),
        ]
        report = self.evaluator.run(queries)
        assert len(report.results) == 3
        assert 0.0 <= report.mean_recall <= 1.0

    def test_report_format(self):
        queries = [EvalQuery("q1", "bird dipper", ["birds"], ["bird"])]
        report = self.evaluator.run(queries)
        text = report.format()
        assert "Recall" in text
        assert "MRR" in text
        assert "q1" in text

    def test_load_fixtures(self, tmp_path):
        fixtures = [
            {
                "query_id": "f1",
                "query": "bird dusk observation",
                "expected_doc_ids": ["birds"],
                "expected_keywords": ["bird", "dusk"],
                "min_citations": 1,
            }
        ]
        path = tmp_path / "fixtures.json"
        path.write_text(json.dumps(fixtures), encoding="utf-8")
        loaded = Evaluator.load_fixtures(path)
        assert len(loaded) == 1
        assert loaded[0].query_id == "f1"
        assert loaded[0].expected_doc_ids == ["birds"]

    def test_report_to_dict(self):
        queries = [EvalQuery("q1", "weather wind", ["weather"], ["wind"])]
        report = self.evaluator.run(queries)
        d = report.to_dict()
        assert "summary" in d
        assert "per_query" in d
        assert d["summary"]["num_queries"] == 1
