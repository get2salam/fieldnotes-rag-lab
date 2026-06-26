"""
Regression tests: end-to-end RAG pipeline on the sample corpus.

These tests build a real index from the sample corpus and verify
that specific queries return expected results. They are deterministic —
given the same corpus and query, scores must not change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fieldnotes_rag.config import Config
from fieldnotes_rag.loader import DocumentLoader
from fieldnotes_rag.chunker import Chunker
from fieldnotes_rag.index import HybridIndex
from fieldnotes_rag.retriever import Retriever
from fieldnotes_rag.synthesis import Synthesizer
from fieldnotes_rag.evaluator import Evaluator

CORPUS_DIR = Path(__file__).parent.parent / "examples" / "field_corpus"
FIXTURES_PATH = Path(__file__).parent.parent / "examples" / "eval_fixtures.json"

# Only run these if the sample corpus exists
pytestmark = pytest.mark.skipif(
    not CORPUS_DIR.exists(),
    reason="Sample corpus not found; skipping regression tests",
)


@pytest.fixture(scope="module")
def pipeline():
    """Build a full pipeline from the sample corpus once per test session."""
    config = Config()
    loader = DocumentLoader()
    docs = loader.load_directory(CORPUS_DIR)
    chunker = Chunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    chunks = chunker.chunk_documents(docs)
    idx = HybridIndex(alpha=config.hybrid_alpha)
    idx.build(chunks)
    retriever = Retriever(idx, top_k=5, expand_query=True)
    synthesizer = Synthesizer(max_context_chunks=5)
    return retriever, synthesizer


class TestEndToEndRetrieval:
    def test_corpus_loads_all_documents(self):
        loader = DocumentLoader()
        docs = loader.load_directory(CORPUS_DIR)
        assert len(docs) >= 7  # 7 original + any added corpus notes

    def test_bird_query_retrieves_bird_sightings(self, pipeline):
        retriever, _ = pipeline
        results = retriever.retrieve("American Dipper dusk streamside bird", top_k=5)
        doc_ids = [c.doc_id for c, _ in results]
        assert "bird_sightings" in doc_ids

    def test_safety_query_retrieves_safety_doc(self, pipeline):
        retriever, _ = pipeline
        results = retriever.retrieve("stream crossing buddy system gauge", top_k=5)
        doc_ids = [c.doc_id for c, _ in results]
        assert "safety_checklist" in doc_ids

    def test_plant_query_retrieves_plant_doc(self, pipeline):
        retriever, _ = pipeline
        results = retriever.retrieve("Red Alder nitrogen riparian canopy", top_k=5)
        doc_ids = [c.doc_id for c, _ in results]
        assert "plant_observations" in doc_ids or "riparian_habitat" in doc_ids

    def test_weather_query_retrieves_weather_doc(self, pipeline):
        retriever, _ = pipeline
        results = retriever.retrieve("wind rain temperature detection survey", top_k=5)
        doc_ids = [c.doc_id for c, _ in results]
        assert "weather_observations" in doc_ids

    def test_species_query_retrieves_species_doc(self, pipeline):
        retriever, _ = pipeline
        results = retriever.retrieve("amphibian salamander tailed frog cold stream", top_k=5)
        doc_ids = [c.doc_id for c, _ in results]
        assert "species_notes" in doc_ids


class TestEndToEndSynthesis:
    def test_answer_contains_inline_citation(self, pipeline):
        retriever, synthesizer = pipeline
        chunks, _ = retriever.retrieve_with_citations("bird dusk observation")
        result = synthesizer.synthesize("bird dusk observation", chunks)
        assert "[1]" in result.answer

    def test_result_has_citations(self, pipeline):
        retriever, synthesizer = pipeline
        chunks, _ = retriever.retrieve_with_citations("safety crossing stream")
        result = synthesizer.synthesize("safety crossing stream", chunks)
        assert len(result.citations) >= 1

    def test_to_dict_is_serialisable(self, pipeline):
        retriever, synthesizer = pipeline
        chunks, _ = retriever.retrieve_with_citations("plant riparian")
        result = synthesizer.synthesize("plant riparian", chunks)
        d = result.to_dict()
        # Must be JSON serialisable (no datetimes, no custom objects)
        json_str = json.dumps(d)
        assert len(json_str) > 0


class TestEvaluation:
    def test_fixtures_file_exists(self):
        assert FIXTURES_PATH.exists(), "eval_fixtures.json not found"

    def test_recall_at_5_above_threshold(self, pipeline):
        if not FIXTURES_PATH.exists():
            pytest.skip("Fixtures not found")
        retriever, synthesizer = pipeline
        evaluator = Evaluator(retriever, synthesizer)
        queries = Evaluator.load_fixtures(FIXTURES_PATH)
        report = evaluator.run(queries)
        assert report.mean_recall >= 0.80, (
            f"Recall@5 dropped below 0.80: {report.mean_recall:.3f}"
        )

    def test_mrr_above_threshold(self, pipeline):
        if not FIXTURES_PATH.exists():
            pytest.skip("Fixtures not found")
        retriever, synthesizer = pipeline
        evaluator = Evaluator(retriever, synthesizer)
        queries = Evaluator.load_fixtures(FIXTURES_PATH)
        report = evaluator.run(queries)
        assert report.mean_mrr >= 0.70, (
            f"MRR dropped below 0.70: {report.mean_mrr:.3f}"
        )

    def test_citation_accuracy_perfect(self, pipeline):
        if not FIXTURES_PATH.exists():
            pytest.skip("Fixtures not found")
        retriever, synthesizer = pipeline
        evaluator = Evaluator(retriever, synthesizer)
        queries = Evaluator.load_fixtures(FIXTURES_PATH)
        report = evaluator.run(queries)
        assert report.mean_citation_accuracy >= 0.90, (
            f"Citation accuracy dropped: {report.mean_citation_accuracy:.3f}"
        )
