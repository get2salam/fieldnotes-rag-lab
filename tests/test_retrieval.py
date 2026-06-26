"""Tests for retrieval and synthesis pipeline."""

import pytest

from fieldnotes_rag.chunker import Chunk
from fieldnotes_rag.index import HybridIndex
from fieldnotes_rag.retriever import Retriever, _expand_query, _deduplicate
from fieldnotes_rag.synthesis import Synthesizer, SynthesisResult
from fieldnotes_rag.citation import Citation, build_citations, format_sources_section


def _chunk(chunk_id, doc_id, text):
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_path=f"/corpus/{doc_id}.md",
        title=doc_id.title(),
        text=text,
        start_line=1,
        end_line=10,
    )


def _build_index(*chunks):
    idx = HybridIndex()
    idx.build(list(chunks))
    return idx


BIRD_CHUNK = _chunk("b0", "birds", (
    "The American Dipper forages in fast riffles by walking underwater. "
    "It bobs its tail continuously and is found year-round near cold streams. "
    "When recording a dusk observation, note light phase, behaviour, and exact microhabitat."
))
PLANT_CHUNK = _chunk("p0", "plants", (
    "Red Alder dominates the riparian canopy. "
    "Its leaf litter provides allochthonous energy inputs to the stream. "
    "Native willows stabilise eroded banks and support nesting warblers."
))
SAFETY_CHUNK = _chunk("s0", "safety", (
    "Stream crossings require the buddy system and a loose hip belt before wading. "
    "Do not cross when gauge reading exceeds 0.40 m above baseline. "
    "Emergency equipment is cached at the trailhead and at R-5 shelter."
))
WEATHER_CHUNK = _chunk("w0", "weather", (
    "Wind above 20 kph reduces auditory detection probability in bird surveys. "
    "Heavy rain suppresses passerine activity. "
    "Monitor the R-4 gauge reading after any precipitation event above 15 mm."
))


class TestQueryExpansion:
    def test_stream_expands(self):
        expanded = _expand_query("stream crossing")
        assert "creek" in expanded or "riffle" in expanded or "river" in expanded

    def test_bird_expands(self):
        expanded = _expand_query("bird observation")
        assert "avian" in expanded or "passerine" in expanded

    def test_unknown_word_unchanged(self):
        original = "xyq_very_unusual_query"
        expanded = _expand_query(original)
        assert original in expanded


class TestDeduplicate:
    def test_limits_per_doc(self):
        idx = _build_index(
            _chunk("a1", "doc_a", "Alpha content one with some detail."),
            _chunk("a2", "doc_a", "Alpha content two with more detail."),
            _chunk("a3", "doc_a", "Alpha content three additional detail."),
            _chunk("b1", "doc_b", "Beta content with different words here."),
        )
        results = idx.score("content detail", top_k=10)
        deduped = _deduplicate(results, max_per_doc=1)
        doc_ids = [c.doc_id for c, _ in deduped]
        assert doc_ids.count("doc_a") <= 1

    def test_allows_multiple_per_doc(self):
        idx = _build_index(
            _chunk("a1", "doc_a", "Alpha content one with some detail."),
            _chunk("a2", "doc_a", "Alpha content two with more detail."),
            _chunk("b1", "doc_b", "Beta content different words here."),
        )
        results = idx.score("content detail", top_k=10)
        deduped = _deduplicate(results, max_per_doc=2)
        doc_ids = [c.doc_id for c, _ in deduped]
        assert doc_ids.count("doc_a") <= 2


class TestRetriever:
    def setup_method(self):
        self.idx = _build_index(BIRD_CHUNK, PLANT_CHUNK, SAFETY_CHUNK, WEATHER_CHUNK)
        self.retriever = Retriever(self.idx, top_k=3, expand_query=True)

    def test_retrieve_returns_results(self):
        results = self.retriever.retrieve("bird dusk observation")
        assert len(results) >= 1

    def test_retrieve_safety_query(self):
        results = self.retriever.retrieve("stream crossing buddy system")
        doc_ids = [c.doc_id for c, _ in results]
        assert "safety" in doc_ids

    def test_retrieve_with_citations(self):
        chunks, citations = self.retriever.retrieve_with_citations("weather wind rain survey")
        assert len(citations) >= 1
        assert all(isinstance(c, Citation) for c in citations)

    def test_retrieve_respects_top_k(self):
        results = self.retriever.retrieve("content", top_k=2)
        assert len(results) <= 2

    def test_min_score_filters(self):
        retriever = Retriever(self.idx, top_k=5, min_score=999.0)
        results = retriever.retrieve("bird")
        assert results == []

    def test_citation_ref_numbers(self):
        _, citations = self.retriever.retrieve_with_citations("dipper riffle stream")
        for i, c in enumerate(citations, start=1):
            assert c.ref_num == i


class TestSynthesizer:
    def setup_method(self):
        self.idx = _build_index(BIRD_CHUNK, PLANT_CHUNK, SAFETY_CHUNK, WEATHER_CHUNK)
        self.retriever = Retriever(self.idx, top_k=3)
        self.synthesizer = Synthesizer(max_context_chunks=3)

    def test_synthesize_returns_result(self):
        chunks, _ = self.retriever.retrieve_with_citations("dipper observation dusk")
        result = self.synthesizer.synthesize("dipper observation dusk", chunks)
        assert isinstance(result, SynthesisResult)
        assert result.query == "dipper observation dusk"

    def test_synthesize_answer_nonempty(self):
        chunks, _ = self.retriever.retrieve_with_citations("bird stream")
        result = self.synthesizer.synthesize("bird stream", chunks)
        assert len(result.answer) > 20

    def test_synthesize_contains_citations(self):
        chunks, _ = self.retriever.retrieve_with_citations("safety crossing")
        result = self.synthesizer.synthesize("safety crossing", chunks)
        assert "[1]" in result.answer

    def test_synthesize_no_chunks_returns_fallback(self):
        result = self.synthesizer.synthesize("totally unknown query", [])
        assert "No relevant" in result.answer or len(result.answer) > 0

    def test_format_output(self):
        chunks, _ = self.retriever.retrieve_with_citations("bird observation")
        result = self.synthesizer.synthesize("bird observation", chunks)
        formatted = result.format()
        assert "Query:" in formatted
        assert "Answer:" in formatted
        assert "Sources:" in formatted

    def test_to_dict(self):
        chunks, _ = self.retriever.retrieve_with_citations("plant riparian")
        result = self.synthesizer.synthesize("plant riparian", chunks)
        d = result.to_dict()
        assert "query" in d
        assert "answer" in d
        assert "citations" in d
        assert isinstance(d["citations"], list)


class TestCitations:
    def test_build_citations_numbering(self):
        idx = _build_index(BIRD_CHUNK, PLANT_CHUNK)
        results = idx.score("bird dipper", top_k=2)
        citations = build_citations(results)
        for i, c in enumerate(citations, start=1):
            assert c.ref_num == i

    def test_citation_snippet_truncated(self):
        idx = _build_index(BIRD_CHUNK)
        results = idx.score("bird", top_k=1)
        citations = build_citations(results, snippet_words=5)
        assert citations[0].snippet.count(" ") <= 6  # 5 words ≈ 4–5 spaces

    def test_citation_format_bibliography(self):
        citation = Citation(
            ref_num=1,
            doc_id="birds",
            title="Bird Sightings Log",
            source_path="/corpus/bird_sightings.md",
            start_line=12,
            end_line=24,
            score=0.85,
            snippet="The American Dipper forages…",
        )
        bib = citation.format_bibliography()
        assert "[1]" in bib
        assert "Bird Sightings Log" in bib
        assert "12" in bib

    def test_format_sources_section_empty(self):
        assert format_sources_section([]) == ""
