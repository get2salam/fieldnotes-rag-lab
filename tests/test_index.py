"""Tests for the hybrid TF-IDF index."""

import json
import math
from pathlib import Path

import pytest

from fieldnotes_rag.chunker import Chunk
from fieldnotes_rag.index import HybridIndex, IndexedChunk, _idf, _hash_features, _cosine_hashed


def _make_chunk(
    chunk_id: str,
    doc_id: str,
    text: str,
    start_line: int = 1,
    end_line: int = 5,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_path=f"/corpus/{doc_id}.md",
        title=doc_id.replace("_", " ").title(),
        text=text,
        start_line=start_line,
        end_line=end_line,
    )


SAMPLE_CHUNKS = [
    _make_chunk("bird_c0", "birds", "The American Dipper forages in fast-moving riffles and streams."),
    _make_chunk("bird_c1", "birds", "Belted Kingfisher rattles its call above open pools near riparian zones."),
    _make_chunk("plant_c0", "plants", "Red Alder dominates the riparian canopy and fixes nitrogen through root symbionts."),
    _make_chunk("safety_c0", "safety", "Stream crossing hazard requires buddy system and loose hip belt before wading."),
    _make_chunk("weather_c0", "weather", "Wind above 20 kph reduces auditory detection probability for bird surveys."),
]


class TestIdf:
    def test_idf_single_doc(self):
        # Term appearing in all docs → low IDF
        score = _idf(n_docs=5, df=5)
        assert score < _idf(n_docs=5, df=1)

    def test_idf_rare_term(self):
        score = _idf(n_docs=100, df=1)
        assert score > 1.0

    def test_idf_never_negative(self):
        for n, df in [(1, 1), (10, 5), (100, 100)]:
            assert _idf(n, df) >= 0


class TestHashFeatures:
    def test_returns_dict(self):
        vec = _hash_features(["bird", "water"])
        assert isinstance(vec, dict)

    def test_l2_normalised(self):
        vec = _hash_features(["bird", "water", "habitat"])
        norm = math.sqrt(sum(v * v for v in vec.values()))
        assert abs(norm - 1.0) < 1e-9

    def test_empty_tokens(self):
        vec = _hash_features([])
        assert vec == {}

    def test_same_tokens_same_result(self):
        a = _hash_features(["dipper", "riffle"])
        b = _hash_features(["dipper", "riffle"])
        assert a == b


class TestCosineHashed:
    def test_identical_vectors(self):
        vec = _hash_features(["bird", "water"])
        score = _cosine_hashed(vec, vec)
        assert abs(score - 1.0) < 1e-9

    def test_disjoint_vectors(self):
        a = _hash_features(["alpha_unique_term_xq"])
        b = _hash_features(["zeta_unique_term_yq"])
        score = _cosine_hashed(a, b)
        assert 0.0 <= score <= 1.0

    def test_empty_vectors(self):
        score = _cosine_hashed({}, {})
        assert score == 0.0


class TestHybridIndex:
    def setup_method(self):
        self.idx = HybridIndex()
        self.idx.build(SAMPLE_CHUNKS)

    def test_build_sets_counts(self):
        assert self.idx.num_chunks == len(SAMPLE_CHUNKS)
        assert self.idx.num_docs == 4  # birds (2 chunks), plants, safety, weather → 4 unique doc_ids

    def test_score_returns_results(self):
        results = self.idx.score("dipper riffle bird", top_k=3)
        assert len(results) <= 3
        assert len(results) >= 1

    def test_score_top_result_relevant(self):
        results = self.idx.score("dipper riffle bird", top_k=5)
        top_chunk, top_score = results[0]
        assert "bird" in top_chunk.doc_id or "dipper" in top_chunk.text.lower()

    def test_score_sorted_descending(self):
        results = self.idx.score("stream crossing safety hazard", top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_score_safety_query(self):
        results = self.idx.score("crossing buddy system wading safety", top_k=5)
        doc_ids = [c.doc_id for c, _ in results]
        assert "safety" in doc_ids

    def test_empty_query_returns_empty(self):
        results = self.idx.score("", top_k=5)
        assert results == []

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "test.json"
        self.idx.save(path)
        loaded = HybridIndex.load(path)
        assert loaded.num_chunks == self.idx.num_chunks
        assert loaded.vocab_size == self.idx.vocab_size

    def test_save_load_scores_match(self, tmp_path):
        path = tmp_path / "test.json"
        self.idx.save(path)
        loaded = HybridIndex.load(path)
        original = self.idx.score("bird dipper", top_k=3)
        reloaded = loaded.score("bird dipper", top_k=3)
        assert len(original) == len(reloaded)
        for (oc, os_), (rc, rs) in zip(original, reloaded):
            assert oc.chunk_id == rc.chunk_id
            assert abs(os_ - rs) < 1e-9

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            HybridIndex.load("/no/such/index.json")
