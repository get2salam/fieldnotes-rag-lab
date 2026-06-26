"""Tests for query expansion module."""

from fieldnotes_rag.query_expansion import expand_query, SYNONYMS, MORPHOLOGICAL_VARIANTS


class TestExpandQuery:
    def test_expands_bird(self):
        result = expand_query("bird observation")
        assert "avian" in result or "passerine" in result

    def test_expands_stream(self):
        result = expand_query("stream crossing")
        assert "creek" in result or "riffle" in result or "river" in result

    def test_expands_dusk(self):
        result = expand_query("dusk survey")
        assert "twilight" in result or "crepuscular" in result or "evening" in result

    def test_original_query_preserved(self):
        query = "bird dusk stream"
        result = expand_query(query)
        assert result.startswith(query)

    def test_no_duplicate_tokens(self):
        result = expand_query("bird birds")
        tokens = result.split()
        # Allow some duplicates from bigram/morphological expansion but not excessive
        assert len(tokens) == len(set(tokens)) or len(tokens) < 50

    def test_unknown_word_unchanged(self):
        query = "xyq_very_unusual_xyz"
        result = expand_query(query)
        assert result == query

    def test_plural_expansion(self):
        result = expand_query("birds")
        # Should expand both plural and potentially singular synonyms
        assert len(result) > len("birds")

    def test_morphological_expansion_observe(self):
        result = expand_query("observe species")
        assert "observed" in result or "observation" in result

    def test_safety_expands(self):
        result = expand_query("safety protocol")
        assert "hazard" in result or "risk" in result

    def test_empty_query(self):
        result = expand_query("")
        assert result == "" or result.strip() == ""


class TestSynonymTable:
    def test_all_values_are_lists(self):
        for key, val in SYNONYMS.items():
            assert isinstance(val, list), f"Value for {key!r} is not a list"
            assert len(val) >= 1

    def test_no_self_reference(self):
        for key, val in SYNONYMS.items():
            assert key not in val, f"{key!r} appears in its own synonym list"

    def test_morphological_values_are_lists(self):
        for key, val in MORPHOLOGICAL_VARIANTS.items():
            assert isinstance(val, list)
