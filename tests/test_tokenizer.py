"""Tests for the tokenizer module."""

from fieldnotes_rag.tokenizer import (
    normalize,
    tokenize,
    split_sentences,
    ngrams,
    term_frequency,
    STOP_WORDS,
)


class TestNormalize:
    def test_lowercases(self):
        assert normalize("BIRD") == "bird"

    def test_strips_punctuation(self):
        result = normalize("Hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_preserves_hyphens(self):
        result = normalize("black-capped chickadee")
        assert "black-capped" in result

    def test_collapses_whitespace(self):
        result = normalize("  multiple   spaces  ")
        assert "  " not in result
        assert result.strip() == result


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = tokenize("The American Dipper forages in riffles.")
        assert "american" in tokens or "dipper" in tokens

    def test_removes_stop_words(self):
        tokens = tokenize("The bird is in the water.", remove_stopwords=True)
        assert "the" not in tokens
        assert "is" not in tokens
        assert "in" not in tokens

    def test_keeps_words_when_no_filter(self):
        tokens = tokenize("The bird is in the water.", remove_stopwords=False)
        assert "the" in tokens

    def test_returns_list(self):
        result = tokenize("Species name here.")
        assert isinstance(result, list)

    def test_empty_string(self):
        assert tokenize("") == []

    def test_only_stop_words(self):
        tokens = tokenize("the is a")
        assert tokens == []


class TestSplitSentences:
    def test_splits_on_period(self):
        text = "The dipper forages. The kingfisher hovers."
        sentences = split_sentences(text)
        assert len(sentences) >= 2

    def test_single_sentence(self):
        text = "The dipper forages underwater."
        sentences = split_sentences(text)
        assert len(sentences) == 1

    def test_empty_string(self):
        sentences = split_sentences("")
        assert sentences == []


class TestNgrams:
    def test_bigrams(self):
        tokens = ["american", "dipper", "forages", "underwater"]
        result = ngrams(tokens, 2)
        assert "american dipper" in result
        assert "dipper forages" in result
        assert len(result) == 3

    def test_unigrams(self):
        tokens = ["bird", "water"]
        result = ngrams(tokens, 1)
        assert result == ["bird", "water"]

    def test_n_larger_than_tokens(self):
        tokens = ["bird"]
        result = ngrams(tokens, 3)
        assert result == []

    def test_empty_tokens(self):
        assert ngrams([], 2) == []


class TestTermFrequency:
    def test_sums_to_one(self):
        tokens = ["bird", "water", "bird", "habitat"]
        tf = term_frequency(tokens)
        assert abs(sum(tf.values()) - 1.0) < 1e-9

    def test_relative_frequency(self):
        tokens = ["bird", "bird", "water"]
        tf = term_frequency(tokens)
        assert abs(tf["bird"] - 2 / 3) < 1e-9
        assert abs(tf["water"] - 1 / 3) < 1e-9

    def test_empty_tokens(self):
        assert term_frequency([]) == {}
