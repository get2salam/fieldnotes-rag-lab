# Design Decisions — FieldNotes RAG Lab

## Why local-first RAG?

Field researchers often work in environments with no reliable internet connectivity. A nature observation notebook that requires cloud API calls for search is unusable in the field. FieldNotes RAG Lab is designed to run entirely offline on a laptop or field tablet.

## Why no external embedding model?

Most RAG tutorials begin "first, download a 400MB BERT model…". For a field notebook:

1. **Offline constraint.** Downloading a model requires internet access at setup time. Some field stations have no internet at all.
2. **Determinism constraint.** Model weights change between versions; a corpus indexed with model v1 may give different scores with model v2. FieldNotes RAG Lab guarantees identical scores for identical corpus + query combinations across any Python 3.9+ environment.
3. **Dependency constraint.** PyTorch, transformers, and their transitive dependencies total hundreds of megabytes and have complex installation requirements. stdlib + pyflakes is the entire stack.

The TF-IDF + hashed-feature hybrid is less powerful than dense neural embeddings for paraphrase retrieval, but it is more than adequate for the specific vocabulary of a field notes corpus, where exact term matches (species names, habitat terms, protocol keywords) are highly informative.

## Why bigrams in the index?

Bigrams capture compound field terms that lose meaning when split:
- "American Dipper" should score higher than matching "American" and "Dipper" separately
- "buddy system" is a specific protocol term
- "hip belt" appears in stream-crossing safety — matching the whole phrase matters

The bigram index is stored as plain string keys alongside unigrams, adding roughly 3–5× vocabulary size but no computational complexity change.

## Why sentence-aware chunking?

Alternative chunking strategies considered:
1. **Fixed character window:** Fast but splits sentences mid-word at boundaries.
2. **Paragraph-based:** Natural for prose but yields very uneven chunk sizes, with some paragraphs being a single sentence and others being 500 words.
3. **Sentence-based sliding window (chosen):** Sentences are the natural unit of semantic content. By accumulating sentences until hitting the `chunk_size` word budget, chunks remain coherent and are bounded in size.

The sliding window overlap ensures that content near chunk boundaries is retrievable regardless of which chunk the query lands in.

## Why citation by line number?

Line numbers in the source file are the most stable and verifiable reference point:
- They survive document reformatting that does not add/remove lines
- Any text editor can navigate to `file.md:45`
- They make manual verification of citations trivial
- They do not require storing large text snippets in the index (only the range)

## Why template synthesis instead of LLM generation?

The synthesis step is intentionally not an LLM call. Reasons:

1. **Hallucination prevention.** A language model synthesising from field notes might confidently state species information that is not in the retrieved passages. Template synthesis strictly pastes retrieved text.
2. **Offline operation.** No LLM API, no local model, no GPU required.
3. **Auditability.** Every word in the answer is traceable to a specific chunk. The citation `[2]` directly identifies the lines in the source file.

The trade-off is that the answer may read awkwardly because it stitches together sentences from different passages. For a field research notebook, verifiability is more important than fluency.

## Configuration layering

Configuration resolves in this order (later overrides earlier):
1. `Config` dataclass defaults
2. `FNRAG_*` environment variables (e.g., `FNRAG_TOP_K=10`)
3. CLI flags (e.g., `--top-k 10`)

This allows CI/CD pipelines to override defaults without modifying config files, and allows per-invocation overrides without changing environment variables.

## Future directions (not implemented)

- **Reranking pass:** After TF-IDF retrieval, apply a lightweight BM25 reranker on the candidate set
- **Semantic routing:** Route queries to topic-specific sub-indices (birds, plants, safety) before global scoring
- **Incremental indexing:** Append new documents to an existing index without full rebuild
- **Export to Markdown:** Generate a cited field report from a sequence of queries
