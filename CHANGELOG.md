# Changelog

All notable changes to FieldNotes RAG Lab are documented here.

Format: [version] - YYYY-MM-DD

---

## [Unreleased]

### Added
- Initial implementation of the complete RAG pipeline
- Document loader with front-matter stripping and inline metadata extraction
- Sentence-aware sliding-window chunker with line-number tracking
- TF-IDF + hashed-feature cosine hybrid index (fully offline, no model downloads)
- Retriever with synonym-based query expansion and per-document deduplication
- Template-driven answer synthesizer with inline citation markers [1], [2]…
- Evaluation harness with Recall@k, MRR, citation accuracy, and answer coverage
- CLI with `ingest`, `ask`, `eval`, `stats`, and `export` subcommands
- Sample field corpus: 7 Markdown documents covering bird sightings, riparian habitat,
  plant observations, weather logs, species notes, safety checklists, and survey protocol
- 10 evaluation fixtures with ground-truth expected documents and keywords
- GitHub Actions CI: multi-Python matrix (3.9–3.12), compile check, pytest, CLI smoke tests
- Query expansion module with 50+ field-research synonyms and morphological variants
- Configuration dataclass with environment-variable override support
- Custom exception hierarchy for clear error messages
- Architecture, design, and evaluation documentation in `docs/`
- Makefile with `install`, `test`, `lint`, `demo`, `eval`, and `clean` targets
- Demo script for batch query demonstration

### Performance (sample corpus, default settings)
- Recall@5: 0.950
- MRR: 0.900
- Citation accuracy: 1.000
- Composite score: 0.854

---

## Versioning

This project follows [Semantic Versioning](https://semver.org/):
- **MAJOR**: breaking changes to the CLI or index format
- **MINOR**: new features, backward-compatible
- **PATCH**: bug fixes and documentation updates

The index JSON format is versioned implicitly by the `alpha` and `n_buckets` fields.
An index built with one version may score differently with a different version if
algorithm internals change — rebuild the index after upgrading.
