# Contributing to FieldNotes RAG Lab

Thank you for your interest in contributing! This project welcomes contributions of all kinds: bug reports, new corpus notes, additional evaluation fixtures, improvements to the retrieval algorithm, and documentation.

---

## Getting Started

```bash
# Fork and clone
git clone https://github.com/get2salam/fieldnotes-rag-lab.git
cd fieldnotes-rag-lab

# Install in development mode with test dependencies
pip install -e ".[dev]"

# Run tests to verify your setup
make test

# Run the full demo
make demo
```

---

## Types of Contributions

### Bug Reports
Open an issue with:
- Python version and OS
- The command that failed
- The full error output

### New Corpus Notes
Add `.md` files to `examples/field_corpus/`. Notes should be:
- Ecologically plausible (invented but realistic)
- Written in the field-guide style already used in the corpus
- Public-safe (no private individuals, real locations with GPS, or sensitive data)
- At least 200 words to ensure meaningful chunking

### New Evaluation Fixtures
Add entries to `examples/eval_fixtures.json` following the schema:
```json
{
  "query_id": "q11",
  "query": "Your question here?",
  "expected_doc_ids": ["corpus_filename_without_extension"],
  "expected_keywords": ["keyword1", "keyword2"],
  "min_citations": 1
}
```
Run `make eval` after adding fixtures to check your expected scores.

### Algorithm Improvements
For changes to the retrieval algorithm (`index.py`, `retriever.py`):
1. Baseline the current evaluation scores first: `make eval`
2. Make your change
3. Compare the new scores — improvements to composite score are welcome
4. Document the rationale in your PR description

### Synonym Table Additions
Add entries to `fieldnotes_rag/query_expansion.py`. Field-research vocabulary expansions are especially welcome. Follow the format:
```python
"fieldterm": ["synonym1", "synonym2", "synonym3"],
```

---

## Code Style

- Standard library only — no new runtime dependencies unless well-justified
- Type hints on all public functions
- Tests for new public functions (in `tests/`)
- No comments explaining *what* the code does — only *why* when non-obvious
- Line length: 88 characters (Black-compatible, but Black is not required)

---

## Testing

```bash
# Run the full test suite
make test

# Run a specific test file
python -m pytest tests/test_index.py -v

# Run tests with coverage
python -m pytest --cov=fieldnotes_rag --cov-report=term-missing
```

All tests must pass before a PR is merged. New functionality should be accompanied by tests.

---

## Pull Request Process

1. Create a branch: `git checkout -b feature/your-feature-name`
2. Make changes with meaningful commits (one logical change per commit)
3. Run `make test` and `make lint` — both must pass
4. Open a PR against `main` with a description of what changed and why
5. The CI will run automatically; fix any failures before requesting review

---

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct. Be respectful, constructive, and kind.
