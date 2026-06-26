.PHONY: install test lint compile demo eval clean help

PYTHON := python3
INDEX  := .fieldnotes-index.json
CORPUS := examples/field_corpus
FIXTURES := examples/eval_fixtures.json

help:
	@echo "FieldNotes RAG Lab — available targets:"
	@echo "  make install   Install package in dev mode"
	@echo "  make test      Run the test suite"
	@echo "  make lint      Run pyflakes lint check"
	@echo "  make compile   Syntax-check all Python files"
	@echo "  make demo      Ingest sample corpus and run demo query"
	@echo "  make eval      Run evaluation fixtures"
	@echo "  make clean     Remove generated files"

install:
	$(PYTHON) -m pip install -e ".[dev]"

compile:
	$(PYTHON) -m compileall fieldnotes_rag tests

lint:
	$(PYTHON) -m pyflakes fieldnotes_rag/

test:
	$(PYTHON) -m pytest -q --tb=short

demo: $(INDEX)
	$(PYTHON) -m fieldnotes_rag.cli ask \
		"What should I record after seeing a streamside bird at dusk?" \
		--index $(INDEX) \
		--top-k 3

$(INDEX):
	$(PYTHON) -m fieldnotes_rag.cli ingest $(CORPUS) --index $(INDEX)

eval: $(INDEX)
	$(PYTHON) -m fieldnotes_rag.cli eval \
		--index $(INDEX) \
		--fixtures $(FIXTURES)

stats: $(INDEX)
	$(PYTHON) -m fieldnotes_rag.cli stats --index $(INDEX)

clean:
	rm -f $(INDEX)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf *.egg-info dist build
