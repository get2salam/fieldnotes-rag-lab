# FieldNotes RAG Lab

**Local-first retrieval-augmented generation for field researchers, naturalists, and outdoor scientists.**

FieldNotes RAG Lab is a fully offline RAG pipeline that ingests your Markdown and plain-text field notes — bird sightings, plant observations, habitat descriptions, weather logs, and safety checklists — then retrieves the most relevant passages and synthesizes cited answers to your queries. No cloud, no API keys, no heavy model downloads.

---

## Why RAG for field notes?

Field researchers accumulate hundreds of observation logs, species checklists, trail guides, and safety protocols. Standard keyword search misses context; large language models hallucinate species names and dates. RAG bridges the gap: retrieve exact passages from your own corpus, then compose grounded answers with inline citations.

---

## Architecture

```
                  ┌─────────────────────────────────┐
                  │           Your Corpus            │
                  │  (Markdown / TXT field notes)    │
                  └──────────────┬──────────────────┘
                                 │ ingest
                                 ▼
                  ┌──────────────────────────────────┐
                  │         Document Loader           │
                  │  reads .md / .txt, strips front  │
                  │  matter, extracts metadata        │
                  └──────────────┬───────────────────┘
                                 │ chunk
                                 ▼
                  ┌──────────────────────────────────┐
                  │           Chunker                 │
                  │  sentence-aware sliding window    │
                  │  preserves metadata + line refs   │
                  └──────────────┬───────────────────┘
                                 │ index
                                 ▼
                  ┌──────────────────────────────────┐
                  │        Hybrid Index               │
                  │  TF-IDF lexical + cosine-like     │
                  │  hashed feature scoring           │
                  └──────────────┬───────────────────┘
                                 │ retrieve
                                 ▼
                  ┌──────────────────────────────────┐
                  │         Retriever                 │
                  │  top-k chunks + relevance scores  │
                  │  + inline citations (file:line)   │
                  └──────────────┬───────────────────┘
                                 │ synthesize
                                 ▼
                  ┌──────────────────────────────────┐
                  │        Synthesizer                │
                  │  template-driven answer builder   │
                  │  with [1], [2]… citation refs     │
                  └──────────────┬───────────────────┘
                                 │
                                 ▼
                        Cited Answer + Sources
```

---

## Quickstart

```bash
# Install (no heavy dependencies)
pip install -e ".[dev]"

# Ingest the sample corpus
fieldnotes ingest examples/field_corpus --index my-index.json

# Ask a question
fieldnotes ask "What birds are common near fast-moving streams at dusk?" \
    --index my-index.json --top-k 3

# Run evaluation
fieldnotes eval --index my-index.json --fixtures examples/eval_fixtures.json

# Show index stats
fieldnotes stats --index my-index.json
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `fieldnotes ingest <dir>` | Load and index all `.md`/`.txt` files |
| `fieldnotes ask "<query>"` | Retrieve and synthesize a cited answer |
| `fieldnotes eval` | Run deterministic evaluation against fixtures |
| `fieldnotes stats` | Print index statistics |
| `fieldnotes export` | Export corpus as structured JSON |

---

## Sample Output

```
Query: What should I record after seeing a streamside bird at dusk?

Answer:
After observing a streamside bird at dusk, record the species (if known) and any
distinguishing field marks such as beak shape, tail posture, and wing pattern [1].
Note the exact microhabitat — whether the bird was perching on emergent rocks, hovering
over riffles, or foraging along the bank [2]. Include light conditions, estimated
visibility distance, and any vocalisations heard [3].

Sources:
[1] examples/field_corpus/bird_sightings.md (lines 12-24)
[2] examples/field_corpus/riparian_habitat.md (lines 8-19)
[3] examples/field_corpus/observation_protocol.md (lines 45-57)
```

---

## Project Layout

```
fieldnotes-rag-lab/
├── fieldnotes_rag/          # Core Python package
│   ├── loader.py            # Document loading & metadata extraction
│   ├── chunker.py           # Sentence-aware sliding-window chunker
│   ├── tokenizer.py         # Whitespace + punctuation tokenizer
│   ├── index.py             # TF-IDF + hybrid scoring index
│   ├── retriever.py         # Top-k retrieval with scoring
│   ├── citation.py          # Citation model (file, line range)
│   ├── synthesis.py         # Answer synthesis with citations
│   ├── evaluator.py         # Deterministic evaluation harness
│   ├── config.py            # Configuration dataclass
│   └── cli.py               # Typer-free argparse CLI
├── examples/
│   ├── field_corpus/        # Invented public-safe field notes
│   └── eval_fixtures.json   # Evaluation ground truth
├── tests/                   # Pytest test suite
├── docs/                    # Architecture & design docs
├── .github/workflows/       # GitHub Actions CI
└── pyproject.toml
```

---

## Evaluation

FieldNotes RAG Lab ships with a deterministic evaluation harness that measures:

- **Recall@k** — fraction of relevant documents retrieved
- **MRR** — mean reciprocal rank of first relevant result
- **Citation accuracy** — do synthesized citations resolve to real passages?
- **Answer coverage** — fraction of expected keywords present in answer

Run with: `fieldnotes eval --index <index> --fixtures <fixtures.json>`

---

## Design Decisions

- **No external AI APIs.** Everything runs locally on plain Python.
- **No binary model downloads.** TF-IDF and cosine similarity are computed from scratch using stdlib math.
- **Deterministic.** Given the same corpus and query, you always get the same ranked results.
- **Corpus-agnostic.** Works on any Markdown or TXT notes directory.

See `docs/design.md` for the full rationale.

---

## Troubleshooting

If a query returns unexpected results, use the query diagnosis tool to see
exactly how the hybrid score is computed and which terms are missing from the
corpus:

```bash
python examples/explain_query.py \
    --query "What salamanders live in cold mountain streams?" \
    --corpus-dir examples/field_corpus
```

The tool prints the expanded query, token IDF values, and a per-result score
breakdown into TF-IDF and hashed-cosine components.  See
[docs/troubleshooting.md](docs/troubleshooting.md) for common failure modes and
fixes.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
