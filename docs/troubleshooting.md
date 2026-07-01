# Troubleshooting — FieldNotes RAG Lab

## Diagnosis tool

When a query returns unexpected results, start with `examples/explain_query.py`.
It shows how the query is expanded, the IDF value of each token, and a
component-level breakdown of the hybrid score for every retrieved chunk.

```bash
python examples/explain_query.py \
    --query "What salamanders live in cold mountain streams?" \
    --corpus-dir examples/field_corpus
```

Sample output (truncated):

```
======================================================================
Explain query:  'What salamanders live in cold mountain streams?'
======================================================================

Expanded terms appended: creek riffle river brook watercourse drainage temperature frost chill cool

Effective tokens (after stopword removal): ['salamanders', 'live', 'cold', 'mountain', 'streams', 'creek', ...]

Term IDF in this index  (higher = rarer = more discriminating):
  salamanders              2.3219  ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪
  cold                     1.5850  ▪▪▪▪▪▪▪▪▪▪▪▪
  mountain                 2.0000  ▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪
  ...
  Not in corpus: live

Top-5 results  (α=0.60;  hybrid = α×TF-IDF + (1-α)×Cosine):

  [1] doc=species_notes  chunk=species_notes-0  lines 1–18
       hybrid=0.2847  = 0.60×TF-IDF(0.3921) + 0.40×Cosine(0.1023)
       TF-IDF driven by: salamanders(0.0984)  cold(0.0421)  tailed(0.0219)
       ┄ Pacific Giant Salamander and Tailed Frog inhabit cold, fast-moving …
...
```

The tool accepts a pre-built index to avoid re-ingesting on every run:

```bash
fieldnotes ingest examples/field_corpus --index my-index.json
python examples/explain_query.py -q "your query" --index my-index.json
```

---

## Common problems

### 1. "No relevant passages found" for a query that should match

**Cause:** Query terms and all their expanded synonyms are absent from the
indexed corpus.

**How to diagnose:** Run `explain_query.py` and look for:
- `TF-IDF: 0.0000` on every result row
- Many tokens listed under `Not in corpus`

**Fix options:**
- Add field notes that use the vocabulary your queries use (species names,
  protocol terms, habitat words).
- Extend `SYNONYMS` or `MORPHOLOGICAL_VARIANTS` in
  `fieldnotes_rag/query_expansion.py` with domain-specific synonyms.
- Lower `--min-score` (env: `FNRAG_MIN_SCORE`, default `0.01`) to let weaker
  hashed-cosine matches through.

---

### 2. Wrong documents ranked first

**Cause:** A generic query term (e.g. `record`, `species`, `observation`)
scores highly against many documents, letting an off-topic chunk win.

**How to diagnose:** In `explain_query.py`, check the `TF-IDF driven by:` line
for the top result.  If the contributing term is generic rather than
topic-specific, the query is under-specified.

**Fix options:**
- Add topic-specific terms to your query (species names, protocol keywords,
  habitat types).
- Reduce alpha to give more weight to the hashed-cosine component, which is
  more tolerant of vocabulary mismatch:
  ```bash
  FNRAG_HYBRID_ALPHA=0.4 fieldnotes ask "your query" --index my-index.json
  ```
- Add the specific term you care about to the relevant field notes so IDF
  discriminates better.

---

### 3. Answer coverage is low even though the right document is retrieved

**Cause:** Template synthesis pastes retrieved chunk text verbatim.  If the
expected keywords appear in a section of the document that falls outside every
retrieved chunk, they won't appear in the answer.

**Fix options:**
- Increase `--top-k` (env: `FNRAG_TOP_K`) to pull in more chunks.
- Decrease `chunk_size` (env: `FNRAG_CHUNK_SIZE`, default `200` words) so the
  relevant sentence lands in its own smaller chunk.
- Increase `chunk_overlap` (env: `FNRAG_CHUNK_OVERLAP`, default `40` words) so
  cross-boundary content appears in multiple chunks.

---

### 4. Evaluation fixture shows `Recall@k = 0.0` for a query

**Cause:** The `expected_doc_ids` in the fixture don't match the `doc_id`
values assigned by the loader.  The loader assigns `doc_id` equal to the
filename stem without extension.

**How to check:**
```bash
fieldnotes stats --index my-index.json
# or inspect directly:
python -c "
from fieldnotes_rag.index import HybridIndex
idx = HybridIndex.load('my-index.json')
print(sorted({c.doc_id for c in idx._chunks}))
"
```

Ensure fixture `expected_doc_ids` values match (e.g. `"bird_sightings"` for
`bird_sightings.md`).

---

### 5. Index build is slow for a large corpus

The index is write-once.  Build it once and reload on subsequent runs:

```bash
fieldnotes ingest /path/to/corpus --index my-index.json   # slow (once)
fieldnotes ask "query" --index my-index.json               # fast (always)
```

---

## Environment variables quick reference

| Variable | Default | Effect |
|---|---|---|
| `FNRAG_TOP_K` | `5` | Chunks retrieved per query |
| `FNRAG_MIN_SCORE` | `0.01` | Minimum hybrid score threshold |
| `FNRAG_HYBRID_ALPHA` | `0.6` | TF-IDF weight (0.0–1.0) |
| `FNRAG_CHUNK_SIZE` | `200` | Target words per chunk |
| `FNRAG_CHUNK_OVERLAP` | `40` | Overlap words between chunks |

See [`docs/scoring.md`](scoring.md) for the full hybrid score formula and
[`docs/evaluation.md`](evaluation.md) for metric definitions.
