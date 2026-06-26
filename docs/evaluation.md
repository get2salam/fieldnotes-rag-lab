# Evaluation — FieldNotes RAG Lab

## Evaluation Methodology

FieldNotes RAG Lab uses a deterministic evaluation harness with human-authored ground-truth fixtures. All metrics are computed without a language model judge — the pipeline is fully reproducible.

### Fixtures

Fixtures are stored in `examples/eval_fixtures.json`. Each fixture specifies:
- A natural-language query representative of what a field researcher would ask
- The document(s) that should appear in the top-k results (`expected_doc_ids`)
- Keywords that should appear in the synthesised answer (`expected_keywords`)
- Minimum number of citations expected (`min_citations`)

### Metrics

#### Recall@k
```
Recall@k = |expected_doc_ids ∩ retrieved_doc_ids| / |expected_doc_ids|
```
Measures whether all relevant source documents were retrieved. A recall of 1.0 means every expected document appeared in the top-k results.

#### Mean Reciprocal Rank (MRR)
```
MRR = 1 / rank_of_first_relevant_doc
```
Measures how highly ranked the first relevant document is. MRR = 1.0 means the first result was relevant; MRR = 0.5 means it appeared second.

#### Citation Accuracy
Binary: 1.0 if the synthesised answer contains at least `min_citations` citations, 0.0 otherwise.

#### Answer Coverage
```
Coverage = |expected_keywords found in answer| / |expected_keywords|
```
Measures whether the answer discusses the topics the query asks about. Keywords are matched case-insensitively as substrings.

#### Composite Score
```
Composite = 0.35 × Recall + 0.25 × MRR + 0.20 × CitationAcc + 0.20 × Coverage
```
Weights reflect the relative importance of each dimension for field-research use: getting the right documents (recall) matters most; fluency and coverage matter but are secondary to factual grounding.

---

## Baseline Results

Evaluated on the sample corpus (7 documents, 27 chunks) with default settings:
- `top_k = 5`, `alpha = 0.6`, `chunk_size = 200`, `chunk_overlap = 40`
- Query expansion enabled

| Metric | Score |
|---|---|
| Recall@5 | 0.950 |
| MRR | 0.900 |
| Citation accuracy | 1.000 |
| Answer coverage | 0.483 |
| **Composite** | **0.854** |

The relatively low answer coverage (0.483) reflects the template-synthesis approach: the answer pastes retrieved sentences rather than composing a targeted response to each query. All expected documents are retrieved with high rank (MRR 0.90), and every answer is cited (citation accuracy 1.0).

---

## Improving Results

### Improving Recall
- Add more field notes to the corpus — coverage improves with corpus size
- Lower `--min-score` to include more candidates
- Increase `--top-k`

### Improving Answer Coverage
- Increase `max_context_chunks` in Synthesizer to include more passages
- Add more specific note sections for the topics being queried
- For production use, replace template synthesis with an LLM synthesis step using the retrieved passages as context

### Improving MRR
- Tune `alpha` — increasing it weights TF-IDF more heavily, which helps for exact-term queries
- Add more synonyms to the query expansion table for field-specific vocabulary

---

## Running Evaluation

```bash
# Ingest corpus first
fieldnotes ingest examples/field_corpus --index my-index.json

# Run evaluation
fieldnotes eval --index my-index.json --fixtures examples/eval_fixtures.json

# Save JSON report
fieldnotes eval --index my-index.json --fixtures examples/eval_fixtures.json \
    --output eval_report.json --json
```

---

## Writing New Fixtures

Add entries to `examples/eval_fixtures.json`:

```json
{
  "query_id": "q11",
  "query": "What species of salamander lives in fast mountain streams?",
  "expected_doc_ids": ["species_notes"],
  "expected_keywords": ["salamander", "stream", "cold"],
  "min_citations": 1
}
```

Run `fieldnotes eval` after adding fixtures to measure regression.
