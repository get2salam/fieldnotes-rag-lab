# Scoring — FieldNotes RAG Lab

## Hybrid Score Formula

For a query **Q** and a chunk **C**, the hybrid relevance score is:

```
score(Q, C) = α · tfidf(Q, C) + (1 − α) · cosine_hashed(Q, C)
```

where the default `α = 0.6`.

---

## TF-IDF Component

The TF-IDF score is computed as a dot product between the query TF-IDF vector and the chunk TF-IDF vector:

### Term Frequency (TF)
For a term `t` in a chunk `C` with `n` total tokens:

```
tf(t, C) = count(t in C) / n
```

### Inverse Document Frequency (IDF)
For a term `t` appearing in `df` out of `N` chunks, using smoothed IDF:

```
idf(t) = log((1 + N) / (1 + df)) + 1
```

The `+1` additive smoothing prevents terms that appear in all documents from getting a zero score, and ensures terms that don't appear in the corpus still have a small positive weight.

### TF-IDF Weight
```
tfidf_weight(t, C) = tf(t, C) × idf(t)
```

Each chunk vector is L2-normalised before scoring.

### Features
Both **unigrams** and **bigrams** are indexed. Bigrams capture compound terms like "American Dipper", "hip belt", "stream crossing", and "buddy system" that carry more information as a pair than as individual words.

---

## Hashed Feature Component

The hashed feature component provides soft similarity even when exact terms don't overlap.

Each token is hashed to one of `n_buckets = 1024` integer buckets using MD5:

```
bucket(t) = MD5(t) mod n_buckets
```

The feature vector for a chunk is a sparse mapping `{bucket: count}`, L2-normalised. The cosine similarity between the query feature vector and chunk feature vector gives the hashed score.

**Hash collisions** mean that different terms may map to the same bucket, providing a soft matching effect similar to (but weaker than) dense semantic embeddings. The probability of a useful collision is low for most terms, but it does provide non-zero scores for semantically related vocabulary that shares common character substrings.

---

## Query Expansion

Before scoring, the query is expanded using a hand-coded synonym table. Example:

- `"stream"` → expands to include `creek`, `riffle`, `river`, `brook`, `drainage`
- `"bird"` → expands to include `avian`, `passerine`, `raptor`, `waterfowl`
- `"dusk"` → expands to include `twilight`, `crepuscular`, `evening`, `sunset`

Expanded tokens are appended to the query string before tokenisation, increasing the chance of term overlap with relevant chunks.

---

## Deduplication

After scoring, results are deduplicated to allow at most 2 chunks per source document. This ensures that a single document does not dominate the results when multiple chunks from it score highly. Deduplication preserves the highest-scored chunks from each document.

---

## Alpha Tuning

The `alpha` parameter controls the balance between lexical and hashed matching:

| Alpha | Effect |
|---|---|
| 1.0 | Pure TF-IDF (exact term matching only) |
| 0.6 (default) | Mostly TF-IDF with soft hashed augmentation |
| 0.5 | Equal weight |
| 0.0 | Pure hashed cosine (soft matching only) |

For **species names and technical terms** (where exact spelling matters), higher alpha (0.7–0.9) is recommended.

For **paraphrase-heavy queries** (where the user may not use the exact corpus vocabulary), lower alpha (0.4–0.5) may improve recall.

Tune alpha via `FNRAG_HYBRID_ALPHA` env var or `--alpha` flag at ingest time.
