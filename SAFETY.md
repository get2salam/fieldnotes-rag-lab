# Public Safety and Data Notice

## What This Project Contains

FieldNotes RAG Lab is a demonstration project for retrieval-augmented generation (RAG) techniques applied to field-research notebooks. All content in this repository is:

- **Invented.** The field notes, species observations, survey data, and location markers in `examples/field_corpus/` are fictional. They do not represent any real survey area, real species population data, or real field station.
- **Ecologically plausible.** The invented content is based on common field-ecology practice and publicly available natural history information, but should not be used as a guide for real field operations.
- **Public-safe.** No real GPS coordinates, private individuals' data, legally sensitive information, or confidential survey data is present anywhere in this repository.

## What This Project Does NOT Contain

- No real survey data from any research institution or government agency
- No coordinates of sensitive species nesting sites or rare plant locations
- No personal information about any real people
- No proprietary or confidential field protocols
- No data covered by the Endangered Species Act or equivalent regulations

## Using Your Own Corpus

If you use FieldNotes RAG Lab with your own field notes:

1. **Strip sensitive coordinates** before ingestion — the index stores chunk text verbatim
2. **Review species sensitivity** — locations of nesting rare species should not be in a shared index
3. **Respect data licences** — field data collected under research permits may have restrictions on sharing
4. **Treat the index as the corpus** — the `.fieldnotes-index.json` file contains the text of all indexed chunks and should be treated with the same sensitivity as the source notes

## Bug Reports

If you discover that any unintended private, sensitive, or incorrect data has been included in this repository, please open an issue immediately so it can be removed.
