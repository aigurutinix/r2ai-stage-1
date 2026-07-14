# R2AI2026 Legal Assistant Project Guide

This project builds a Vietnamese legal information retrieval and question answering
system for the competition:

- Competition: R2AI2026 BUILD AI LEGAL ASSISTANT
- Platform: https://leaderboard.aiguru.com.vn/competitions/13/
- Domain: Vietnamese legal QA for SME/business use cases
- Primary task: retrieve relevant legal documents/articles and generate grounded answers

## Operating Objective

Optimize the final submission for:

1. `ARTICLES_F2MACRO`
2. `DOCS_F2MACRO`
3. Article/document precision and recall
4. QA quality: legally accurate, complete, practical, clear, and easy to verify

Treat `ARTICLES_F2MACRO` as the main target. The system should retrieve the most
relevant legal articles first, then derive relevant legal documents from those
articles.

## Data Policy

The team may collect and use data from any lawful and accessible source.

Preferred sources:

- Official Vietnamese legal portals and gazettes
- Laws, decrees, circulars, resolutions, and official guidance documents
- Government ministry or agency websites
- Public legal datasets
- Other legal sources that can be traced and reproduced

Every collected corpus file should be recorded with:

- Source URL or origin
- Collection date
- Document title
- Document code or identifier
- Effective date, if available
- Processing notes

Do not silently mix unknown data into the corpus. If a source is uncertain,
store it separately and mark it for review.

## Model Policy

Final competition runs must use only models that satisfy the competition rules:

- Open-source or publicly downloadable model weights
- Fewer than 14B parameters
- Officially released before 2026-03-01 Vietnam time
- Reproducible download/source information

Do not use closed LLM APIs such as GPT-4o, Gemini, Claude, or similar systems for
the final competition submission. They may be used only for local brainstorming
or analysis if the user explicitly requests it, and any such use must not leak
into the final generated answers unless allowed by the competition rules.

## Canonical Data Shape

Keep canonical legal data separate from retrieval-specific chunks.

Recommended structure:

```text
data/
  raw/
    competition/
      test_questions.json
    legal_sources/
  canonical/
    legal_documents.jsonl
    legal_articles.jsonl
  processed/
    chunks.jsonl
  indexes/
    bm25/
    vector/
  manifests/
    sources.csv
```

Canonical document record:

```json
{
  "document_number": "04/2017/QH14",
  "doc_name": "Luat 04/2017/QH14 Luat Ho tro doanh nghiep nho va vua",
  "doc_type": "Luat",
  "issued_date": "2017-06-12",
  "effective_date": "2018-01-01",
  "source_url": "...",
  "text": "..."
}
```

Canonical article record:

```json
{
  "article_id": "04/2017/QH14|Luat 04/2017/QH14 Luat Ho tro doanh nghiep nho va vua|Dieu 5",
  "document_number": "04/2017/QH14",
  "doc_name": "Luat 04/2017/QH14 Luat Ho tro doanh nghiep nho va vua",
  "article_no": "Dieu 5",
  "title": "...",
  "text": "..."
}
```

Retrieval chunk record:

```json
{
  "chunk_id": "chunk_000001",
  "article_id": "04/2017/QH14|Luat 04/2017/QH14 Luat Ho tro doanh nghiep nho va vua|Dieu 5",
  "document_number": "04/2017/QH14",
  "doc_name": "Luat 04/2017/QH14 Luat Ho tro doanh nghiep nho va vua",
  "article_no": "Dieu 5",
  "text": "..."
}
```

Every retrieval chunk must map back to exactly one canonical `article_id` when
possible. If a chunk spans multiple articles, split it before indexing.

## Current Elasticsearch Fields

The current Elasticsearch index is `legal_chunks`. It contains these fields:

| Field | Type | Indexed/searchable | Notes |
| --- | --- | --- | --- |
| `chunk_id` | `keyword` | Yes | Unique chunk identifier, also used as the ES document id. |
| `doc_id` | `keyword` | Yes | Source document id from the crawled/processed corpus. |
| `legal_type` | `keyword` | Yes | Document type, for example `Luật`, `Nghị định`, `Thông tư`, `Quyết định`. |
| `document_number` | `keyword` | Yes | Legal document number, for example `04/2017/QH14`, `80/2021/NĐ-CP`. |
| `effect_date` | `date` | Yes | Effective date, accepts `yyyy-MM-dd`, `dd/MM/yyyy`, strict date, or epoch millis. |
| `effectless_date` | `date` | Yes | End-of-effect date with the same date formats as `effect_date`. |
| `article_no` | `keyword` | Yes | Article/section label, for example `Điều 5`, `PHỤ LỤC`, `Mục 4`. |
| `raw_title` | `keyword` | No | Full document title returned in `_source`; not searchable/filterable in ES. |
| `article_title` | `keyword` | No | Article title returned in `_source`; not searchable/filterable in ES. |
| `raw_content` | `keyword` | No | Original chunk text returned in `_source`; not searchable/filterable in ES. |
| `content_search` | `text` | Yes | Main BM25 field, analyzed with `vi_whitespace_lower`. |

Important limitations:

- The current ES index does not contain `issuing_authority`, `signers`, `issuance_date`, or `is_local`.
- `issuing_authority` exists in source metadata, but it is not copied into `legal_chunks`.
- Filtering local documents in ES currently requires expensive wildcard checks on `document_number`; a future index should add an indexed boolean field such as `is_local`.

## Retrieval Strategy

Use article-level retrieval as the core design.

Recommended pipeline:

```text
question
  -> normalize Vietnamese legal query
  -> retrieve candidate chunks/articles with BM25 and/or dense vectors
  -> aggregate scores by article_id
  -> rerank candidate articles
  -> select relevant_articles
  -> derive relevant_docs from selected articles
  -> generate grounded answer from selected articles
  -> validate output schema
  -> package results.json into submission.zip
```

Important:

- Chunking is an internal search technique, not the final output unit.
- Final output must cite legal documents and legal articles in the required format.
- Favor high recall for articles, but avoid dumping unrelated articles.
- Never invent article numbers, law identifiers, or document names.

## Submission Schema

The final `results.json` must be a JSON array. Each item must contain:

```json
{
  "id": 1,
  "question": "...",
  "answer": "...",
  "relevant_docs": [
    "04/2017/QH14|Luat 04/2017/QH14 Luat Ho tro doanh nghiep nho va vua"
  ],
  "relevant_articles": [
    "04/2017/QH14|Luat 04/2017/QH14 Luat Ho tro doanh nghiep nho va vua|Dieu 5"
  ]
}
```

Rules:

- File name must be exactly `results.json`.
- Package as a flat zip named `submission.zip`.
- The zip must contain only `results.json` at the root.
- Do not put `results.json` inside a nested folder.
- Preserve the original `id` and `question` from the test set.
- `relevant_docs` format: `<document_number>|<doc_name>`.
- `relevant_articles` format: `<document_number>|<doc_name>|<article_no>`.

## Answer Style

Answers should:

- Directly answer the user's legal question
- Mention the legal basis in natural Vietnamese
- Be concise but sufficiently complete
- Include practical caveats when needed
- Avoid absolute legal advice when the facts are incomplete
- Avoid unsupported claims

Generation should be grounded only in retrieved legal articles. If retrieval is
weak, produce a cautious answer and cite the best available legal basis rather
than inventing authority.

## Evaluation Notes

`ARTICLES_F2MACRO`:

- Measures whether predicted `relevant_articles` match the hidden gold articles.
- F2 gives more weight to recall than precision.
- Macro means the score is calculated per question, then averaged.

`DOCS_F2MACRO`:

- Measures whether predicted `relevant_docs` match the hidden gold documents.
- Usually derive docs from selected articles to keep article and document outputs
  consistent.

## Standard Commands

Use these command names when implementing scripts:

```text
scripts/build_corpus.ps1
scripts/build_index.ps1
scripts/run_inference.ps1
scripts/validate_submission.ps1
scripts/make_submission_zip.ps1
```

Expected workflow:

```text
build_corpus -> build_index -> run_inference -> validate_submission -> make_submission_zip
```

## Experiment Tracking

Each experiment should store:

```text
experiments/
  YYYYMMDD_HHMM_name/
    config.yaml
    notes.md
    metrics.md
    errors.md
```

Record:

- Corpus version
- Index version
- Retrieval settings
- Model name/version
- Prompt/template version
- Submission file path
- Leaderboard score, if submitted

## Current Competition Dates

All deadlines are Vietnam time, UTC+07.

- 2026-06-03: opening and test set release
- 2026-06-30 23:59: public phase submission deadline
- 2026-07-05: Top 10 announcement
- 2026-07-11: DemoDay and final results

## Build Priorities

1. Create a reliable legal corpus with source manifests.
2. Parse documents into clean article-level records.
3. Build a strong BM25 baseline.
4. Add dense retrieval or reranking only after the baseline is measurable.
5. Validate `results.json` strictly before every submission.
6. Keep all outputs reproducible.

When in doubt, prefer a simple, auditable pipeline over a complex one that is
hard to debug before the competition deadline.
