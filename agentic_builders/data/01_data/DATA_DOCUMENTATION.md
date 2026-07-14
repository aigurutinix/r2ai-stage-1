# Data Documentation

## 1. Data Sources

The system uses a Vietnamese legal corpus for SME legal question answering.

Main sources:

```text
1. articles.jsonl
   Original Phapdien legal corpus.

2. web_corpus_builder/raw_pages/
   Manually added full-text legal documents from public legal sources.
```

The final canonical corpus used by the notebook is:

```text
articles_phapdien64_web_canonical.jsonl
```

## 2. Data Build Summary

The 71k corpus build used:

```text
Phapdien base rows: 64,464
Web rows after dedup: 6,586
Final merged rows: 71,050
Raw web items parsed: 92
Unique web document codes: 89
```

The raw documents used in the 71k build are listed in:

```text
01_data/raw_documents_used_for_71k_corpus.md
```

## 3. Data Format

Each corpus row is a legal article-level record in JSONL format.

Important fields:

```text
doc_code              Legal document code, e.g. 04/2017/QH14
doc_title             Canonical legal document title
canonical_doc_key     doc_code + canonical title
article_ref           Legal article reference, e.g. Điều 4
article_title         Article title
content               Full article text
topic_title           Legal topic/domain
chapter_title         Chapter title, if available
source_note           Source/legal note, if available
bm25_search           Text field used for BM25 retrieval
```

The test questions are stored in:

```text
R2AIStage1DATA.json
```

Expected submission format:

```json
[
  {
    "id": 1,
    "question": "...",
    "answer": "...",
    "relevant_docs": ["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa"],
    "relevant_articles": ["04/2017/QH14|Luật Hỗ trợ doanh nghiệp nhỏ và vừa|Điều 4"]
  }
]
```

## 4. Data Access

Data artifacts should be shared through Google Drive, OneDrive, or an equivalent platform.

Paste final links into:

```text
SHARED_LINKS_TEMPLATE.md
```

Required shared data files:

```text
R2AIStage1DATA.json
articles.jsonl
articles_phapdien64_web_canonical.jsonl
bm25_dual_with_docs_phapdien64_web_canonical.pkl
dense_bge_m3_phapdien64_web.npy
```

## 5. Where To Add New Raw Data

Add new raw legal documents here:

```text
01_data/raw_pages/
```

Preferred filename style:

```text
04_2017_QH14_Luat_Ho_tro_doanh_nghiep_nho_va_vua.html
80_2021_ND_CP_Huong_dan_Luat_Ho_tro_DNNVV.html
50_2005_QH11_Luat_So_huu_tri_tue.html
```

## 6. Corpus Build Flow

Run from the project root:

```bash
python web_corpus_builder/build_web_corpus_from_urls.py --sleep 0
python web_corpus_builder/dedup_web_corpus.py
python web_corpus_builder/merge_phapdien64_with_web.py
python canonicalize_legal_corpus.py
python build_bm25_bundle_canonical.py
```

Flow:

```text
raw_pages/
-> web_corpus_articles.jsonl
-> web_corpus_articles_dedup.jsonl
-> articles_phapdien64_web_bm25.jsonl
-> articles_phapdien64_web_canonical.jsonl
-> bm25_dual_with_docs_phapdien64_web_canonical.pkl
-> dense_bge_m3_phapdien64_web.npy
```

If the number of rows in `articles_phapdien64_web_canonical.jsonl` changes, rebuild dense embeddings.

