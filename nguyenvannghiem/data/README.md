# Thư mục `data/`

Thư mục này chứa các cache/artifact nhỏ cần cho pipeline. Dữ liệu lớn (corpus, indexes)
cần được tải về hoặc rebuild trước khi chạy.

## Dữ liệu lớn (cần tải hoặc rebuild)

| Thư mục / File | Dung lượng | Cách lấy |
|----------------|-----------|----------|
| `vbpl_dataset/` | 6.6 GB | Tải từ [NghiemAbe/r2ai-legal-data](https://huggingface.co/datasets/NghiemAbe/r2ai-legal-data) |
| `data_final/` | 616 MB | Tải từ [NghiemAbe/r2ai-legal-data](https://huggingface.co/datasets/NghiemAbe/r2ai-legal-data) |
| `synthetic_qa/` | 939 MB | Tải từ [NghiemAbe/r2ai-legal-data](https://huggingface.co/datasets/NghiemAbe/r2ai-legal-data) |
| `retrieval_index_dense/` | 4.8 GB | Rebuild: `python3 code/retrieval_dense.py build-index` |
| `retrieval_index_bm25s_v7/` | 2.2 GB | Rebuild: `python3 code/retrieval_bm25s.py build-index` |
| `fewshot_index/` | 927 MB | Rebuild: `python3 code/build_fewshot_index.py` |

## Cache (đi kèm submission)

| File | Dung lượng | Mô tả |
|------|-----------|-------|
| `R2AIStage1DATA.json` | 520 KB | Tập test 2000 câu hỏi |
| `data_final_hard_negatives.jsonl` | 259 MB | Train data reranker (96,456 dòng) |
| `query_decompose_cache.json` | 708 KB | Cache sub-queries |
| `hyde_candidates.pkl` | 28 MB | HyDE candidates per query |
| `hyde_cache.json` | 1.3 MB | HyDE LLM cache |
| `rerank_intersection_scores.pkl` | 7.6 MB | Reranker scores trên intersection pool |
| `fewshot_search_top30.pkl` | 704 KB | Top-30 few-shot QA tương tự mỗi query |
| `fewshot_top5_reranked.pkl` | 12 MB | Few-shot top-5 đã rerank |
| `submission_3_1_decompose_bm25_top150.json` | 99 MB | BM25S decompose top-150 |

Mô tả chi tiết: xem [`../docs/data_description.md`](../docs/data_description.md).
