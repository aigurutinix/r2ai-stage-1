# Run From Scratch

## 1. Prepare Environment

Install dependencies:

```bash
pip install -r 04_configs/requirements.txt
pip install -U ollama
```

Start Ollama:

```bash
ollama serve
```

Pull LLM:

```bash
ollama pull qwen2.5:7b
```

## 2. Place Required Files

Copy these files from `01_data/data_for_run_full_final_pipeline/` to the project root:

```text
R2AIStage1DATA.json
articles_phapdien64_web_canonical.jsonl
bm25_dual_with_docs_phapdien64_web_canonical.pkl
dense_bge_m3_phapdien64_web.npy
```

If rebuilding corpus from raw data, also place:

```text
articles.jsonl
web_corpus_builder/raw_pages/
```

## 3. Optional: Rebuild Corpus

```bash
python web_corpus_builder/build_web_corpus_from_urls.py --sleep 0
python web_corpus_builder/dedup_web_corpus.py
python web_corpus_builder/merge_phapdien64_with_web.py
python canonicalize_legal_corpus.py
python build_bm25_bundle_canonical.py
```

If corpus row count changes, rebuild:

```text
dense_bge_m3_phapdien64_web.npy
```

## 4. Run Submission Notebook

Open:

```text
03_source_code/retrieval_pipeline/final_pipeline.ipynb
```

Set:

```python
RUN_LIMIT = None
```

Run all cells.

## 5. Final Output

The notebook creates:

```text
submission/<timestamp>_canonical_v6_3_same_doc_band90_92/results.json
submission/<timestamp>_canonical_v6_3_same_doc_band90_92/submission.zip
```

Submit:

```text
submission/<timestamp>_canonical_v6_3_same_doc_band90_92/submission.zip
```

Reference score from original v6.3 run:

```text
0.6543  0.6903  0.6000  0.7107  0.7100  0.7000
```


