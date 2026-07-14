# Building AI Legal Assistant - Reproducible Submission

## Package Layout

```text
01_data/        Data documentation, raw legal documents, build reports
02_models/      Model/checkpoint documentation
03_source_code/ Source code and notebooks
04_configs/     Requirements and run config
05_run_guides/  Step-by-step run guide
06_outputs/     Sample outputs on public dataset
```

## Required Files

Before running, copy these files from `01_data/data_for_run_full_final_pipeline/` to the working directory where the notebook runs:

```text
R2AIStage1DATA.json
articles_phapdien64_web_canonical.jsonl
bm25_dual_with_docs_phapdien64_web_canonical.pkl
dense_bge_m3_phapdien64_web.npy
```

If rebuilding data from raw sources, also place:

```text
articles.jsonl
web_corpus_builder/raw_pages/
```

## Install

```bash
pip install -r 04_configs/requirements.txt
pip install -U ollama
```

Start Ollama and download the LLM:

```bash
ollama serve
ollama pull qwen2.5:7b
```

## Run

Open:

```text
03_source_code/retrieval_pipeline/final_pipeline.ipynb
```

Set:

```python
RUN_LIMIT = None
```

Run all cells.

Final file:

```text
submission/<timestamp>_canonical_v6_3_same_doc_band90_92/submission.zip
```

Reference score from original v6.3 run:

```text
0.6543  0.6903  0.6000  0.7107  0.7100  0.7000
```

Reference public output is included at:

```text
06_outputs/submiss_0.6543_public/submission.zip
```

This output is for comparison only. Re-running the notebook creates a new `submission.zip`.

## Data And Model Links

Fill links in:

```text
SHARED_LINKS_TEMPLATE.md
```

## More Documentation

```text
01_data/DATA_DOCUMENTATION.md
02_models/MODEL_DOCUMENTATION.md
05_run_guides/RUN_FROM_SCRATCH.md
SUBMISSION_STRUCTURE.md
```




