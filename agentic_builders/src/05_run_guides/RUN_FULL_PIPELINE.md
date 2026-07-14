# Run Full Pipeline

Full run guide:

```text
05_run_guides/RUN_FROM_SCRATCH.md
```

Minimal reproduction flow:

```bash
pip install -r 04_configs/requirements.txt
pip install -U ollama
ollama pull qwen2.5:7b
```

Copy the files from:

```text
01_data/data_for_run_full_final_pipeline/
```

to the working directory, then open:

```text
03_source_code/retrieval_pipeline/final_pipeline.ipynb
```

Set:

```python
RUN_LIMIT = None
```

Run all cells. The final output is:

```text
submission/<timestamp>_canonical_v6_3_same_doc_band90_92/submission.zip
```
