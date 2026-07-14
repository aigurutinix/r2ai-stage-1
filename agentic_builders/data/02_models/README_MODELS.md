# Models Folder

This folder documents the models used by the final pipeline.

Main documentation:

```text
02_models/MODEL_DOCUMENTATION.md
```

Models used:

```text
BAAI/bge-m3
BAAI/bge-reranker-v2-m3
qwen2.5:7b via Ollama
```

The notebook downloads the HuggingFace models through `sentence-transformers` if they are not already cached. The Ollama model can be prepared with:

```bash
ollama pull qwen2.5:7b
```
