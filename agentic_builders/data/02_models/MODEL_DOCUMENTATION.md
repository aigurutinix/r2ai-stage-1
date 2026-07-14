# Model Documentation

## 1. Models Used

The system uses three model components:

```text
1. Dense embedding model:
   BAAI/bge-m3

2. Cross-encoder reranker:
   BAAI/bge-reranker-v2-m3

3. Local LLM for answer generation and citation verification:
   qwen2.5:7b via Ollama
```

## 2. Checkpoints

### BAAI/bge-m3

Usage:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")
```

Checkpoint source:

```text
https://huggingface.co/BAAI/bge-m3
```

### BAAI/bge-reranker-v2-m3

Usage:

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
```

Checkpoint source:

```text
https://huggingface.co/BAAI/bge-reranker-v2-m3
```

### qwen2.5:7b

Usage through Ollama:

```bash
ollama pull qwen2.5:7b
```

## 3. Shared Checkpoint Links

If checkpoints are pre-downloaded or exported, share them through Google Drive, OneDrive, or an equivalent platform.

Paste final links into:

```text
SHARED_LINKS_TEMPLATE.md
```

Required checkpoint/model links:

```text
BAAI/bge-m3 checkpoint or HuggingFace model link
BAAI/bge-reranker-v2-m3 checkpoint or HuggingFace model link
qwen2.5:7b Ollama model 
```


