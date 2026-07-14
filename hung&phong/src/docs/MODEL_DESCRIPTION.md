# TÃ€I LIá»†U MÃ” Táº¢ MÃ” HÃŒNH Sá»¬ Dá»¤NG

**Dá»± Ã¡n:** Chatbot tra cá»©u vÄƒn báº£n phÃ¡p luáº­t Viá»‡t Nam
**PhiÃªn báº£n tÃ i liá»‡u:** 1.0
**NgÃ y cáº­p nháº­t:** 30/06/2026
**ÄÆ°á»ng dáº«n chia sáº» dá»¯ liá»‡u/checkpoint:** https://drive.google.com/drive/folders/1NvkAEpOjOqHNqF9kC-FvrOs5fjIiscTj

## 1. Má»¥c Ä‘Ã­ch tÃ i liá»‡u

TÃ i liá»‡u nÃ y mÃ´ táº£ cÃ¡c mÃ´ hÃ¬nh Ä‘Æ°á»£c sá»­ dá»¥ng trong há»‡ thá»‘ng chatbot tra cá»©u vÄƒn báº£n phÃ¡p luáº­t Viá»‡t Nam, bao gá»“m:

- MÃ´ hÃ¬nh sinh cÃ¢u tráº£ lá»i.
- MÃ´ hÃ¬nh embedding dÃ¹ng Ä‘á»ƒ táº¡o vector truy há»“i.
- MÃ´ hÃ¬nh reranker dÃ¹ng Ä‘á»ƒ xáº¿p háº¡ng láº¡i káº¿t quáº£ truy há»“i.
- ThÃ´ng tin checkpoint, cÃ¡ch táº£i vÃ  cÃ¡ch cáº¥u hÃ¬nh sá»­ dá»¥ng.

Há»‡ thá»‘ng hiá»‡n táº¡i **khÃ´ng fine-tune mÃ´ hÃ¬nh riÃªng cho báº£n váº­n hÃ nh má»›i nháº¥t**. Do Ä‘Ã³, khÃ´ng cÃ³ checkpoint ná»™i bá»™ báº¯t buá»™c pháº£i bÃ n giao. CÃ¡c checkpoint cáº§n sá»­ dá»¥ng Ä‘á»u lÃ  checkpoint public tá»« Ollama hoáº·c Hugging Face.

## 2. Tá»•ng quan mÃ´ hÃ¬nh

| ThÃ nh pháº§n | MÃ´ hÃ¬nh sá»­ dá»¥ng | Nguá»“n checkpoint | Vai trÃ² |
|---|---|---|---|
| LLM sinh cÃ¢u tráº£ lá»i | `qwen-vbpl`, táº¡o tá»« `qwen3.5:9b` | Ollama | Sinh cÃ¢u tráº£ lá»i, judge/verify, phÃ¢n rÃ£ hoáº·c chá»n cÄƒn cá»© trong má»™t sá»‘ pipeline |
| Embedding | `AITeamVN/Vietnamese_Embedding_v2` | Hugging Face | MÃ£ hÃ³a cÃ¢u há»i vÃ  vÄƒn báº£n thÃ nh vector 1024 chiá»u |
| Reranker | `AITeamVN/Vietnamese_Reranker` | Hugging Face | Cháº¥m láº¡i cáº·p cÃ¢u há»i - Ä‘iá»u luáº­t Ä‘á»ƒ tÄƒng Ä‘á»™ chÃ­nh xÃ¡c top-k |
| BM25 | KhÃ´ng pháº£i neural model | File local `.pkl` | Truy há»“i lexical, há»— trá»£ cÃ¡c truy váº¥n cÃ³ sá»‘ hiá»‡u vÄƒn báº£n/Ä‘iá»u/kÃ½ hiá»‡u |

## 3. Káº¿t luáº­n vá» checkpoint ná»™i bá»™

Báº£n váº­n hÃ nh má»›i nháº¥t cá»§a há»‡ thá»‘ng **khÃ´ng sá»­ dá»¥ng checkpoint fine-tune ná»™i bá»™**.

Do Ä‘Ã³:

- KhÃ´ng cÃ³ file checkpoint tá»± huáº¥n luyá»‡n báº¯t buá»™c pháº£i upload.
- KhÃ´ng cáº§n bÃ n giao thÆ° má»¥c `models/` náº¿u chá»‰ tÃ¡i láº­p pipeline má»›i nháº¥t.
- CÃ¡c checkpoint cáº§n dÃ¹ng Ä‘Æ°á»£c táº£i trá»±c tiáº¿p tá»« nguá»“n public:
  - Ollama: `qwen3.5:9b`
  - Hugging Face: `AITeamVN/Vietnamese_Embedding_v2`
  - Hugging Face: `AITeamVN/Vietnamese_Reranker`

LÆ°u Ã½: repo cÃ³ thÆ° má»¥c `models/` chá»©a má»™t sá»‘ checkpoint reranker thá»­ nghiá»‡m/fine-tune cÅ©, vÃ­ dá»¥ `models/reranker_vbpl_v2`, `models/reranker_vbpl_v3`. CÃ¡c checkpoint nÃ y **khÃ´ng pháº£i mÃ´ hÃ¬nh Ä‘ang dÃ¹ng trong pipeline má»›i nháº¥t** vÃ¬ pipeline má»›i Ä‘áº·t `RERANKER_MODEL=AITeamVN/Vietnamese_Reranker`.

Náº¿u trong tÆ°Æ¡ng lai chuyá»ƒn sang dÃ¹ng `RERANKER_MODEL=models/reranker_vbpl_v2` hoáº·c má»™t checkpoint ná»™i bá»™ khÃ¡c, khi Ä‘Ã³ báº¯t buá»™c pháº£i chia sáº» checkpoint tÆ°Æ¡ng á»©ng.

## 4. MÃ´ hÃ¬nh sinh cÃ¢u tráº£ lá»i

### 4.1. ThÃ´ng tin mÃ´ hÃ¬nh

| Thuá»™c tÃ­nh | GiÃ¡ trá»‹ |
|---|---|
| TÃªn local trong há»‡ thá»‘ng | `qwen-vbpl` |
| Base model | `qwen3.5:9b` |
| Nguá»“n | Ollama model library |
| Vai trÃ² | Sinh cÃ¢u tráº£ lá»i cuá»‘i, há»— trá»£ judge/listwise/selector trong cÃ¡c pipeline thá»­ nghiá»‡m |
| CÃ¡ch gá»i | OpenAI-compatible API qua Ollama |
| Endpoint máº·c Ä‘á»‹nh | `http://localhost:11434/v1` |
| Context window cáº¥u hÃ¬nh | 16,384 token |
| Max output cáº¥u hÃ¬nh | 2,048 token |
| Temperature | 0.0 trong `.env`, 0.2 trong `scripts/Modelfile.qwen-vbpl` |

Model `qwen-vbpl` khÃ´ng pháº£i checkpoint fine-tune. ÄÃ¢y lÃ  má»™t model tag cá»¥c bá»™ trong Ollama, Ä‘Æ°á»£c táº¡o tá»« `qwen3.5:9b` báº±ng `scripts/Modelfile.qwen-vbpl` Ä‘á»ƒ tÄƒng context window vÃ  giá»›i háº¡n output phÃ¹ há»£p vá»›i RAG.

### 4.2. Modelfile sá»­ dá»¥ng

File cáº¥u hÃ¬nh:

```text
scripts/Modelfile.qwen-vbpl
```

Ná»™i dung chÃ­nh:

```text
FROM qwen3.5:9b
PARAMETER num_ctx 16384
PARAMETER num_predict 2048
PARAMETER temperature 0.2
```

### 4.3. HÆ°á»›ng dáº«n táº£i vÃ  táº¡o model local

CÃ i Ollama, sau Ä‘Ã³ cháº¡y:

```powershell
ollama pull qwen3.5:9b
ollama create qwen-vbpl -f scripts/Modelfile.qwen-vbpl
```

Kiá»ƒm tra model:

```powershell
ollama list
ollama show qwen-vbpl
```

Cháº¡y thá»­:

```powershell
ollama run qwen-vbpl
```

### 4.4. Cáº¥u hÃ¬nh trong `.env`

```env
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen-vbpl
LLM_MAX_TOKENS=2048
LLM_TEMPERATURE=0.0
LLM_REASONING_EFFORT=none
```

## 5. MÃ´ hÃ¬nh embedding

### 5.1. ThÃ´ng tin mÃ´ hÃ¬nh

| Thuá»™c tÃ­nh | GiÃ¡ trá»‹ |
|---|---|
| Model ID | `AITeamVN/Vietnamese_Embedding_v2` |
| Nguá»“n | Hugging Face |
| Loáº¡i mÃ´ hÃ¬nh | Sentence Transformer / bi-encoder embedding |
| Base model | `BAAI/bge-m3` |
| NgÃ´n ngá»¯ | Tiáº¿ng Viá»‡t |
| Sá»‘ chiá»u vector | 1024 |
| License | Apache 2.0 |
| Vai trÃ² | MÃ£ hÃ³a cÃ¢u há»i vÃ  cÃ¡c chunk vÄƒn báº£n phÃ¡p luáº­t thÃ nh vector Ä‘á»ƒ truy há»“i dense |

Theo model card cá»§a Hugging Face, `AITeamVN/Vietnamese_Embedding_v2` lÃ  embedding model Ä‘Æ°á»£c fine-tune tá»« `BAAI/bge-m3`, tá»‘i Æ°u cho truy há»“i tiáº¿ng Viá»‡t. ÄÃ¢y lÃ  checkpoint public, khÃ´ng pháº£i checkpoint do nhÃ³m tá»± fine-tune.

### 5.2. HÆ°á»›ng dáº«n táº£i

Model Ä‘Æ°á»£c táº£i tá»± Ä‘á»™ng bá»Ÿi `sentence-transformers` khi cháº¡y láº§n Ä‘áº§u:

```powershell
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('AITeamVN/Vietnamese_Embedding_v2')"
```

Hoáº·c táº£i trÆ°á»›c báº±ng Hugging Face CLI:

```powershell
huggingface-cli download AITeamVN/Vietnamese_Embedding_v2
```

### 5.3. Cáº¥u hÃ¬nh sá»­ dá»¥ng

```env
EMBED_BACKEND=st
EMBED_ST_MODEL=AITeamVN/Vietnamese_Embedding_v2
EMBED_DIM=1024
EMBED_BATCH_SIZE=64
```

Trong code, model Ä‘Æ°á»£c náº¡p táº¡i:

```text
backend/embed.py
```

Khi `EMBED_BACKEND=st`, há»‡ thá»‘ng dÃ¹ng `SentenceTransformer`, cháº¡y trÃªn GPU `cuda`, vÃ  chuyá»ƒn sang fp16 Ä‘á»ƒ tiáº¿t kiá»‡m VRAM.

## 6. MÃ´ hÃ¬nh reranker

### 6.1. ThÃ´ng tin mÃ´ hÃ¬nh

| Thuá»™c tÃ­nh | GiÃ¡ trá»‹ |
|---|---|
| Model ID | `AITeamVN/Vietnamese_Reranker` |
| Nguá»“n | Hugging Face |
| Loáº¡i mÃ´ hÃ¬nh | Cross-encoder reranker |
| Base model | `BAAI/bge-reranker-v2-m3` |
| NgÃ´n ngá»¯ | Tiáº¿ng Viá»‡t |
| Max sequence length | 2304 token theo model card |
| License | Apache 2.0 |
| Vai trÃ² | Cháº¥m Ä‘iá»ƒm cáº·p `(cÃ¢u há»i, chunk)` vÃ  xáº¿p háº¡ng láº¡i káº¿t quáº£ truy há»“i |

Theo model card cá»§a Hugging Face, `AITeamVN/Vietnamese_Reranker` lÃ  reranker public Ä‘Æ°á»£c fine-tune tá»« `BAAI/bge-reranker-v2-m3` cho tiáº¿ng Viá»‡t. ÄÃ¢y khÃ´ng pháº£i checkpoint do nhÃ³m tá»± fine-tune.

### 6.2. HÆ°á»›ng dáº«n táº£i

Model Ä‘Æ°á»£c táº£i tá»± Ä‘á»™ng khi gá»i `FlagEmbedding.FlagReranker` láº§n Ä‘áº§u:

```powershell
python -c "from FlagEmbedding import FlagReranker; FlagReranker('AITeamVN/Vietnamese_Reranker', use_fp16=True)"
```

Hoáº·c táº£i trÆ°á»›c báº±ng Hugging Face CLI:

```powershell
huggingface-cli download AITeamVN/Vietnamese_Reranker
```

### 6.3. Cáº¥u hÃ¬nh sá»­ dá»¥ng

```env
USE_RERANKER=true
RERANKER_MODEL=AITeamVN/Vietnamese_Reranker
```

Trong code, model Ä‘Æ°á»£c náº¡p táº¡i:

```text
backend/reranker.py
```

Pipeline má»›i nháº¥t Ä‘áº·t máº·c Ä‘á»‹nh:

```python
os.environ.setdefault("RERANKER_MODEL", "AITeamVN/Vietnamese_Reranker")
```

trong cÃ¡c file:

```text
scripts/build_v72_metadata_arch.py
scripts/build_v75_prompt_selector.py
scripts/build_v75_v72_selector_repair.py
```

## 7. Chá»‰ má»¥c khÃ´ng pháº£i checkpoint mÃ´ hÃ¬nh

Má»™t sá»‘ file lá»›n trong repo lÃ  chá»‰ má»¥c hoáº·c dá»¯ liá»‡u phá»¥c vá»¥ retrieval, khÃ´ng pháº£i checkpoint neural model:

| File | Loáº¡i | CÃ³ pháº£i checkpoint model khÃ´ng? | Ghi chÃº |
|---|---|---|---|
| `data/bm25_vbpl_aiteam_meta_parsefix_20260628.pkl` | BM25 index | KhÃ´ng | Cáº§n chia sáº» Ä‘á»ƒ tÃ¡i láº­p retrieval lexical |
| `data/corpus_vbpl_v2_parsefix_20260628/` | Corpus | KhÃ´ng | Dá»¯ liá»‡u nguá»“n Ä‘Ã£ xá»­ lÃ½ |
| Qdrant collection `vbpl_aiteam_meta_parsefix_20260628` | Vector index | KhÃ´ng | CÃ³ thá»ƒ tÃ¡i láº­p tá»« corpus + embedding model |
| `data/ft_rerank_train.jsonl` | Dá»¯ liá»‡u huáº¥n luyá»‡n | KhÃ´ng | KhÃ´ng báº¯t buá»™c náº¿u khÃ´ng fine-tune |

## 8. HÆ°á»›ng dáº«n cháº¡y há»‡ thá»‘ng vá»›i mÃ´ hÃ¬nh hiá»‡n táº¡i

### 8.1. CÃ i Ä‘áº·t phá»¥ thuá»™c

```powershell
pip install -r requirements.txt
pip install sentence-transformers FlagEmbedding torch
```

TÃ¹y mÃ´i trÆ°á»ng GPU, cáº§n cÃ i Ä‘Ãºng phiÃªn báº£n `torch` tÆ°Æ¡ng thÃ­ch CUDA.

### 8.2. Cáº¥u hÃ¬nh biáº¿n mÃ´i trÆ°á»ng

```powershell
$env:LLM_API_KEY="ollama"
$env:LLM_BASE_URL="http://localhost:11434/v1"
$env:LLM_MODEL="qwen-vbpl"
$env:LLM_MAX_TOKENS="2048"
$env:LLM_TEMPERATURE="0.0"
$env:LLM_REASONING_EFFORT="none"

$env:EMBED_BACKEND="st"
$env:EMBED_ST_MODEL="AITeamVN/Vietnamese_Embedding_v2"
$env:EMBED_DIM="1024"

$env:USE_RERANKER="true"
$env:RERANKER_MODEL="AITeamVN/Vietnamese_Reranker"

$env:QDRANT_COLLECTION="vbpl_aiteam_meta_parsefix_20260628"
$env:BM25_INDEX_PATH="data/bm25_vbpl_aiteam_meta_parsefix_20260628.pkl"
```

### 8.3. Cháº¡y pipeline má»›i nháº¥t

```powershell
python scripts/build_v75_v72_selector_repair.py
```

## 9. Checkpoint cáº§n chia sáº»

### 9.1. TrÆ°á»ng há»£p hiá»‡n táº¡i

VÃ¬ há»‡ thá»‘ng hiá»‡n táº¡i khÃ´ng fine-tune mÃ´ hÃ¬nh riÃªng, checkpoint cáº§n chia sáº» Ä‘Æ°á»£c xÃ¡c Ä‘á»‹nh nhÆ° sau:

| NhÃ³m checkpoint | Tráº¡ng thÃ¡i | CÃ¡ch chia sáº» |
|---|---|---|
| Checkpoint LLM ná»™i bá»™ | KhÃ´ng cÃ³ | KhÃ´ng cáº§n upload |
| Checkpoint embedding ná»™i bá»™ | KhÃ´ng cÃ³ | DÃ¹ng checkpoint public trÃªn Hugging Face |
| Checkpoint reranker ná»™i bá»™ | KhÃ´ng dÃ¹ng trong pipeline má»›i nháº¥t | KhÃ´ng cáº§n upload náº¿u khÃ´ng dÃ¹ng cÃ¡c model trong `models/` |
| Checkpoint public | CÃ³ | Chia sáº» báº±ng Ä‘Æ°á»ng link model hub |

### 9.2. Link checkpoint public

| MÃ´ hÃ¬nh | Link táº£i/checkpoint |
|---|---|
| `qwen3.5:9b` | https://ollama.com/library/qwen3.5:9b |
| `AITeamVN/Vietnamese_Embedding_v2` | https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2 |
| `AITeamVN/Vietnamese_Reranker` | https://huggingface.co/AITeamVN/Vietnamese_Reranker |

### 9.3. Link Google Drive bÃ n giao

ÄÆ°á»ng dáº«n Drive dÃ¹ng Ä‘á»ƒ chia sáº» tÃ i liá»‡u vÃ  cÃ¡c artifact cá»§a dá»± Ã¡n:

```text
https://drive.google.com/drive/folders/1NvkAEpOjOqHNqF9kC-FvrOs5fjIiscTj
```

Do khÃ´ng cÃ³ checkpoint fine-tune ná»™i bá»™ trong báº£n váº­n hÃ nh má»›i nháº¥t, Drive khÃ´ng báº¯t buá»™c chá»©a model weights. Drive nÃªn chá»©a:

- `docs/MODEL_DESCRIPTION.md`
- `docs/DATA_DESCRIPTION.md`
- GÃ³i dá»¯ liá»‡u `data/vbpl_data_package_20260630.zip`
- Náº¿u muá»‘n cháº¡y offline hoÃ n toÃ n, cÃ³ thá»ƒ bá»• sung snapshot Hugging Face/Ollama cache, nhÆ°ng Ä‘Ã¢y lÃ  tÃ¹y chá»n, khÃ´ng pháº£i checkpoint do nhÃ³m tá»± huáº¥n luyá»‡n.

## 10. TrÆ°á»ng há»£p cáº§n chia sáº» checkpoint ná»™i bá»™

Chá»‰ cáº§n upload checkpoint ná»™i bá»™ náº¿u há»‡ thá»‘ng Ä‘Æ°á»£c cáº¥u hÃ¬nh Ä‘á»ƒ dÃ¹ng cÃ¡c model trong thÆ° má»¥c `models/`, vÃ­ dá»¥:

```env
RERANKER_MODEL=models/reranker_vbpl_v2
```

Khi Ä‘Ã³ cáº§n chia sáº» toÃ n bá»™ thÆ° má»¥c checkpoint tÆ°Æ¡ng á»©ng, bao gá»“m tá»‘i thiá»ƒu:

- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `sentencepiece.bpe.model` náº¿u cÃ³

Vá»›i báº£n hiá»‡n táº¡i, cÃ¡c checkpoint nÃ y lÃ  káº¿t quáº£ thá»­ nghiá»‡m, khÃ´ng pháº£i thÃ nh pháº§n báº¯t buá»™c Ä‘á»ƒ tÃ¡i láº­p pipeline má»›i nháº¥t.

## 11. TÃ i liá»‡u tham chiáº¿u trong repo

| File | Ná»™i dung liÃªn quan |
|---|---|
| `.env` | Cáº¥u hÃ¬nh LLM/embedding máº·c Ä‘á»‹nh cÅ© |
| `.env.example` | Máº«u cáº¥u hÃ¬nh mÃ´i trÆ°á»ng |
| `scripts/Modelfile.qwen-vbpl` | Cáº¥u hÃ¬nh táº¡o model Ollama `qwen-vbpl` |
| `backend/embed.py` | CÃ¡ch náº¡p embedding model |
| `backend/reranker.py` | CÃ¡ch náº¡p reranker |
| `backend/llm.py` | OpenAI-compatible LLM client |
| `scripts/build_v75_v72_selector_repair.py` | Pipeline má»›i nháº¥t |

## 12. Káº¿t luáº­n bÃ n giao

MÃ´ hÃ¬nh sá»­ dá»¥ng trong há»‡ thá»‘ng hiá»‡n táº¡i Ä‘á»u lÃ  model public hoáº·c model tag local táº¡o tá»« public checkpoint. NhÃ³m khÃ´ng fine-tune vÃ  khÃ´ng phÃ¡t sinh checkpoint model riÃªng cho báº£n váº­n hÃ nh má»›i nháº¥t.

VÃ¬ váº­y, tÃ i liá»‡u bÃ n giao cáº§n nÃªu rÃµ:

- Model nÃ o Ä‘Æ°á»£c sá»­ dá»¥ng.
- Checkpoint public láº¥y tá»« Ä‘Ã¢u.
- CÃ¡ch táº£i vÃ  cáº¥u hÃ¬nh.
- KhÃ´ng cÃ³ checkpoint ná»™i bá»™ cáº§n upload, trá»« khi chuyá»ƒn sang dÃ¹ng cÃ¡c model thá»­ nghiá»‡m trong thÆ° má»¥c `models/`.
