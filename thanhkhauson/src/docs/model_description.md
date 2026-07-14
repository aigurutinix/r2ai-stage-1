# Mô Tả Model Và Hướng Dẫn Tải Checkpoint

Trạng thái: bản làm việc để bổ sung vào bộ tài liệu thuyết minh sản phẩm.

## 1. Tổng Quan Model

| Thành phần | Model card | Đường dẫn local | Vai trò trong hệ thống |
| --- | --- | --- | --- |
| LLM | https://huggingface.co/Qwen/Qwen3.5-9B | `model_cache/Qwen3.5-9B` | Lập kế hoạch truy hồi, lọc ngữ cảnh, sinh câu trả lời pháp lý qua endpoint OpenAI-compatible |
| Reranker | https://huggingface.co/namdp-ptit/ViRanker | `model_cache/ViRanker` | Rerank các chunk trả về từ Elasticsearch/Qdrant |
| Embedding | https://huggingface.co/mainguyen9/vietlegal-harrier-0.6b | `model_cache/vietlegal-harrier-0.6b` | Tạo embedding truy vấn và chunk cho dense retrieval |
| Vietnamese word segmentation | https://pypi.org/project/py-vncorenlp/ | `model_cache/vncorenlp` | Tách từ tiếng Việt cho trường `content_search` của BM25 |

Thông tin metadata kiểm tra từ HuggingFace API ngày 2026-06-30:

| Model | Created at | Last modified | Library | Pipeline/tag chính | License tag |
| --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-9B` | `2026-02-27T12:58:26Z` | `2026-03-02T00:51:43Z` | `transformers` | `image-text-to-text` | `apache-2.0` |
| `namdp-ptit/ViRanker` | `2024-08-14T02:58:28Z` | `2025-06-16T02:21:36Z` | `transformers` | `text-classification`, `cross-encoder`, `rerank`, `vi` | `apache-2.0` |
| `mainguyen9/vietlegal-harrier-0.6b` | `2026-04-04T06:07:52Z` | `2026-04-05T04:40:22Z` | `sentence-transformers` | `sentence-similarity`, `retrieval`, `legal`, `vietnamese` | Chưa thấy license tag trong API summary |

Lưu ý điều kiện cuộc thi trong `AGENTS.md`: model dùng cho bản nộp cuối cần là open/public, nhỏ hơn 14B parameters và phát hành trước `2026-03-01` theo giờ Việt Nam. `Qwen/Qwen3.5-9B` và `ViRanker` cần lưu bằng chứng model card/source. `vietlegal-harrier-0.6b` có `createdAt` trên HuggingFace sau mốc này nên cần xác minh lại với ban tổ chức hoặc thay bằng checkpoint hợp lệ nếu áp dụng nghiêm ngặt điều kiện ngày phát hành.

## 2. Cài Dependencies

Từ thư mục gốc dự án:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Gói VnCoreNLP dùng trong code là module import `py_vncorenlp`, package PyPI là `py-vncorenlp` tại https://pypi.org/project/py-vncorenlp/. Trang PyPI trỏ tới wrapper source https://github.com/thelinhbkhn2014/VnCoreNLP_Wrapper. Dependency trong `requirements.txt`:

```text
py-vncorenlp==0.1.4
```

VnCoreNLP cần Java. Kiểm tra:

```bash
java -version
python -m pip show py-vncorenlp
```

Nếu chạy Qwen trên máy cục bộ bằng vLLM như `start_eval.sh`, cần cài vLLM phù hợp với CUDA/PyTorch của máy:

```bash
python -m pip install vllm
```

## 3. Tải Checkpoint Từ HuggingFace

Cách khuyến nghị là dùng HuggingFace CLI. Nếu chưa có:

```bash
python -m pip install -U huggingface-hub
```

Tải 3 checkpoint vào đúng vị trí mà mã nguồn đang đọc:

```bash
mkdir -p model_cache

huggingface-cli download Qwen/Qwen3.5-9B \
  --local-dir model_cache/Qwen3.5-9B

huggingface-cli download namdp-ptit/ViRanker \
  --local-dir model_cache/ViRanker

huggingface-cli download mainguyen9/vietlegal-harrier-0.6b \
  --local-dir model_cache/vietlegal-harrier-0.6b
```

Nếu checkpoint được chia sẻ qua Google Drive/OneDrive, giải nén hoặc đặt thư mục vào cùng các đường dẫn trên:

```text
model_cache/
  Qwen3.5-9B/
  ViRanker/
  vietlegal-harrier-0.6b/
  vncorenlp/
```

## 4. Tải Và Khởi Tạo VnCoreNLP

Code trong `Agents/tools/search_legal.py` sẽ tự tải model VnCoreNLP nếu `model_cache/vncorenlp/models` chưa tồn tại. Có thể tải thủ công trước:

```bash
python - <<'PY'
import os
import py_vncorenlp

save_dir = "model_cache/vncorenlp"
os.makedirs(save_dir, exist_ok=True)
py_vncorenlp.download_model(save_dir=save_dir)

rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)
print(rdrsegmenter.word_segment("Doanh nghiệp nhỏ và vừa được hỗ trợ như thế nào?"))
PY
```

## 5. Cấu Hình Đường Dẫn Model

File `config.json` và `Agents/config.py` sử dụng các đường dẫn tương đối sau:

```json
{
  "reranker_path": "model_cache/ViRanker",
  "embedding_model_path": "model_cache/vietlegal-harrier-0.6b",
  "llm_model": "qwen3.5-9b",
  "llm_base_url": "http://localhost:8000/v1",
  "vncorenlp_dir": "model_cache/vncorenlp"
}
```

Có thể ghi đè bằng biến môi trường khi chạy trên máy khác:

```bash
export LLM_MODEL=qwen3.5-9b
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_API_KEY=EMPTY
```

## 6. Sử Dụng Qwen3.5-9B

Dự án kết nối LLM qua `langchain_openai.ChatOpenAI`, endpoint mặc định là `http://localhost:8000/v1`. Script `start_eval.sh` serve Qwen bằng vLLM:

```bash
bash start_eval.sh
```

Script mặc định dùng GPU `1`. Có thể đổi GPU:

```bash
GPU_ID=0 bash start_eval.sh
```

Kiểm tra endpoint:

```bash
curl http://localhost:8000/v1/models
```

Nếu thiếu VRAM, giảm `--max-model-len`, giảm `--gpu-memory-utilization` trong `start_eval.sh`, hoặc chuyển sang GPU có bộ nhớ lớn hơn.

## 7. Sử Dụng Embedding Model

Embedding model được nạp trong `Agents/tools/search_legal.py` bằng `SentenceTransformer`:

```python
import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "model_cache/vietlegal-harrier-0.6b",
    model_kwargs={"torch_dtype": torch.float32},
)
model.max_seq_length = 512

query = "Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\nQuery: điều kiện hỗ trợ doanh nghiệp nhỏ và vừa"
embedding = model.encode([query], normalize_embeddings=True)
```

Vector index Qdrant dùng collection `legal_chunks` với cosine distance và vector size `1024`.

## 8. Sử Dụng Reranker

ViRanker được nạp bằng Transformers:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("model_cache/ViRanker")
model = AutoModelForSequenceClassification.from_pretrained("model_cache/ViRanker").to(device)
model.eval()

pairs = [["câu hỏi pháp lý", "đoạn văn bản pháp lý cần xếp hạng"]]
inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(device)
with torch.no_grad():
    scores = model(**inputs).logits.squeeze(-1)
print(scores)
```

Trong pipeline hiện tại, số lượng kết quả trước rerank là `RETRIEVER_TOP_K=15`, sau rerank là `RERANKER_TOP_K=4`.
