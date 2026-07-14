# Mô hình sử dụng (Models & Checkpoints)

Tài liệu này mô tả toàn bộ mô hình AI mà hệ thống **Legal-RAG-Vietnam** (pipeline INTERSECT) sử dụng,
kèm **phiên bản checkpoint chính xác** (commit hash trên HuggingFace Hub) và hướng dẫn tải/sử dụng.

> **Lưu ý quan trọng:** Hệ thống **KHÔNG fine-tune / huấn luyện lại** bất kỳ mô hình nào.
> Toàn bộ là **checkpoint pretrained công khai** tải trực tiếp từ HuggingFace Hub.
> Vì vậy, "phiên bản checkpoint" ở đây = **model ID + commit hash (revision)** của bản đã dùng để tạo ra kết quả nộp.
> Các checkpoint được **tải tự động** khi chạy pipeline lần đầu (qua `from_pretrained` / `SentenceTransformer`);
> không cần upload file trọng số thủ công.

---

## 1. Tổng quan 3 mô hình + 1 artifact dữ liệu

| # | Vai trò trong pipeline | Mô hình (HuggingFace ID) | Phiên bản checkpoint (commit SHA) | Cách nạp |
|---|------------------------|--------------------------|-----------------------------------|----------|
| 1 | **Embedding** – vector hóa câu hỏi & corpus (dense retrieval) | [`AITeamVN/Vietnamese_Embedding`](https://huggingface.co/AITeamVN/Vietnamese_Embedding) | `dea33aa1ab339f38d66ae0a40e6c40e0a9249568` | `sentence-transformers`, FP16/FP32, dim **1024** |
| 2 | **Reranker** – cross-encoder chấm lại (query, passage) | [`BAAI/bge-reranker-large`](https://huggingface.co/BAAI/bge-reranker-large) | `55611d7bca2a7133960a6d3b71e083071bbfc312` | `sentence-transformers.CrossEncoder` |
| 3 | **LLM** – sinh câu trả lời tiếng Việt | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | `a09a35458c702b33eeacc393d103063234e8bc28` | `transformers`, **lượng tử hóa 4-bit NF4** (bitsandbytes) |
| 4 | **Vector index** (artifact tự tạo, không phải model) | Qdrant Cloud — collection `law_2026` | — (snapshot chia sẻ qua link) | `qdrant-client` |

Nơi khai báo trong mã nguồn: [`config/settings.py`](../config/settings.py)
(`EMBEDDING_MODEL` dòng 25, `RERANKER_MODEL` dòng 42, `LLM_MODEL_NAME` dòng 59).

---

## 2. Chi tiết từng checkpoint

### 2.1. Embedding — `AITeamVN/Vietnamese_Embedding`
- **Phiên bản (revision):** `dea33aa1ab339f38d66ae0a40e6c40e0a9249568`
- **Loại:** Sentence-embedding tiếng Việt, số chiều vector = **1024** (khớp `Settings.EMBEDDING_DIM`).
- **Dùng để:** Embed câu hỏi và toàn bộ corpus pháp lý; vector lưu trong Qdrant Cloud (`law_2026`).
- **Tải/sử dụng:**
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer(
      "AITeamVN/Vietnamese_Embedding",
      revision="dea33aa1ab339f38d66ae0a40e6c40e0a9249568",
      device="cuda",
  )
  vec = model.encode("Doanh nghiệp nhỏ và vừa được hỗ trợ gì?")
  ```

### 2.2. Reranker — `BAAI/bge-reranker-large`
- **Phiên bản (revision):** `55611d7bca2a7133960a6d3b71e083071bbfc312`
- **Loại:** Cross-encoder reranker.
- **Dùng để:** Chấm lại các ứng viên sau hybrid retrieval, giữ pool top-K đưa cho LLM
  (`Settings.RERANKER_THRESHOLD = 0.30`, pool theo `--pool-k`).
- **Tải/sử dụng:**
  ```python
  from sentence_transformers import CrossEncoder
  reranker = CrossEncoder(
      "BAAI/bge-reranker-large",
      revision="55611d7bca2a7133960a6d3b71e083071bbfc312",
      max_length=512,
  )
  score = reranker.predict([("câu hỏi", "đoạn văn bản pháp lý")])
  ```

### 2.3. LLM — `Qwen/Qwen2.5-7B-Instruct`
- **Phiên bản (revision):** `a09a35458c702b33eeacc393d103063234e8bc28`
- **Loại:** LLM instruct 7B; nạp **4-bit NF4** qua `bitsandbytes` để vừa GPU Kaggle T4 (xem [`src/answer_generator.py`](../src/answer_generator.py)).
- **Cấu hình lượng tử hóa:** `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`,
  `bnb_4bit_compute_dtype=float16`, `bnb_4bit_use_double_quant=True`, `device_map="auto"`.
- **Tham số sinh:** `temperature=0.1`, `top_p=0.85`, `repetition_penalty=1.1`, `max_new_tokens=1024`
  (xem `config/settings.py`).
- **Tải/sử dụng:**
  ```python
  import torch
  from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
  rev = "a09a35458c702b33eeacc393d103063234e8bc28"
  bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
  tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", revision=rev, trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", revision=rev,
              quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
  ```

### 2.4. Vector index — Qdrant Cloud collection `law_2026`
- **Bản chất:** Không phải model — là **artifact dữ liệu tự tạo** (toàn bộ corpus đã embed bằng model 2.1).
- **Thông số:** collection `law_2026`, vector 1024 chiều, ~28.000 đoạn (chunk) văn bản pháp lý.
- **Chia sẻ:** Snapshot Qdrant (`.snapshot`) hoặc file `data/law_corpus_clean.json` (75MB) qua Google Drive/OneDrive — xem mục "Dữ liệu" trong README.
- **Tái dựng:** Nếu có file gốc, dùng [`export_corpus.py`](../export_corpus.py) để export lại từ Qdrant; BM25 index dựng lại từ `law_corpus_clean.json` qua `src/index_bm25.py`.

---

## 3. Cách tải toàn bộ checkpoint

### Cách A — Tự động (khuyến nghị)
Chỉ cần cài dependencies rồi chạy pipeline; HuggingFace `transformers`/`sentence-transformers`
sẽ tự tải đúng checkpoint về cache (`~/.cache/huggingface`) ở lần chạy đầu:
```bash
pip install -r requirements.txt
python fast_retrieval.py --input data/R2AIStage1DATA.json --llm-answer --pool-k 8 --max-select 5
```

### Cách B — Tải thủ công đúng phiên bản (để pin chính xác)
```bash
pip install -U "huggingface_hub[cli]"

huggingface-cli download AITeamVN/Vietnamese_Embedding \
    --revision dea33aa1ab339f38d66ae0a40e6c40e0a9249568

huggingface-cli download BAAI/bge-reranker-large \
    --revision 55611d7bca2a7133960a6d3b71e083071bbfc312

huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --revision a09a35458c702b33eeacc393d103063234e8bc28
```

> Dung lượng tải ước tính: Embedding ~1.3GB, Reranker ~1.3GB, LLM ~15GB (FP16; sau khi nạp 4-bit còn ~5–6GB VRAM).

---

## 4. Yêu cầu phần cứng để nạp checkpoint
- **GPU:** ≥ 16GB VRAM (Kaggle T4 16GB / P100 16GB). LLM nạp 4-bit + reranker + embedding chạy đồng thời vừa đủ 1× T4.
- **Đĩa:** ~20GB trống cho cache HuggingFace.
- **Mạng:** Lần chạy đầu cần Internet để tải checkpoint và kết nối Qdrant Cloud.
