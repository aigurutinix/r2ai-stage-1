# Đánh giá hệ thống — RAG tra cứu & hỏi đáp VBPL (DN/SME)

> Tài liệu này mô tả **kiến trúc, dữ liệu, phương pháp đánh giá và kết quả** của
> hệ thống, đồng thời ghi lại **cách lấy/dựng mô hình** (theo yêu cầu thể lệ
> §3 — "đưa thông tin về cách thức lấy mô hình vào bài báo").

## 1. Kiến trúc hệ thống

| Thành phần | Lựa chọn | Ghi chú |
|---|---|---|
| **Sinh câu trả lời (LLM)** | `qwen3.5:9b` qua **Ollama** (`qwen-vbpl`, num_ctx 16384, reasoning off) | Local, OpenAI-compatible `:11434/v1` |
| **Embedding** | **BAAI/bge-m3** (1024-dim) qua Ollama | Mở, miễn phí, đa ngữ; tốt cho tiếng Việt |
| **Vector store** | **Qdrant embedded** (`file://`, persistent) | Không cần Docker; brute-force đủ cho ~250k vector |
| **Pipeline** | RAG: embed query → retrieve top-k (hybrid filter→vector) → prompt ép trích `Điều X` → LLM |

Lý do chọn **bge-m3** thay vì `text-embedding-3-large`: chất lượng truy hồi
tiếng Việt cao (MIRACL nDCG@10 ~67.8 dense vs 54.9 của OpenAI), **chi phí $0**
(chạy local), không phụ thuộc API ngoài.

### Cách lấy / dựng mô hình (tái lập)
```powershell
# 1. Ollama + models
ollama pull qwen3.5:9b
ollama pull bge-m3
ollama create qwen-vbpl -f scripts/Modelfile.qwen-vbpl   # tăng context, tắt thinking

# 2. Cấu hình .env (LLM + EMBED đều trỏ Ollama; Qdrant embedded file://)
#    QDRANT_URL=file:///.../data/qdrant_local ; EMBED_MODEL=bge-m3 ; EMBED_DIM=1024
#    INGEST_SCOPE_FILTER=true ; INGEST_LIMIT=0

# 3. Ingest corpus DN/SME
python -m ingest.run --recreate
```

## 2. Dữ liệu

- **Nguồn**: HuggingFace `th1nhng0/vietnamese-legal-documents` (metadata 153,420
  bản · content 178,665 bản).
- **Lọc phạm vi** (`ingest/scope.py`): chỉ giữ **văn bản quy phạm lõi** (Luật, Bộ
  luật, Nghị định, Thông tư, Pháp lệnh, VBHN, Nghị quyết...) **khớp từ khoá DN/SME**
  (doanh nghiệp, đầu tư, thuế, lao động, hợp đồng, kế toán, phá sản, cạnh tranh...).
  → **153,420 → 20,321 docs** (18,792 có cả metadata + content).

### Chất lượng dữ liệu (đã kiểm chứng)
- **~14.9%** bản trong scope là **vỏ HTML rỗng** (không có nội dung) → tự bỏ khi ingest.
- Dataset có **bản trùng `so_ky_hieu`** (1 rỗng + 1 đầy đủ) — pipeline parse bỏ bản
  rỗng, chunk bản đầy đủ. Các luật trọng tâm đều có **full text** (Luật DN 2020:
  324k ký tự / 213 điều; BLLĐ 2012: 212 điều; Đầu tư 2020: 75 điều; ...).
- **Hạn chế đã biết**: một số luật bản **mới nhất** (BLLĐ 2019, Quản lý thuế 2019)
  là bản rỗng; bản có nội dung là phiên bản cũ hơn (2012, 2006).

### Chunking & index
- **1 Điều = 1 chunk** (đơn vị semantic tự nhiên của VBPL); Điều quá dài tách theo Khoản.
- Payload giữ `so_ky_hieu`, `dieu_so`, `loai_van_ban`, `tinh_trang_hieu_luc`, `title`...
- **Kết quả ingest: 250,765 vector** (bge-m3 1024-dim) trong collection `vbpl_bge_m3`.
  Thời gian embed ~3h15 trên RTX 3060 (local).

## 3. Phương pháp đánh giá

BTC giữ kín đáp án → ta tự dựng **bộ đánh giá có nhãn vàng bám corpus**
(`tests/build_eval_set.py`):
- Lấy các **Điều có thật** trong **10 luật trọng tâm DN/SME** (~1,200 điều khả dụng).
- `qwen` sinh **câu hỏi tự nhiên** mà điều đó trả lời được (không nhắc số điều/số hiệu).
- **Gold = (so_ky_hieu, Điều)** của chính điều nguồn → nhãn đúng theo cấu trúc.
- **60 câu** (6/luật × 10 luật).

> Hạn chế: nhãn vàng là **1 điều/câu** (điều nguồn) → **Precision bị chặn trần ~1/k**
> khi truy hồi top-k; thực tế thi có nhiều điều liên quan/câu nên Precision sẽ cao hơn.
> Vì vậy chỉ số trọng tâm ở đây là **Recall** (đúng tinh thần F2 của thể lệ).

**Chỉ số** (`tests/metrics.py`, đúng công thức thể lệ §4.1 — đã unit-test 10/10):
`Precision = đúng/đã-truy-hồi`, `Recall = đúng/liên-quan`, `F2 = 5PR/(4P+R)` (macro).

**Hai lớp đo** (`tests/f2_eval.py`):
- **Track A — End-to-end**: rút `Điều X` từ `answer` do LLM sinh → so gold. *(đúng cách BTC chấm)*
- **Track B — Retrieval ceiling**: tập `Điều X` trong top-k hit → so gold. *(trần truy hồi)*

## 4. Kết quả

### 4.1 Truy hồi (Track B, top_k=10) — `n=60`

| Chỉ số | Giá trị | Ý nghĩa |
|---|---:|---|
| **Recall (Điều)** | **0.917** | điều gold nằm trong top-10 ở 92% câu |
| **Pair-recall (so_ky_hieu+Điều)** | **0.867** | đúng cả luật + điều ở 87% câu |
| **MRR (theo cặp)** | **0.685** | điều gold thường ở hạng #1–2 |
| Precision | 0.125 | (trần ~1/k do gold 1-điều) |
| F2 | 0.393 | bị kéo bởi precision trần |
| Latency | 2.29s/câu | brute-force 250k vector (embedded) |

→ **Khâu truy hồi rất mạnh**: bge-m3 + corpus DN/SME tìm đúng căn cứ **~92%**.

### 4.2 End-to-end (Track A, điểm thi thực tế) — `n=60`, top_k=10

| Track | Precision | Recall | **F2** |
|---|---:|---:|---:|
| **A. End-to-end** (rút `Điều X` từ answer LLM) | 0.392 | **0.883** | **0.654** |
| B. Retrieval ceiling (top-10) | 0.125 | 0.917 | 0.393 |

**Nhận xét:**
1. **Track A F2 (0.654) > Track B (0.393)**: LLM *cải thiện* so với truy hồi thô
   bằng cách chỉ trích **một tập điều chính xác** (~2–4 điều) thay vì cả 10 →
   precision tăng (0.39 vs 0.13) trong khi recall gần như giữ nguyên. Đây đúng là
   việc một câu trả lời RAG tốt cần làm.
2. **Recall hầu như không giảm A↔B (0.883 vs 0.917)**: LLM trích đúng điều gold ở
   ~96% trường hợp truy hồi tìm được → **grounding tốt, chống bịa hiệu quả** (chỉ
   mất ~3.7% do khâu sinh).
3. **7/60 câu F2=0**: 5 do **truy hồi trượt** (điều gold không lọt top-10) + 2 do
   khâu sinh ở các câu "phạm vi điều chỉnh" (Điều 1) chung chung → LLM trích điều
   nội dung khác. Hướng cải thiện: tăng `top_k`, thêm rerank/hybrid.

> **Bối cảnh điểm số**: nhãn vàng ở đây là **1 điều/câu** nên Precision bị chặn;
> đề thi thật có nhiều điều liên quan/câu → Precision (và F2) thường **cao hơn**.
> Chỉ số đáng tin nhất để so sánh nội bộ là **Recall = 0.883** và **trần truy hồi
> 0.917**.

### 4.3 Độ nhạy theo `top_k` (retrieval ceiling)

| top_k | Recall (Điều) | Pair-recall | MRR |
|---:|---:|---:|---:|
| 10 | 0.917 | 0.867 | 0.685 |
| 20 | 0.933 | 0.933 | 0.690 |

Tăng `top_k` 10→20 chỉ thêm ~1.6% recall điều (các câu trượt còn lại là **trượt
ngữ nghĩa thật**, không phải do thiếu độ sâu) nhưng giúp **pair-recall** rõ hơn
(+6.6%, lấy đúng *luật + điều* thay vì trùng số điều ở luật khác). `top_k=10` là
điểm vận hành cân bằng (recall cao, context LLM gọn). Hướng nâng tiếp: rerank /
hybrid sparse-dense (bge-m3 hỗ trợ sẵn) thay vì chỉ tăng độ sâu.

## 5. Tái lập đánh giá
```powershell
$env:PYTHONPATH="."; $env:PYTHONUTF8="1"
python -m tests.build_eval_set --per-law 6          # sinh data/eval_set.json
python -m tests.f2_eval --top-k 10 --retrieval-only # Track B (nhanh)
python -m tests.f2_eval --top-k 10                  # Track A + B (có LLM)
# Báo cáo: data/f2_eval_report.md · data/f2_eval_results.json
```
