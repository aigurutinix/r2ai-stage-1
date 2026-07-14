# NextGen — Tổng quan Pipeline hệ thống (Câu hỏi → Kết quả)

Trợ lý pháp luật cho doanh nghiệp nhỏ và vừa, dùng **RAG (Retrieval-Augmented Generation)**:
truy hồi văn bản pháp luật chính thức → rerank → LLM sinh câu trả lời **kèm trích dẫn kiểm chứng được**.

## Sơ đồ luồng tổng quát

```
                          ┌───────────────────────────── FRONTEND (React) ─────────────────────────────┐
   Người dùng ── câu hỏi ─►  Giao diện chat  ──(POST /ask, SSE)──►                                        │
   (web/mobile)          │      ▲   ▲                                                                    │
                         │      │   └── (2) citations tới TRƯỚC (~2s): panel "Văn bản liên quan"         │
                         │      └────── (3) answer STREAM dần từng chữ                                    │
                         └────────────────────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼  BACKEND (FastAPI)
   ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
   │ ① Mở rộng truy vấn   ② Embedding câu hỏi        ③ TRUY HỒI LAI (Hybrid)          ④ RERANK        │
   │  Multi-Query   ─►   AITeamVN 1024-d   ─►   ┌── Dense: Qdrant (law_2026) ─┐  ─►  bge-reranker    │
   │  (3 biến thể)                              └── Sparse: BM25 (underthesea)─┘      -large (top-5)  │
   │                                              hợp nhất bằng RRF (top-30)                          │
   └───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                     │
                        ┌────────────────────────────┼─────────────────────────────┐
                        ▼                                                            ▼
   ⑤ DỰNG TRÍCH DẪN (citations)                              ⑥ SINH CÂU TRẢ LỜI (LLM, streaming)
   - map số hiệu → URL (doc_url_map.json)                    - đóng gói ngữ cảnh (≤12k ký tự)
   - tên chuẩn (law_manifest.json)                           - LLM < 14B qua API (Groq llama-3.1-8b)
   - đoạn trích + Điều X                                     - bám ngữ cảnh, dẫn "Theo Điều X..."
                        │                                                            │
                        └───────────────► gộp về frontend ◄──────────────────────────┘
                                                     │
                                     ⑦ Hiển thị + LƯU LỊCH SỬ (nếu đã đăng nhập)
                                        Supabase (Google OAuth + Postgres + RLS)
```

## Chi tiết từng giai đoạn

**① Mở rộng truy vấn (Multi-Query Expansion)**
Câu hỏi gốc được sinh thêm ~3 biến thể (thay thế từ đồng nghĩa theo miền pháp lý: viết tắt SME, thuật ngữ thuế…) để tăng độ phủ khi truy hồi.

**② Vector hóa câu hỏi (Embedding)**
Mỗi biến thể được embed bằng **`AITeamVN/Vietnamese_Embedding`** (vector **1024 chiều**) — đúng model đã dùng để index corpus, nên câu hỏi và văn bản nằm chung không gian ngữ nghĩa.

**③ Truy hồi lai (Hybrid Retrieval)** — trái tim độ chính xác
- **Dense (ngữ nghĩa):** tìm trên **Qdrant Cloud** (collection `law_2026`, ~30 nghìn đoạn văn bản pháp luật đã chia nhỏ) bằng truy vấn 2 giai đoạn (prefetch + rescore).
- **Sparse (từ khóa):** **BM25** (tách từ tiếng Việt bằng `underthesea` + từ điển đồng nghĩa/viết tắt pháp lý) — bắt đúng số hiệu, thuật ngữ.
- Hai nguồn hợp nhất bằng **RRF (Reciprocal Rank Fusion)** → danh sách ~**30 ứng viên** tốt nhất.

**④ Rerank (Cross-Encoder)**
**`BAAI/bge-reranker-large`** chấm lại độ liên quan của từng cặp (câu hỏi, đoạn văn bản), lọc theo ngưỡng và giữ **top-5** đoạn liên quan nhất → tín hiệu đáng tin nhất để trích dẫn.

**⑤ Dựng trích dẫn (Citations)** — điểm khác biệt chống ảo giác
Từ các đoạn đã rerank, hệ thống dựng thẻ trích dẫn: **tên văn bản** (chuẩn hóa qua `law_manifest.json`), **số hiệu**, **Điều X**, **đoạn trích gốc**, và **link** tới văn bản trên Thư viện Pháp luật (map qua `doc_url_map.json`, 597 văn bản). Gửi về giao diện **ngay (~2s)** — người dùng thấy căn cứ trước cả khi câu trả lời viết xong.

**⑥ Sinh câu trả lời (LLM, streaming)**
Các đoạn liên quan được đóng gói thành ngữ cảnh (≤ 12.000 ký tự) đưa vào **LLM dưới 14 tỷ tham số** (tuân thủ quy định BTC — hiện dùng `llama-3.1-8b` qua API tương thích OpenAI). LLM **chỉ trả lời dựa trên ngữ cảnh**, nêu căn cứ dạng *"Theo quy định tại Điều X…"*, và **stream từng chữ** về giao diện.

**⑦ Hiển thị + Lưu lịch sử**
Frontend hiển thị câu trả lời (markdown) + panel văn bản. Nếu người dùng **đăng nhập Google** (qua **Supabase Auth**), toàn bộ hội thoại được lưu vào **Postgres (Supabase)** với **RLS** (mỗi người chỉ thấy dữ liệu của mình) và hiện lại trong sidebar lịch sử.

## Ngăn xếp công nghệ

| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| Embedding | AITeamVN/Vietnamese_Embedding (1024-d) | Vector hóa câu hỏi & corpus |
| Vector DB | Qdrant Cloud (`law_2026`, ~30k đoạn) | Tìm kiếm ngữ nghĩa (dense) |
| Sparse | BM25 + underthesea | Tìm theo từ khóa/số hiệu |
| Hợp nhất | RRF | Gộp dense + sparse |
| Rerank | BAAI/bge-reranker-large | Chấm lại, giữ top liên quan |
| LLM | Model **< 14B** qua API (Groq) | Sinh câu trả lời tiếng Việt |
| Trích dẫn | law_manifest.json + doc_url_map.json | Tên chuẩn + link văn bản gốc |
| Auth + Lịch sử | Supabase (Google OAuth + Postgres/RLS) | Người dùng + lưu hội thoại |
| Backend / Frontend | FastAPI (SSE) + React (Vite) | API + giao diện streaming |
| Hạ tầng | Hugging Face Spaces (Docker, free) | Host online 24/7 |

## Điểm mạnh (đưa lên slide)
- **Có căn cứ, kiểm chứng được:** mỗi câu trả lời kèm Điều luật + link văn bản gốc → chống ảo giác.
- **Truy hồi lai:** kết hợp ngữ nghĩa + từ khóa → phủ tốt cả câu hỏi diễn giải lẫn tra số hiệu.
- **Tuân thủ quy định:** LLM < 14 tỷ tham số.
- **Trải nghiệm nhanh:** trích dẫn hiện ~2s, câu trả lời stream dần.
- **Sẵn sàng thương mại:** đăng nhập, lưu lịch sử, mở rộng lĩnh vực = chỉ cần nạp thêm văn bản.
```
