# Deploy app demo lên Hugging Face Spaces (CPU free, luôn online)

App demo (`app/` + React `web/`) chạy trên **HF Spaces CPU free** (16GB RAM, 2 vCPU): FastAPI phục vụ
cả API lẫn React build, gọi **LLM qua API** + **Qdrant Cloud**. Kiến trúc gộp 1 nơi nên không lo CORS.

> Tốc độ: ~10–30s/câu trên CPU. Space **ngủ khi lâu không dùng**, tự thức khi có request (cold-start
> ~30–60s nếu đã bake model vào image; lâu hơn nếu chưa).

---

## 1. Tạo Space
1. Vào https://huggingface.co/new-space
2. **Owner/Name**: tùy chọn (vd `legal-rag-demo`).
3. **SDK**: chọn **Docker** → **Blank**.
4. **Hardware**: **CPU basic (free)**.
5. **Visibility**: Public hoặc Private đều được.

## 2. Front-matter cấu hình Space — ĐÃ CÓ SẴN
`README.md` trong repo **đã có sẵn** khối YAML front-matter (title/emoji/`sdk: docker`/`app_port: 7860`)
ở đầu file, nên khi push lên Space sẽ tự nhận cấu hình. **Không cần sửa tay gì thêm.**

## 3. Đưa code lên Space
Space là một git repo riêng. Từ máy bạn (đang ở nhánh `main`):

```bash
# thêm remote HF (thay <user>/<space>)
git remote add hf https://huggingface.co/spaces/<user>/<space>

# Gói code thành 1 commit SẠCH (nhánh deploy) rồi push — LẶP LẠI MỖI LẦN CẬP NHẬT.
# (main có lịch sử chứa blob corpus >10MB + ảnh PNG/notebook → HF chặn nếu push thẳng main.)
git branch -D deploy
git checkout --orphan deploy
git rm -r --cached docs/images
git rm --cached fast-retrieval.ipynb pipeline-intersect.ipynb piplne-and-debug.ipynb
git commit -m "Deploy NextGen Legal RAG (HF Spaces)"
git push hf deploy:main --force
git checkout -f main
```

- Corpus `data/law_corpus_clean.json` đi kèm qua **Git LFS** (đã bật sẵn trong repo) — HF Spaces hỗ trợ LFS.
- Nhánh `deploy` là orphan 1 commit → không mang lịch sử file to nên HF chấp nhận.
- `bm25_corpus.pkl` KHÔNG đẩy (bị `.gitignore`) — Docker sẽ **tự dựng** trong lúc build.

> Nếu `git push hf` báo cần đăng nhập: tạo **Access Token** tại https://huggingface.co/settings/tokens
> (quyền *write*) rồi dùng làm mật khẩu khi push (username = tên HF của bạn).

## 4. Đặt Secrets (khóa bí mật) cho Space
Vào **Settings → Variables and secrets** của Space, thêm **Secrets**:

| Tên | Giá trị |
|---|---|
| `QDRANT_URL` | URL Qdrant Cloud của bạn |
| `QDRANT_API_KEY` | API key Qdrant |
| `LLM_API_BASE_URL` | `https://api.groq.com/openai/v1` (Groq) |
| `LLM_API_KEY` | API key Groq |
| `LLM_API_MODEL` | `llama-3.1-8b-instant` (Groq, <14B) |
| `SUPABASE_URL` | `https://<ref>.supabase.co` (đăng nhập + lịch sử) |
| `SUPABASE_ANON_KEY` | anon public key (công khai được) |

HF tự đưa các secret này thành **biến môi trường** → `config/settings.py` đọc được ngay. Đổi secret xong
bấm **Restart** Space.

## 5. Chờ build & chạy
- Tab **Logs** của Space: xem quá trình build Docker (cài deps, build React, tải model, dựng BM25).
- Lần đầu build **khá lâu** (tải ~2.6GB model + dựng BM25). Khi log hiện `✅ [RagService] Sẵn sàng phục vụ.`
  và trạng thái Space là **Running** → mở link Space là dùng được.
- Link demo chính là URL của Space (dạng `https://<user>-<space>.hf.space`).

---

## Cập nhật về sau
Sửa code xong chỉ cần:
```bash
git push hf feature/web-react-deploy:main
```
Space tự build lại.

## Xử lý sự cố
- **Build timeout / quá lâu ở bước tải model:** mở `Dockerfile`, xóa 2 dòng `RUN python -c "...SentenceTransformer..."`
  và `RUN python -m src.index_bm25`. App sẽ tải model + dựng BM25 lúc khởi động thay vì lúc build
  (build nhanh hơn, nhưng cold-start lần đầu lâu hơn).
- **Trích dẫn hiện được nhưng câu trả lời báo lỗi:** sai `LLM_API_*` trong Secrets.
- **Retrieval rỗng / lỗi Qdrant:** kiểm tra `QDRANT_URL`/`QDRANT_API_KEY`.
- **Hết RAM:** đảm bảo đang dùng CPU basic free (16GB); pipeline này vừa đủ. Không bật thêm tiến trình nặng khác.

## Chạy thử tại chỗ (tùy chọn, cần Docker)
```bash
docker build -t legal-rag-demo .
docker run --rm -p 7860:7860 \
  -e QDRANT_URL=... -e QDRANT_API_KEY=... \
  -e LLM_API_BASE_URL=... -e LLM_API_KEY=... -e LLM_API_MODEL=... \
  legal-rag-demo
# mở http://localhost:7860
```
