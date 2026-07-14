# RUNBOOK — Khởi động lại hệ thống sau khi restart máy

> Ghi 2026-06-15. Dùng khi bật lại máy (đã chạy liên tục 1 tuần → restart).
> **Hiện KHÔNG có job nền nào đang chạy** (Phase D đã tắt) → restart AN TOÀN, không mất gì.

## Service đang chạy (cần bật lại sau restart)

| Service | Là gì | Cổng | Trạng thái trước restart |
|---|---|---|---|
| **Docker Desktop** | host cho Qdrant | — | đang chạy (4 proc) |
| **Qdrant** (container `qdrant`) | vector store, collection `vbpl_aiteam` ~285k chunk | 6333-6334 | Up 2 ngày |
| **Ollama** | LLM `qwen-vbpl` (judge + decompose + answer-gen) | 11434 | đang chạy (PID cũ 16032) |

> Phần lớn Windows tự bật lại **Docker Desktop** và **Ollama** (chạy nền/tray) khi đăng nhập.
> Qdrant container **KHÔNG tự start** — phải start tay (xem dưới).

## Các bước bật lại (theo thứ tự)

```powershell
# 1. Đảm bảo Docker Desktop đã chạy (mở app nếu chưa). Rồi start container Qdrant:
docker start qdrant

# 2. Kiểm tra Qdrant sống + đúng collection (~285k điểm):
docker ps                                    # phải thấy qdrant Up, cổng 6333
curl http://localhost:6333/collections       # phải có "vbpl_aiteam"

# 3. Kiểm tra Ollama + model:
curl http://localhost:11434/api/tags         # phải có "qwen-vbpl"
#   (nếu Ollama chưa chạy: mở app Ollama, hoặc chạy `ollama serve`)
```

## Biến môi trường khi chạy script Python (BẮT BUỘC)

```powershell
$env:PYTHONUTF8=1; $env:PYTHONPATH="."
$env:QDRANT_COLLECTION="vbpl_aiteam"; $env:QDRANT_URL="http://localhost:6333"
$env:EMBED_BACKEND="st"; $env:EMBED_ST_MODEL="AITeamVN/Vietnamese_Embedding_v2"
$env:HYBRID_SEARCH="true"; $env:USE_HNSW="false"; $env:USE_RERANKER="true"
$env:BM25_INDEX_PATH="data/bm25_vbpl_aiteam.pkl"
```
(Bash: dùng `PYTHONUTF8=1 PYTHONPATH=. QDRANT_COLLECTION=vbpl_aiteam ... python ...`)

## Trạng thái công việc hiện tại (để nhớ khi quay lại)
- **Best submission: v20** (ART_F2=0.5985) — chốt làm baseline.
- **v22 (phân rã sâu) THẤT BẠI** −0.094 → đã loại. Xem `POST_SUBMISSION_REVIEW.md` Review #5.
- **Đang thử "phân rã VỪA"**: `backend/decompose_mid.py` (prompt bảo thủ, chỉ tiếng Việt, retry chặn CJK).
  Mẫu 20 câu đã sinh: `data/subqueries_mid_sample.json`. Chưa sinh full / chưa retrieve / chưa nộp.
- **Việc tiếp (task #18, #19):** sửa judge giữ điều luật-khung; fine-tune reranker.
