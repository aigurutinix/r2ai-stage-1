"""Central configuration for the offline competition pipeline.

Keeps model names, paths, and the curated SME law allowlist in ONE place so
ingestion / retrieval / corpus-building stay in sync (DRY).

Compliance (competition rules): all models are open-source, <14B params, and
released before 2026-03-01.
  - Embedding : AITeamVN/Vietnamese_Embedding (BGE-M3, 0.6B, 1024-dim, MIT, 2025)
  - Reranker  : AITeamVN/Vietnamese_Reranker  (BGE-reranker-v2-m3, 0.6B, MIT, 2025)
  - LLM       : Qwen2.5-7B-Instruct           (7.6B, Apache-2.0, 2024-09) via Ollama
"""
import os

# ---- Paths ------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "..", "data"))
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")   # legacy, no longer used
CORPUS_JSONL = os.path.join(DATA_DIR, "corpus_articles.jsonl")
CORPUS_EMB = os.path.join(DATA_DIR, "corpus_emb.npy")        # float32 (N×dim), row i ↔ jsonl line i
CORPUS_EMB_IDS = os.path.join(DATA_DIR, "corpus_emb_ids.json")  # id order in .npy (incremental check)
TEST_QUESTIONS = os.path.join(DATA_DIR, "test_questions.json")
GOLD_DEV = os.path.join(DATA_DIR, "gold_dev.json")

# ---- Models -----------------------------------------------------------------
EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding"
RERANKER_MODEL = "AITeamVN/Vietnamese_Reranker"
EMBEDDING_DIM = 1024
# BGE-M3 default max_seq_length=8192 (fine-tuned at 1024). NOT capping it makes
# encoding O(n²)-heavy & very slow. Legal articles fit well under 1024 tokens.
EMBED_MAX_SEQ_LEN = 1024   # embedding (bi-encoder)
RERANK_MAX_LEN = 512       # reranker (query + article snippet)

# ---- Retrieval cutoff — TUNED qua leaderboard thật (c50_t3m15 = 0.5371) -------
# Lưu top-N candidate KÈM điểm rerank (RETRIEVE_CAND_SAVE) → cutoff thành bước RẺ,
# sweep offline qua leaderboard (scratch/sweep_cutoff.py).
# Journey: 0.317 → 0.3877 (t3m3) → 0.4616 (corpus 93K) → 0.4887 (lọc hiệu lực)
# → 0.4975 (t3m2) → 0.5371 (phễu rerank 50 + t3m15; P 0.563/R 0.553).
# m1.5 vs m2.0 trên phễu 50: recall Y HỆT 0.553, precision 0.563 vs 0.52 → điều trong
# khoảng margin 1.5-2.0 toàn sai → margin chặt là đúng.
RETRIEVE_TOP_K = 3          # top-3 — mọi cấu hình rộng hơn đều thua trên leaderboard
RETRIEVE_MIN_SCORE = None   # bỏ ngưỡng tuyệt đối (điểm rerank là logit, thang lệch theo câu)
RETRIEVE_MARGIN = 1.5       # giữ điều có điểm >= top - 1.5
RETRIEVE_CAND_SAVE = 12     # số candidate (kèm điểm) lưu vào retrieved.json để sweep cutoff offline

# ---- Hybrid fusion (RRF CÓ TRỌNG SỐ — BM25 nặng hơn) ------------------------
# Research VN legal IR: BM25 là tín hiệu chính (số luật/điều/thuật ngữ khớp chính xác) →
# ưu BM25 hơn dense. Đổi W_BM25=W_DENSE=0.5 để so sánh (equal-weight = hành vi cũ).
RRF_K = 60
RRF_W_BM25 = 0.65
RRF_W_DENSE = 0.35
OLLAMA_MODEL = "qwen2.5:7b-instruct-q4_K_M"
OLLAMA_HOST = "http://localhost:11434"

CHROMA_COLLECTION = "vietnamese_laws"


def get_device() -> str:
    """Best available torch device: CUDA (Colab/Kaggle GPU) > MPS (Mac) > CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ---- HuggingFace corpus datasets (recommended by organizers) ----------------
HF_VBPL = "tmquan/vbpl-vn"            # config="documents" — primary statutory corpus
HF_PHAPDIEN = "tmquan/phapdien-moj-gov-vn"   # config="articles" — codified (secondary)
HF_ANLE = "tmquan/anle-toaan-gov-vn"         # config="documents" — case law (optional)

# ---- SME-relevant law allowlist (mã văn bản) --------------------------------
# Curated core laws covering the competition's SME scope (doanh nghiệp, lao động,
# thuế, thương mại, BHXH...). doc_number in vbpl-vn is matched case-insensitively.
# Extend freely — coverage here is the ceiling for retrieval recall.
SME_LAW_ALLOWLIST = {
    # ── Core (đã verify trong dataset, 2411 Điều) ──
    "59/2020/QH14",   # Luật Doanh nghiệp
    "04/2017/QH14",   # Luật Hỗ trợ doanh nghiệp nhỏ và vừa
    "80/2021/NĐ-CP",  # NĐ hướng dẫn Luật Hỗ trợ DNNVV
    "01/2021/NĐ-CP",  # NĐ về đăng ký doanh nghiệp
    "45/2019/QH14",   # Bộ luật Lao động
    "145/2020/NĐ-CP", # NĐ hướng dẫn BLLĐ về điều kiện & quan hệ lao động
    "36/2005/QH11",   # Luật Thương mại
    "58/2014/QH13",   # Luật Bảo hiểm xã hội (2014)
    "38/2019/QH14",   # Luật Quản lý thuế
    "61/2020/QH14",   # Luật Đầu tư
    "91/2015/QH13",   # Bộ luật Dân sự
    "88/2015/QH13",   # Luật Kế toán
    # ── Bổ sung theo phân tích 2000 câu test (~400 câu chưa phủ) ──
    "50/2005/QH11",   # Luật Sở hữu trí tuệ            (~134 câu)
    "07/2022/QH15",   # Luật sửa đổi, bổ sung Luật SHTT 2022
    "123/2020/NĐ-CP", # NĐ về hóa đơn, chứng từ        (~121 câu)
    "47/2010/QH12",   # Luật các tổ chức tín dụng 2010 (~74 câu)
    "32/2024/QH15",   # Luật các tổ chức tín dụng 2024
    "45/2013/QH13",   # Luật Đất đai 2013              (~24 câu)
    "31/2024/QH15",   # Luật Đất đai 2024
    "200/2014/TT-BTC",# TT chế độ kế toán doanh nghiệp (có thể KHÔNG có trên vbpl-vn)
    "133/2016/TT-BTC",# TT chế độ kế toán DN nhỏ và vừa (có thể KHÔNG có trên vbpl-vn)
    # ── Mở rộng đợt 2 theo phân tích 2000 câu test (lỗ hổng độ phủ) ──
    "12/2022/NĐ-CP",  # NĐ xử phạt VPHC lao động/BHXH (~114 câu — mức phạt)
    "84/2015/QH13",   # Luật An toàn, vệ sinh lao động      (~43 câu)
    "14/2008/QH12",   # Luật Thuế thu nhập doanh nghiệp     (~32 câu)
    "32/2013/QH13",   # Luật sửa đổi Thuế TNDN
    "71/2014/QH13",   # Luật sửa đổi các luật về thuế
    "54/2010/QH12",   # Luật Trọng tài thương mại          (~30 câu)
    "13/2008/QH12",   # Luật Thuế giá trị gia tăng          (~25 câu)
    "31/2013/QH13",   # Luật sửa đổi Thuế GTGT
    "38/2013/QH13",   # Luật Việc làm (BH thất nghiệp)      (~16 câu)
    "23/2018/QH14",   # Luật Cạnh tranh                     (~13 câu)
    "12/2012/QH13",   # Luật Công đoàn                      (~10 câu)
    "54/2019/QH14",   # Luật Chứng khoán                    (~10 câu)
    # 04/2007/QH12 (Luật Thuế TNCN): markdown NULL trong vbpl-vn (chỉ có 1 Nghị quyết
    # trùng số) → không lấy được từ dataset này → bỏ. ~8 câu TNCN chấp nhận thiếu.
}

# Document types to keep when scanning vbpl-vn (legal_type field, lowercased contains).
# Only gates --mode keywords; allowlist mode bypasses this (matches by doc_number).
# CHỈ giữ văn bản QUY PHẠM. Bỏ "quyết định" + "nghị quyết": khi quét rộng chúng chiếm
# 53% (131K/248K điều) nhưng phần lớn là quyết định hành chính cá biệt / NQ không quy phạm
# → nhiễu nặng, chôn gold. ("thông tư" đã bao "thông tư liên tịch" qua substring.)
KEEP_LEGAL_TYPES = ("luật", "bộ luật", "nghị định", "thông tư", "pháp lệnh")

# Keyword sweep cho --mode keywords (quét RỘNG toàn bộ vbpl-vn theo chủ đề SME).
# Từ khóa khớp substring trên title+summary (lowercase) → chọn từ ĐẶC TRƯNG, tránh
# substring ngắn gây nhiễu ("giá","phí"→"chi phí"). Đây là CỬA chính đặt trần recall.
SME_TITLE_KEYWORDS = (
    # Doanh nghiệp & tổ chức kinh doanh
    "doanh nghiệp", "hộ kinh doanh", "hợp tác xã", "hỗ trợ doanh nghiệp",
    # Lao động & an sinh
    "lao động", "việc làm", "tiền lương", "công đoàn", "vệ sinh lao động",
    "bảo hiểm xã hội", "bảo hiểm y tế", "bảo hiểm thất nghiệp",
    # Thuế - kế toán - hóa đơn
    "thuế", "lệ phí", "hóa đơn", "kế toán", "kiểm toán",
    # Thương mại - đầu tư - cạnh tranh - tiêu dùng
    "thương mại", "đầu tư", "cạnh tranh", "quảng cáo", "người tiêu dùng",
    "xuất khẩu", "nhập khẩu", "hải quan",
    # Tài chính - tín dụng - SHTT
    "tổ chức tín dụng", "chứng khoán", "sở hữu trí tuệ",
    # Đất đai - xây dựng - bất động sản
    "đất đai", "xây dựng", "nhà ở", "bất động sản",
    # An toàn - môi trường
    "bảo vệ môi trường", "phòng cháy", "an toàn thực phẩm",
    # Chế tài (nhiều câu hỏi về mức phạt)
    "xử phạt vi phạm hành chính",
)
