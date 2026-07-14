"""Manual eval: retrieve full pipeline cho các câu BIẾT TRƯỚC gold (gồm câu nhắm VB mới
bổ sung) → in top điều để TỰ CHẤM. Benchmark đáng tin hơn pseudo-gold (không circular).

Chạy: QDRANT_URL=http://localhost:6333 ... python scripts/manual_eval.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
from backend.rag import RAGPipeline

# (câu hỏi, gold mong đợi — để tự đối chiếu). ⭐ = nhắm VB mới bổ sung / parser sửa.
CASES = [
    ("Tuổi nghỉ hưu của người lao động trong điều kiện bình thường được quy định thế nào?",
     "⭐ 135/2020/NĐ-CP (tuổi nghỉ hưu) + BLLĐ 45/2019 Điều 169"),
    ("Đối tượng nào tham gia bảo hiểm xã hội bắt buộc?",
     "⭐ Luật BHXH 41/2024 Điều 2 (data mới, +103 điều)"),
    ("Mức xử phạt đối với hành vi lập hóa đơn không đúng thời điểm là bao nhiêu?",
     "⭐ 125/2020/NĐ-CP (xử phạt thuế/hóa đơn — bổ sung docx)"),
    ("Thuế suất thuế thu nhập doanh nghiệp đối với doanh nghiệp nhỏ là bao nhiêu?",
     "⭐ Luật TNDN 67/2025 (data mới)"),
    ("Thời gian thử việc tối đa đối với người lao động là bao lâu?",
     "BLLĐ 45/2019 Điều 25"),
    ("Vốn điều lệ của công ty cổ phần được quy định như thế nào?",
     "Luật DN 59/2020 Điều 112"),
    ("Doanh nghiệp nhỏ và vừa cần đáp ứng tiêu chí gì?",
     "Luật SME 04/2017 Điều 4, NĐ 80/2021"),
    ("Hồ sơ đăng ký thành lập doanh nghiệp gồm những gì?",
     "⭐ 168/2025/NĐ-CP (đăng ký DN mới) hoặc 01/2021"),
    ("Luật sửa đổi bổ sung về sở hữu trí tuệ 2022 quy định gì về nhãn hiệu?",
     "⭐ 07/2022/QH15 (Luật sửa đổi SHTT — bổ sung docx)"),
    ("Hoạt động khuyến mại được thực hiện với hạn mức tối đa bao nhiêu?",
     "⭐ 81/2018/NĐ-CP (xúc tiến thương mại — bổ sung docx)"),
    ("Hợp đồng thuê tài sản không xác định thời hạn thì chấm dứt thế nào?",
     "BLDS 91/2015 (Điều thuê tài sản)"),
    ("Mức lương tối thiểu vùng hiện hành được quy định ra sao?",
     "⭐ 293/2025/NĐ-CP (lương tối thiểu)"),
    ("Phối hợp liên thông đăng ký thành lập doanh nghiệp và đăng ký thuế thế nào?",
     "⭐ 122/2020/NĐ-CP (liên thông — bổ sung docx)"),
    ("Chế độ kế toán cho doanh nghiệp siêu nhỏ áp dụng ra sao?",
     "132/2018/TT-BTC (kế toán siêu nhỏ)"),
    ("Điều kiện và thủ tục giải thể doanh nghiệp?",
     "Luật DN 59/2020 Điều 207-208"),
]


def main():
    rag = RAGPipeline()
    print(f"Collection: {rag.store.collection} ({rag.store.count():,} điểm)\n" + "="*78)
    for q, gold in CASES:
        hits = rag.retrieve(q, top_k=5)
        print(f"\nQ: {q}")
        print(f"   [gold mong đợi: {gold}]")
        for h in hits[:5]:
            p = h.get("payload", {})
            sc = h.get("rerank_score", h.get("adj_score", h.get("score", 0)))
            print(f"   {sc:.3f} | {str(p.get('so_ky_hieu')):16s} | Điều {p.get('dieu_so')} | {str(p.get('title'))[:42]}")


if __name__ == "__main__":
    main()
