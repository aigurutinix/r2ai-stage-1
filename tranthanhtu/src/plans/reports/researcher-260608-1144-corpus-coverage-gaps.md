# Research: Độ phủ corpus — có cần thêm luật không?

**Ngày:** 2026-06-08 11:44 · **Loại:** Data-driven (quét 2000 câu test thật)

## Executive Summary

**"Quá ít điều luật" — KHÔNG hẳn.** 3777 Điều / 20 văn bản là hợp lý cho SME (recall bị chặn bởi độ phủ **văn bản**, không phải số Điều). NHƯNG quét 2000 câu thật phát hiện **~10 văn bản còn thiếu** phủ ~250-300 câu (ước tính thực ~150-250 sau khi trừ overlap). Đáng bổ sung vì F2 ưu ái recall + chi phí thấp (thêm luật KHÔNG làm chậm retrieval — reranker cố định CAND/câu).

169/2000 câu không khớp keyword chủ đề, nhưng kiểm tay thấy **đa số ĐÃ phủ** (hộ kinh doanh→NĐ 01/2021; SHTT/hóa đơn/kế toán/quấy rối→luật đã có). Độ phủ tốt hơn vẻ ngoài.

## Lỗ hổng THẬT (văn bản chưa có trong allowlist)

| Câu (≈) | Chủ đề | Văn bản cần thêm | Ưu tiên |
|---|---|---|---|
| **114** | Xử phạt VPHC lao động/BHXH (mức phạt) | `12/2022/NĐ-CP` | ⭐⭐⭐ |
| 43 | An toàn vệ sinh lao động | `84/2015/QH13` | ⭐⭐⭐ |
| 32 | Thuế TNDN (thuế suất, chi phí được trừ, ưu đãi) | `14/2008/QH12` (+`32/2013/QH13`, `71/2014/QH13`) | ⭐⭐⭐ |
| 30 | Trọng tài thương mại / tranh chấp | `54/2010/QH12` | ⭐⭐ |
| 25 | Thuế GTGT (khấu trừ, hoàn thuế) | `13/2008/QH12` (+`31/2013/QH13`) | ⭐⭐ |
| 16 | Việc làm / BH thất nghiệp | `38/2013/QH13` | ⭐⭐ |
| 13 | Cạnh tranh | `23/2018/QH14` | ⭐ |
| 10 | Công đoàn / kinh phí công đoàn | `12/2012/QH13` | ⭐ |
| 10 | Chứng khoán (CTCP đại chúng) | `54/2019/QH14` | ⭐ |
| 8 | Thuế TNCN | `04/2007/QH12` (+`26/2012/QH13`) | ⭐ |

**Vì sao quan trọng:** các văn bản này chứa **câu trả lời cụ thể** (mức phạt X triệu, thuế suất Y%, thời hạn...) mà `38/2019` (chỉ là Luật Quản lý thuế — thủ tục) và BLLĐ không có. Gold gần như chắc trỏ tới chúng.

## Vì sao thêm là an toàn (không hại)
- **Recall ↑**: F2 = 5PR/(4P+R), recall nặng 2× → phủ thêm văn bản gold = ăn điểm.
- **Precision không tụt nhiều**: reranker vẫn chỉ rerank CAND=20/câu rồi cutoff theo điểm → văn bản thừa bị loại nếu không liên quan.
- **Tốc độ không đổi**: corpus 3777→~6300 Điều, numpy cosine vẫn tức thì; reranker cố định/câu.
- **Incremental**: chỉ embed Điều mới (`build_corpus --append` + `embed_corpus`), không chạy lại từ đầu.

## Đã làm
Thêm 13 mã (10 luật + 3 sửa đổi thuế) vào `SME_LAW_ALLOWLIST`. Chạy để cập nhật:
```bash
python backend/build_corpus.py --append   # chỉ fetch luật MỚI (xem log "MISSING" nếu mã sai)
python backend/embed_corpus.py            # chỉ embed Điều mới → corpus_emb.npy
```

## Câu hỏi mở
1. Một số luật thuế có nhiều "Văn bản hợp nhất" — vbpl-vn lưu bản gốc + sửa đổi; thêm cả gốc + sửa đổi để phủ. Nếu mã nào báo MISSING trong log → bỏ/đổi.
2. Sau khi thêm, nên đo lại F2 trên 20-mock (sanity) — nhưng gold_dev chỉ phủ luật cũ, không phản ánh đủ; số thật phải chờ leaderboard.
3. Không nên thêm tràn lan (luật ngoài SME) → loãng + chậm. Chỉ thêm theo data như trên.
