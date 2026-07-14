# v54 Real Fusion Pipeline

Mục tiêu của v54 là cải thiện hệ thống thật, không tối ưu bằng hardcode id/range theo điểm submit.

## Lý do đổi hướng

`v50` là full pipeline nhưng precision thấp hơn `v42`: nó bắt thêm được một số căn cứ đúng, đồng thời thả nhiều điều nhiễu. `v42` có điểm cao vì selector sạch hơn, nhưng không phải pipeline chatbot tự vận hành. Vì vậy v54 phải giữ tinh thần full pipeline:

```text
query -> candidate pool rộng hơn -> fusion/rerank -> citation selector -> answer
```

Không dùng:

- hardcode id câu hỏi;
- hardcode block 1001-1200/1401-1600;
- thay đáp án theo điểm probe.

## Tín hiệu từ Mentor Day 3

PDF `R2AI_MENTORDAY3part1.pdf` nhấn mạnh các hướng sau:

- BM25 mạnh exact-match, dense mạnh ngữ nghĩa.
- Hybrid nên dùng RRF thay vì cộng điểm tuyến tính vì điểm giữa retriever khác thang đo.
- Recall phải cao ở retrieval, precision xử lý ở rerank/selection.
- Query Expansion và HyDE giải quyết mismatch giữa ngôn ngữ đời thường và ngôn ngữ luật.
- ColBERT/BGE-M3 late interaction là hướng tiếp theo nếu cần bắt chi tiết điều/khoản tốt hơn.

Repo hiện đã có BM25 + dense + cross-encoder reranker. Phần còn thiếu là RRF fusion chính thức giữa nhiều truy vấn và HyDE/query expansion được kiểm soát.

## Thiết kế v54

Script: `scripts/build_v54_real_fusion.py`

Input:

- câu hỏi gốc trong `C:/Users/PHONG/Downloads/R2AIStage1DATA.json`;
- collection `vbpl_aiteam`;
- embedding `AITeamVN/Vietnamese_Embedding_v2`;
- reranker `AITeamVN/Vietnamese_Reranker`;
- BM25 index `data/bm25_vbpl_aiteam.pkl`;
- subquery cũ nếu có trong `data/subqueries.json`.

Flow mỗi câu:

1. Chuẩn bị truy vấn:
   - query gốc;
   - tối đa 2 subquery có sẵn;
   - tối đa 2 query rewrite do Qwen sinh, prompt bắt giữ nguyên tình tiết, không thêm vế;
   - 1 HyDE passage ngắn, văn phong pháp lý, không bịa số điều/số văn bản.

2. Retrieve từng truy vấn:
   - dùng dense + BM25 raw, chưa cross-encoder rerank ở từng truy vấn;
   - query gốc lấy sâu hơn rewrite/HyDE;
   - lưu sidecar để audit nguồn nào kéo article nào lên.

3. RRF fusion ở cấp điều luật:
   - mỗi danh sách đóng góp theo rank: `weight / (k + rank)`;
   - query gốc weight cao nhất;
   - subquery/rewrite/HyDE weight thấp hơn để tránh trôi dạt;
   - dedupe theo `(so_ky_hieu, dieu_so)`.

4. Cross-encoder rerank một lần:
   - lấy top pool sau RRF;
   - rerank bằng câu hỏi gốc;
   - điểm chọn cuối kết hợp rerank score và RRF agreement.

5. Selector tổng quát:
   - bắt đầu từ top theo fused score;
   - max_k phụ thuộc độ phức tạp câu hỏi;
   - nếu câu có dấu hiệu chế tài/xử phạt thì ưu tiên thêm NĐ xử phạt đúng whitelist;
   - nếu chỉ có Luật mà câu hỏi có thủ tục/hồ sơ/trình tự/mức cụ thể thì thêm NĐ/TT phù hợp;
   - nếu chỉ có NĐ/TT mà câu hỏi hỏi quyền/nghĩa vụ/điều kiện/nguyên tắc thì thêm Luật/Bộ luật;
   - áp dụng collapse/version/domain/blacklist như v47/v49;
   - không để empty.

6. Output:
   - `data/submission_v54_real_fusion_1001_2000.json`;
   - `data/submission_v54_real_fusion_1001_2000.zip`;
   - `data/v54_real_fusion_sidecar.json` để debug từng câu.

## Điểm khác với v53 probe

v53 dùng để định vị lỗi bằng submit block. v54 không dùng điểm block để chọn kết quả. Tất cả rule đều áp dụng toàn bộ câu hỏi, dựa trên đặc tính truy vấn và cấu trúc văn bản.

## Rủi ro

- HyDE/rewrite có thể trôi dạt. Giảm rủi ro bằng:
  - giới hạn số rewrite;
  - weight thấp hơn query gốc;
  - prompt không cho thêm tình tiết;
  - selector vẫn ưu tiên ứng viên được nhiều nguồn đồng thuận.
- RRF có thể kéo thêm nhiễu nếu fetch quá sâu. v54 mặc định giữ top vừa phải và max_k thấp.
- Không dùng ColBERT/BGE-M3 sparse trong v54 vì repo chưa có index tương ứng; đây là hướng v55 nếu v54 chưa đủ.

## Cách chạy

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='.'
python scripts\build_v54_real_fusion.py --lo 1001 --hi 2000 --zip
```

Smoke test:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='.'
python scripts\build_v54_real_fusion.py --lo 1001 --hi 1010 --out data/_v54_smoke.json --zip
```
