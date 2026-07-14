# v55 Mentor Repo Notes

Mục tiêu của v55 là học các phần có thể đưa vào hệ thống thật từ repo `AI-Guru-R2AI/R2AI-MENTOR-DAY3`, không dùng hardcode theo id hoặc block submit.

## Đã đưa vào ngay

Script `scripts/build_v54_real_fusion.py` đã có thêm tham số:

```powershell
--dense-query-mode raw|instruct|both
```

- `raw`: giữ hành vi v54 cũ, embed query trực tiếp.
- `instruct`: dùng query prefix từ repo mentor trước khi embed dense:

```text
Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question
Query: {query}
```

- `both`: retrieve bằng cả raw dense và instructed dense, sau đó đưa cả hai list vào RRF cùng BM25. Cách này tăng cơ hội recall nhưng tốn thời gian hơn.

Smoke test đã chạy:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='.'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python scripts\build_v54_real_fusion.py `
  --lo 1001 --hi 1002 `
  --out data\_v55_smoke_instruct.json `
  --sidecar data\_v55_smoke_instruct_sidecar.json `
  --dense-query-mode instruct `
  --no-llm-expansion `
  --orig-topk 10 --sub-topk 4 --rerank-pool 25 `
  --checkpoint-every 1 --zip
```

Kết quả smoke:

- `data/_v55_smoke_instruct.json`
- `data/_v55_smoke_instruct.zip`
- `data/_v55_smoke_instruct_sidecar.json`
- zip hợp lệ, chỉ chứa `results.json`.

## Chưa đưa vào ngay

Repo mentor dùng Qdrant native hybrid với dense vector + sparse BM25 vector trong cùng collection:

```python
prefetch=[
    models.Prefetch(query=query_dense, using="dense", limit=20),
    models.Prefetch(query=query_sparse, using="sparse", limit=20),
]
query=models.FusionQuery(fusion=models.Fusion.RRF)
```

Collection hiện tại `vbpl_aiteam` của mình đang dùng dense Qdrant + BM25 pickle riêng. Muốn dùng native sparse giống repo mentor cần build collection mới hoặc re-index collection cũ với sparse vector field. Không nên gắn phần này vào v55 nếu chưa re-index, vì sẽ tạo pipeline không phản ánh đúng index đang chạy.

## Lệnh chạy 1000 câu sau

Biến thể thực tế đầu tiên nên submit-test là v55 instruct, vì ít tăng chi phí hơn `both` nhưng đã áp dụng được insight chính từ repo mentor:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='.'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python scripts\build_v54_real_fusion.py `
  --lo 1001 --hi 2000 `
  --out data\submission_v55_mentor_instruct_1001_2000.json `
  --sidecar data\v55_mentor_instruct_sidecar.json `
  --query-cache data\query_cache_v55_mentor_instruct.json `
  --dense-query-mode instruct `
  --max-rewrites 1 `
  --sub-topk 6 --rewrite-topk 6 --hyde-topk 6 `
  --orig-topk 16 --rerank-pool 40 `
  --checkpoint-every 10 --zip
```

Nếu `instruct` không tốt hơn, test tiếp `both` trên cùng dải câu nhưng cần chấp nhận thời gian chạy dài hơn.
