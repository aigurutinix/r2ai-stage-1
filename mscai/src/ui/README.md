# MscAI Chat UI

UI không có file cấu hình API riêng. Browser gọi endpoint tương đối:

```text
/backend-api/api/v1/legal/chat/stream
```

Next.js route proxy đọc `app.port` trực tiếp từ `../backend/config.yaml` rồi
forward request và SSE tới FastAPI.

Chạy development:

```bash
npm ci
npm run dev
```

Nếu đổi backend port, chỉ sửa `backend/config.yaml` và restart backend. Proxy đọc
lại YAML cho mỗi request; không cần `.env.local`.
