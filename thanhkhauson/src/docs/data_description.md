# Mô Tả Dữ Liệu Và Cấu Trúc 4 File JSONL

Trạng thái: bản làm việc phục vụ thuyết minh và tái hiện hệ thống.

Nguồn dữ liệu:

- HuggingFace dataset: https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents
- Folder sử dụng: https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents/tree/main/legacy
- Hai file trong `data/raw_law/law_only/effective/` là dữ liệu crawl Bộ luật/Luật từ `vbpl.vn`, trong đó nội dung điều đã được cập nhật theo các văn bản sửa đổi.

## 1. Tổng Quan File

| Nhóm | File | Số dòng | Kích thước | Vai trò |
| --- | --- | ---: | ---: | --- |
| Chunk điều | `data/raw_law/law_only/others_doc/parsed_others_doc_database.jsonl` | 4,836,218 | 7.3 GB | Nội dung các văn bản khác đã parse/chunk theo điều hoặc mục |
| Metadata | `data/raw_law/law_only/others_doc/metadata_merged_active.jsonl` | 198,674 | 134 MB | Metadata văn bản còn hiệu lực của tập `others_doc` |
| Metadata | `data/raw_law/law_only/effective/law_luoc_do_merged_dedup.jsonl` | 300 | 8.4 MB | Metadata và lược đồ của Bộ luật/Luật đã crawl và dedup |
| Chunk điều | `data/raw_law/law_only/effective/parsed_law_database.jsonl` | 26,591 | 41 MB | Nội dung Bộ luật/Luật đã parse/chunk theo điều |

Liên kết khóa:

- `parsed_others_doc_database.jsonl.doc_id` tham chiếu tới `metadata_merged_active.jsonl.id`. Khi join nên ép cùng kiểu string vì file metadata lưu `id` dạng integer.
- `parsed_law_database.jsonl.doc_id` tham chiếu tới `law_luoc_do_merged_dedup.jsonl.id`.
- Mỗi dòng trong 2 file parsed là một article/chunk pháp lý, có `doc_id`, `article_no`, `article_title`, `text`.

## 2. `metadata_merged_active.jsonl`

Loại file: JSONL metadata, mỗi dòng là metadata của một văn bản.

| Trường | Kiểu | Mô tả | Null/empty quan sát |
| --- | --- | --- | --- |
| `id` | int | Mã văn bản, dùng để join với `parsed_others_doc_database.doc_id` | Không |
| `document_number` | string | Số hiệu văn bản | Không |
| `title` | string | Tiêu đề văn bản | Không |
| `url` | string | URL nguồn | Không |
| `legal_type` | string | Loại văn bản, ví dụ Nghị quyết, Quyết định, Thông tư | Không |
| `legal_sectors` | string | Lĩnh vực pháp lý | Không |
| `issuing_authority` | string | Cơ quan ban hành | 1 empty |
| `issuance_date` | string | Ngày ban hành, thường dạng `DD/MM/YYYY` | Không |
| `signers` | string/null | Người ký, một số giá trị có dạng `Tên:id` | 227 null |
| `effect_status` | string | Tình trạng hiệu lực | Không |
| `effect_date` | string/null | Ngày có hiệu lực, thường dạng `YYYY-MM-DD` | 387 null |
| `effectless_date` | string/null | Ngày hết hiệu lực nếu có | 198,096 null |

Giá trị `effect_status` sau khi lọc:

| Giá trị | Số dòng |
| --- | ---: |
| `In effect` | 198,674 |

Ghi chú: file này đã được lọc bỏ 20,092 dòng có `effect_status = "No longer applicable"` trước khi đóng gói lại.

Ví dụ một dòng:

```json
{
  "id": 693036,
  "document_number": "115/NQ-HDBCQG",
  "title": "Nghị quyết 115/NQ-HDBCQG năm 2026 ...",
  "url": "https://thuvienphapluat.vn/...",
  "legal_type": "Nghị quyết",
  "legal_sectors": "Bộ máy hành chính",
  "issuing_authority": "Hội đồng bầu cử quốc gia",
  "issuance_date": "29/01/2026",
  "signers": "Trần Thanh Mẫn:2140",
  "effect_status": "In effect",
  "effect_date": "2026-01-29",
  "effectless_date": null
}
```

## 3. `law_luoc_do_merged_dedup.jsonl`

Loại file: JSONL metadata và graph/lược đồ quan hệ cho Bộ luật/Luật.

Schema cấp 1:

| Trường | Kiểu | Mô tả |
| --- | --- | --- |
| `id` | string | Mã văn bản, dùng để join với `parsed_law_database.doc_id` |
| `url` | string | URL nguồn |
| `title` | string | Tiêu đề văn bản |
| `thuoc_tinh` | object | Metadata hành chính/pháp lý của văn bản |
| `luoc_do` | object | Quan hệ với các văn bản khác |

`thuoc_tinh`:

| Trường con | Kiểu | Số dòng có trường |
| --- | --- | ---: |
| `Số hiệu` | string | 300 |
| `Loại văn bản` | string | 300 |
| `Nơi ban hành` | string | 300 |
| `Người ký` | string | 300 |
| `Ngày ban hành` | string | 300 |
| `Tình trạng` | string | 300 |
| `Ngày có hiệu lực` | string | 299 |
| `Số hiệu gốc (VBPL)` | string | 4 |

`luoc_do` là object có các trường con dạng list string. Không phải mọi văn bản đều có đủ mọi quan hệ.

| Trường con | Kiểu | Số dòng có trường |
| --- | --- | ---: |
| `Văn bản được căn cứ` | list[string] | 272 |
| `Văn bản liên quan cùng nội dung` | list[string] | 262 |
| `Văn bản hướng dẫn` | list[string] | 239 |
| `Văn bản sửa đổi bổ sung` | list[string] | 188 |
| `Văn bản hợp nhất` | list[string] | 186 |
| `Văn bản bị sửa đổi bổ sung` | list[string] | 158 |
| `Văn bản liên quan ngôn ngữ` | list[string] | 154 |
| `Văn bản bị thay thế` | list[string] | 148 |
| `Văn bản được dẫn chiếu` | list[string] | 140 |
| `Văn bản thay thế` | list[string] | 7 |

Ví dụ rút gọn:

```json
{
  "id": "675267",
  "url": "https://thuvienphapluat.vn/...",
  "title": "Luật Sở hữu trí tuệ sửa đổi 2025",
  "thuoc_tinh": {
    "Số hiệu": "131/2025/QH15",
    "Loại văn bản": "Luật",
    "Nơi ban hành": "Quốc hội",
    "Tình trạng": "Còn hiệu lực",
    "Ngày có hiệu lực": "01/04/2026"
  },
  "luoc_do": {
    "Văn bản bị sửa đổi bổ sung": ["..."],
    "Văn bản được dẫn chiếu": ["..."]
  }
}
```

## 4. `parsed_others_doc_database.jsonl`

Loại file: JSONL chunked articles, mỗi dòng là một điều/mục/nội dung đã parse từ văn bản thuộc nhóm `others_doc`.

| Trường | Kiểu | Mô tả | Null/empty quan sát |
| --- | --- | --- | --- |
| `doc_id` | string | Mã văn bản, join với `metadata_merged_active.id` | Không trong mẫu 200,000 dòng |
| `article_no` | string | Số điều/mục, ví dụ `Điều 1` | Không trong mẫu 200,000 dòng |
| `article_title` | string | Tiêu đề điều/mục | 61,646 empty trong mẫu 200,000 dòng |
| `text` | string | Nội dung điều/mục/chunk | Không trong mẫu 200,000 dòng |
| `metadata` | object | Metadata denormalized của văn bản | Không |

`metadata`:

| Trường con | Kiểu | Mô tả |
| --- | --- | --- |
| `legal_type` | string | Loại văn bản |
| `document_number` | string | Số hiệu văn bản |
| `title` | string | Tiêu đề văn bản |

Ví dụ rút gọn:

```json
{
  "doc_id": "693036",
  "article_no": "Điều 1",
  "article_title": "Phạm vi điều chỉnh",
  "text": "1. Nghị quyết này hướng dẫn ...",
  "metadata": {
    "legal_type": "Nghị quyết",
    "document_number": "115/NQ-HDBCQG",
    "title": "Nghị quyết 115/NQ-HDBCQG năm 2026 ..."
  }
}
```

## 5. `parsed_law_database.jsonl`

Loại file: JSONL chunked articles, mỗi dòng là một điều của Bộ luật/Luật đã crawl từ `vbpl.vn`.

| Trường | Kiểu | Mô tả | Null/empty quan sát |
| --- | --- | --- | --- |
| `doc_id` | string | Mã văn bản, join với `law_luoc_do_merged_dedup.id` | Không |
| `article_no` | string | Số điều, ví dụ `Điều 1` | Không |
| `article_title` | string | Tiêu đề điều | 361 empty |
| `text` | string | Nội dung điều đã cập nhật sửa đổi nếu có | 3 empty |
| `metadata` | object | Trạng thái và liên kết điều liên quan | Không |

`metadata`:

| Trường con | Kiểu | Mô tả |
| --- | --- | --- |
| `status` | string | Trạng thái của chunk/điều. Giá trị quan sát: `active`, `amended` |
| `related_clauses` | list[object] | Danh sách điều khoản cũ/liên quan khi nội dung đã được sửa đổi |

Thống kê `metadata.status`:

| Giá trị | Số dòng |
| --- | ---: |
| `active` | 23,136 |
| `amended` | 3,455 |

`related_clauses`:

- Luôn là list.
- Có 3,455 dòng có list khác rỗng.
- Mỗi object trong list có các trường:
  - `relation_type`: string, ví dụ `amended`.
  - `doc_ref`: string, tham chiếu văn bản sửa đổi nếu có.
  - `old_content`: string, nội dung cũ bị sửa đổi/thay thế.
  - `action_text_raw`: string, mô tả hành động sửa đổi gốc.

Ví dụ rút gọn:

```json
{
  "doc_id": "108065",
  "article_no": "Điều 1",
  "article_title": "Phạm vi điều chỉnh",
  "text": "Luật này quy định về đối tượng chịu thuế ...",
  "metadata": {
    "status": "active",
    "related_clauses": []
  }
}
```

## 6. Cách Sử Dụng Trong Pipeline

Build index:

1. Nạp metadata từ `law_luoc_do_merged_dedup.jsonl` và `metadata_merged_active.jsonl` vào map theo `id`.
2. Đọc các file parsed/chunked.
3. Tạo text đầy đủ để embed/search bằng cách ghép `title`, `article_no`, `article_title`, `text`.
4. Tạo `chunk_id` deterministic từ `doc_id` và full text.
5. Ghi metadata/text vào Elasticsearch index `legal_chunks`.
6. Ghi embedding vào Qdrant collection `legal_chunks`.

Mapping quan trọng khi xuất kết quả:

- `document_number` lấy từ metadata.
- `doc_name/title` lấy từ metadata title.
- `article_no` lấy từ chunk parsed.
- `text` là nội dung dùng cho retrieval và grounded answer.
