import json
import os
from qdrant_client import QdrantClient
from config.settings import Settings

def export_from_qdrant():
    print("--- Đang kết nối Qdrant Cloud để lấy dữ liệu làm Corpus BM25 ---")
    client = QdrantClient(
        url=Settings.QDRANT_URL,
        api_key=Settings.QDRANT_API_KEY
    )
    
    corpus = []
    # Dùng tính năng scroll để kéo toàn bộ data về (mỗi lần 1000 bản ghi)
    # scroll() trả về tuple (records, next_offset); truyền next_offset vào offset= cho trang kế.
    next_offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=Settings.COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False,  # Không lấy vector để file nhẹ, chỉ lấy text/metadata
            offset=next_offset,
        )

        for record in records:
            payload = record.payload or {}
            # Giữ TOÀN BỘ metadata trong payload Qdrant (không lược bớt trường nào).
            # Nếu payload đã có 'metadata' lồng (dict) thì giữ nguyên; nếu payload phẳng
            # thì gom mọi trường ngoài 'text' vào metadata.
            if isinstance(payload.get("metadata"), dict):
                metadata = payload["metadata"]
            else:
                metadata = {k: v for k, v in payload.items() if k != "text"}

            # id = unique_article_id để KHỚP với nhánh dense (HybridRetriever),
            # đảm bảo RRF merge gộp trùng đúng giữa BM25 và Qdrant.
            uid = (payload.get("unique_article_id")
                   or metadata.get("unique_article_id")
                   or payload.get("chunk_id")
                   or metadata.get("chunk_id")
                   or str(record.id))

            corpus.append({
                "id": uid,
                "text": payload.get("text", ""),
                "metadata": metadata,
            })

        if next_offset is None:
            break

    # Ghi ra file data/law_corpus_clean.json
    os.makedirs("data", exist_ok=True)
    output_path = "data/law_corpus_clean.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Hoàn thành! Đã tạo thành công file {output_path} với {len(corpus)} văn bản đoạn luật.")

if __name__ == "__main__":
    export_from_qdrant()