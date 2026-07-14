import json
import os
import glob
import torch
import uuid
import numpy as np
from pathlib import Path
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
ES_HOST = "http://localhost:9201"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
INDEX_NAME = "legal_chunks"
BATCH_SIZE = 128
MAX_SEQ_LENGTH = 512
DEVICE = os.getenv("INDEX_DEVICE", "cuda:1")

START_LINE = int(os.getenv("START_LINE")) if os.getenv("START_LINE") else None
END_LINE = int(os.getenv("END_LINE")) if os.getenv("END_LINE") else None

MODEL_DIR = str(BASE_DIR / "model_cache/vietlegal-harrier-0.6b")
VNCORENLP_DIR = str(BASE_DIR / "model_cache/vncorenlp")

DEFAULT_DATA_FILES = [
    str(BASE_DIR / "data/raw_law/law_only/effective/parsed_law_database_tokenized.jsonl"),
    str(BASE_DIR / "data/raw_law/law_only/others_doc/parsed_others_doc_database_tokenized.jsonl"),
]
DATA_FILES = [
    item.strip()
    for item in os.getenv("INDEX_DATA_FILES", ",".join(DEFAULT_DATA_FILES)).split(",")
    if item.strip()
]

print("Đang tải metadata để lấy các trường bổ sung...")
doc_metadata_dict = {}

with (BASE_DIR / "data/raw_law/law_only/effective/law_luoc_do_merged_dedup.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        doc_id = str(doc.get("id", ""))
        thuoc_tinh = doc.get("thuoc_tinh", {})
        doc_metadata_dict[doc_id] = {
            "title": doc.get("title", ""),
            "legal_type": thuoc_tinh.get("Loại văn bản", ""),
            "document_number": thuoc_tinh.get("Số hiệu", ""),
            "effect_date": thuoc_tinh.get("Ngày có hiệu lực", ""),
            "effectless_date": thuoc_tinh.get("Ngày hết hiệu lực", "")
        }

with (BASE_DIR / "data/raw_law/law_only/others_doc/metadata_merged_active.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        doc_id = str(doc.get("id", ""))
        doc_metadata_dict[doc_id] = {
            "title": doc.get("title", ""),
            "legal_type": doc.get("legal_type", ""),
            "document_number": doc.get("document_number", ""),
            "effect_date": doc.get("effect_date", ""),
            "effectless_date": doc.get("effectless_date", "")
        }

print("Đang load model Embeddings (VietLegal Harrier)...")
model = SentenceTransformer(MODEL_DIR, model_kwargs={"torch_dtype": torch.float32})
model.max_seq_length = MAX_SEQ_LENGTH

try:
    print("Đang biên dịch (compile) model để tăng tốc inference...")
    model[0].auto_model = torch.compile(model[0].auto_model)
except Exception as e:
    print(f"Bỏ qua compile do lỗi (vẫn chạy bình thường): {e}")

print("Đang tải mapping UUID cho các file embedding đã lưu...")
chunk_id_to_location = {}
global_idx = 0
embedding_ids_path = BASE_DIR / "embeddings_output_ids.jsonl"
if embedding_ids_path.exists():
    with embedding_ids_path.open("r") as f:
        for line in f:
            cid = json.loads(line)
            part_idx = global_idx // 25600
            row_idx = global_idx % 25600
            chunk_id_to_location[cid] = (part_idx, row_idx)
            global_idx += 1

current_part_idx = -1
current_part_data = None

def get_precomputed_embedding(chunk_id):
    global current_part_idx, current_part_data
    if chunk_id in chunk_id_to_location:
        part_idx, row_idx = chunk_id_to_location[chunk_id]
        if part_idx != current_part_idx:
            part_path = BASE_DIR / "embeddings" / f"embeddings_output_part_{part_idx}.npy"
            if part_path.exists():
                current_part_data = np.load(part_path)
                current_part_idx = part_idx
            else:
                return None
        
        if current_part_data is not None and row_idx < current_part_data.shape[0]:
            return current_part_data[row_idx].tolist()
    return None

def get_embeddings(texts, chunk_ids):
    embeddings = [None] * len(texts)
    texts_to_compute = []
    indices_to_compute = []
    
    for i, (text, cid) in enumerate(zip(texts, chunk_ids)):
        emb = get_precomputed_embedding(cid)
        if emb is not None:
            embeddings[i] = emb
        else:
            texts_to_compute.append(text)
            indices_to_compute.append(i)
            
    if texts_to_compute:
        computed_embs = model.encode(texts_to_compute, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False).tolist()
        for i, emb in zip(indices_to_compute, computed_embs):
            embeddings[i] = emb
            
    return embeddings

print("Đang kết nối Elasticsearch tại", ES_HOST)
es = Elasticsearch(ES_HOST)

mapping = {
    "settings": {
        "analysis": {
            "analyzer": {
                "vi_whitespace_lower": {
                    "type": "custom",
                    "tokenizer": "whitespace",
                    "filter": ["lowercase"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "legal_type": {"type": "keyword"},
            "document_number": {"type": "keyword"},
            "effect_date": {"type": "date", "format": "yyyy-MM-dd||dd/MM/yyyy||strict_date_optional_time||epoch_millis", "ignore_malformed": True},
            "effectless_date": {"type": "date", "format": "yyyy-MM-dd||dd/MM/yyyy||strict_date_optional_time||epoch_millis", "ignore_malformed": True},
            "raw_title": {"type": "keyword", "index": False},
            "article_no": {"type": "keyword"},
            "article_title": {"type": "keyword", "index": False},
            "raw_content": {"type": "keyword", "index": False},
            "content_search": {"type": "text", "analyzer": "vi_whitespace_lower"}
        }
    }
}
if not es.indices.exists(index=INDEX_NAME):
    es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"Đã tạo index '{INDEX_NAME}' trên Elasticsearch.")
else:
    print(f"Index '{INDEX_NAME}' đã tồn tại trên Elasticsearch, sẽ ghi đè (upsert) dữ liệu...")

print("Đang kết nối Qdrant tại", QDRANT_HOST, ":", QDRANT_PORT)
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

if not qdrant.collection_exists(collection_name=INDEX_NAME):
    qdrant.create_collection(
        collection_name=INDEX_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    print(f"Đã tạo collection '{INDEX_NAME}' trên Qdrant.")
else:
    print(f"Collection '{INDEX_NAME}' đã tồn tại trên Qdrant, sẽ ghi đè (upsert)...")

print("Bắt đầu xử lý và ingest dữ liệu...")

def process_batch(batch):
    embed_texts = []
    chunk_ids = []

    for r in batch:
        doc_id = str(r.get("doc_id") or "").strip()
        meta = doc_metadata_dict.get(doc_id, {})
        law_title = str(meta.get("title", "")).strip()
        article_no = str(r.get("article_no") or "").strip()
        article_title = str(r.get("article_title") or "").strip()
        text = str(r.get("text") or "").strip()

        prefix = ""
        if law_title:
            prefix += law_title + "\n"
        if article_no:
            prefix += article_no
            if article_title:
                prefix += ": " + article_title
            prefix += "\n"
        elif article_title:
            prefix += article_title + "\n"
        
        full_text = prefix + text
        embed_texts.append(full_text)
        
        unique_string = f"{doc_id}:::{full_text}"
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))
        chunk_ids.append(chunk_id)
        
    embeddings = get_embeddings(embed_texts, chunk_ids)
    
    es_actions = []
    qdrant_points = []
    for chunk_id, r, emb, full_text in zip(chunk_ids, batch, embeddings, embed_texts):
        doc_id = str(r.get("doc_id") or "").strip()
        
        meta = doc_metadata_dict.get(doc_id, {})
        legal_type = str(meta.get("legal_type") or "").strip()
        document_number = str(meta.get("document_number") or "").strip()
        effect_date = str(meta.get("effect_date") or "").strip()
        effectless_date = str(meta.get("effectless_date") or "").strip()
        law_title = str(meta.get("title") or "").strip()

        article_no = str(r.get("article_no") or "").strip()
        article_title = str(r.get('article_title') or '').strip()
        content = str(r.get('text') or '').strip()
        
        effect_date = effect_date if effect_date and effect_date != "None" else None
        effectless_date = effectless_date if effectless_date and effectless_date != "None" else None

        tokenized_content_search = r.get("tokenized_content_search", "")

        action = {
            "_op_type": "index",
            "_index": INDEX_NAME,
            "_id": chunk_id,
            "_source": {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "legal_type": legal_type,
                "document_number": document_number,
                "effect_date": effect_date,
                "effectless_date": effectless_date,
                "article_no": article_no,
                "raw_title": law_title,
                "article_title": article_title,
                "raw_content": content,
                "content_search": tokenized_content_search
            }
        }
        es_actions.append(action)

        point = PointStruct(
            id=chunk_id,
            vector=emb,
            payload={
                "chunk_id": chunk_id,
                "legal_type": legal_type
            }
        )
        qdrant_points.append(point)

    qdrant.upsert(
        collection_name=INDEX_NAME,
        points=qdrant_points
    )

    return es_actions

def generate_actions():
    safe_device_name = DEVICE.replace(":", "")
    state_file = BASE_DIR / f"ingestion_state_{safe_device_name}.json"
    processed_state = {}
    if state_file.exists():
        with state_file.open("r") as f:
            processed_state = json.load(f)

    for filepath in DATA_FILES:
        if not os.path.exists(filepath):
            print(f"Warning: Không tìm thấy file {filepath}")
            continue
            
        print(f"\nĐang xử lý file: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            state_start = processed_state.get(filepath, 0)
            base_start = START_LINE if START_LINE is not None else 0
            actual_start = max(base_start, state_start)
            
            if actual_start > 0:
                print(f"Bỏ qua {actual_start} dòng đầu tiên...")
                for _ in range(actual_start):
                    next(f)
                    
            line_current = actual_start
            batch_records = []
            
            for line in f:
                if END_LINE is not None and line_current >= END_LINE:
                    print(f"Đã đạt đến END_LINE ({END_LINE}). Ngừng xử lý file này.")
                    break
                    
                line_current += 1
                if not line.strip(): 
                    continue
                    
                record = json.loads(line)
                batch_records.append(record)
                if len(batch_records) >= BATCH_SIZE:
                    yield from process_batch(batch_records)
                    batch_records = []
                    
                    processed_state[filepath] = line_current
                    with state_file.open("w") as sf:
                        json.dump(processed_state, sf)
            
            if batch_records:
                yield from process_batch(batch_records)
                processed_state[filepath] = line_current
                with state_file.open("w") as sf:
                    json.dump(processed_state, sf)

success, failed = helpers.bulk(es.options(request_timeout=60), tqdm(generate_actions(), desc="Đang Ingest (Chunks)"), stats_only=True)
print(f"Hoàn tất! Ingest thành công {success} chunks vào ES (và Qdrant tương ứng), thất bại {failed} chunks.")
