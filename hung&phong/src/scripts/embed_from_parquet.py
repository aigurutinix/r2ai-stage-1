"""Embed corpus TỪ articles.parquet (đã chốt) → Qdrant. Nhanh hơn run_vbpl vì bỏ qua
pha load 158k VB + parse lại (chậm ~60 phút).

articles.parquet = data cuối (đã sửa parser, dedup, scope whitelist, 11 VB manual).

Chạy:
  QDRANT_COLLECTION=vbpl_aiteam EMBED_BACKEND=st EMBED_ST_MODEL=AITeamVN/Vietnamese_Embedding_v2
  PYTHONUNBUFFERED=1 python scripts/embed_from_parquet.py
"""
from __future__ import annotations
import sys, time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
from tqdm import tqdm

from ingest.parse import ParsedDoc, DieuChunk
from ingest.chunk import chunk_document
from ingest.embed_load import embed_and_upsert
from backend.qdrant_store import QdrantStore
from backend.embed import EmbeddingClient

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus_vbpl_v2"


def main():
    print("Đọc parquet...", flush=True)
    arts = pd.read_parquet(CORPUS / "articles.parquet")
    docs = pd.read_parquet(CORPUS / "documents.parquet")
    # metadata theo so_ky_hieu (nam, lĩnh vực, cơ quan)
    dmeta = {}
    for _, r in docs.iterrows():
        dmeta.setdefault(str(r["so_ky_hieu"]), {
            "nam": str(r.get("nam") or ""),
            "linh_vuc": str(r.get("linh_vuc") or ""),
            "co_quan": str(r.get("co_quan_ban_hanh") or ""),
        })
    print(f"  {len(arts):,} điều · {len(docs):,} VB", flush=True)

    # Group điều theo VĂN BẢN THẬT = (so_ky_hieu, title).
    # KHÔNG gom chỉ theo so_ky_hieu: 155 VB cũ đều mang "Không số" (+ 463 ký hiệu đụng
    # độ khác như 178-CP) → nếu gom theo mình so_ky_hieu sẽ trộn nhiều VB thành 1 "siêu
    # văn bản" và gán nhầm title. Loại luôn ký hiệu RÁC (không thể khớp gold của BTC).
    _GARBAGE_SK = {"", "Không số", "nan", "None", "none"}
    by_doc: dict[tuple, list] = defaultdict(list)
    info: dict[tuple, tuple] = {}
    skipped = 0
    for r in arts.itertuples(index=False):
        sk = str(r.so_ky_hieu).strip()
        if sk in _GARBAGE_SK:
            skipped += 1
            continue
        title = str(r.title or "").strip()
        key = (sk, title)
        ds = int(r.dieu_so) if r.dieu_so is not None else 0
        by_doc[key].append(DieuChunk(dieu_so=ds, dieu_tieu_de=str(r.dieu_tieu_de or ""),
                                      text=str(r.text or "")))
        if key not in info:
            info[key] = (str(r.loai_van_ban or ""), str(r.source_url or ""))
    print(f"  Loại {skipped:,} điều ký hiệu rác (Không số/rỗng) · {len(by_doc):,} văn bản thật", flush=True)

    # Build chunks
    print("Build chunks...", flush=True)
    chunks = []
    for (sk, title), dieus in tqdm(by_doc.items(), desc="chunk", unit="VB"):
        loai, src = info[(sk, title)]
        m = dmeta.get(sk, {})
        dieus.sort(key=lambda d: d.dieu_so)  # đảm bảo đúng thứ tự Điều khi gộp/đè
        doc = ParsedDoc(
            doc_id=f"{sk}::{title[:60]}", so_ky_hieu=sk, loai_van_ban=loai,
            co_quan_ban_hanh=m.get("co_quan", ""), ngay_ban_hanh="", ngay_hieu_luc="",
            tinh_trang_hieu_luc="", linh_vuc=m.get("linh_vuc", ""), title=title, dieus=dieus,
        )
        doc.source_url = src
        doc.nguon = "vbpl.vn (tmquan) + manual"
        doc.nam = m.get("nam", "")
        chunks.extend(chunk_document(doc))
    print(f"  → {len(chunks):,} chunks", flush=True)

    # Sort theo ĐỘ DÀI để mỗi batch đồng đều → giảm padding tới chunk dài nhất, tăng
    # tốc embed GPU đáng kể (chunk_id duy nhất nên thứ tự upsert không ảnh hưởng).
    chunks.sort(key=lambda c: len(c.text))

    # Embed + upsert (recreate collection)
    store = QdrantStore()
    print(f"Collection: {store.collection} — recreate + embed...", flush=True)
    store.ensure_collection(recreate=True)
    t0 = time.time()
    n = embed_and_upsert(chunks, embedder=EmbeddingClient(), store=store, show_progress=True)
    print(f"XONG: {n:,} chunks ({time.time()-t0:.0f}s) → {store.collection}", flush=True)


if __name__ == "__main__":
    main()
