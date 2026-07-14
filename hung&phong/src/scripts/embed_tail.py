"""Embed BÙ các chunk DÀI NHẤT (bị OOM ở đuôi re-embed). Batch nhỏ, KHÔNG recreate.

Idempotent: chunk_id sinh từ md5(text) nên upsert lại các chunk đã có là vô hại; chỉ
các chunk còn thiếu được ghi bổ sung.

Chạy:
  EMBED_BATCH_SIZE=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  QDRANT_COLLECTION=vbpl_aiteam QDRANT_URL=http://localhost:6333
  EMBED_BACKEND=st EMBED_ST_MODEL=AITeamVN/Vietnamese_Embedding_v2
  PYTHONUTF8=1 PYTHONPATH=. python scripts/embed_tail.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

from ingest.parse import ParsedDoc, DieuChunk
from ingest.chunk import chunk_document
from ingest.embed_load import embed_and_upsert
from backend.qdrant_store import QdrantStore
from backend.embed import EmbeddingClient

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus_vbpl_v2"
TAIL_N = int(os.environ.get("EMBED_TAIL_N", "3000"))
_GARBAGE = {"", "Không số", "nan", "None", "none"}


def main() -> None:
    arts = pd.read_parquet(CORPUS / "articles.parquet")
    docs = pd.read_parquet(CORPUS / "documents.parquet")
    dmeta = {}
    for _, r in docs.iterrows():
        dmeta.setdefault(str(r["so_ky_hieu"]), {
            "nam": str(r.get("nam") or ""), "linh_vuc": str(r.get("linh_vuc") or ""),
            "co_quan": str(r.get("co_quan_ban_hanh") or "")})

    by_doc: dict[tuple, list] = defaultdict(list)
    info: dict[tuple, tuple] = {}
    for r in arts.itertuples(index=False):
        sk = str(r.so_ky_hieu).strip()
        if sk in _GARBAGE:
            continue
        key = (sk, str(r.title or "").strip())
        by_doc[key].append(DieuChunk(dieu_so=int(r.dieu_so) if r.dieu_so is not None else 0,
                                     dieu_tieu_de=str(r.dieu_tieu_de or ""), text=str(r.text or "")))
        info.setdefault(key, (str(r.loai_van_ban or ""), str(r.source_url or "")))

    chunks = []
    for (sk, title), dieus in by_doc.items():
        loai, src = info[(sk, title)]
        m = dmeta.get(sk, {})
        dieus.sort(key=lambda d: d.dieu_so)
        doc = ParsedDoc(doc_id=f"{sk}::{title[:60]}", so_ky_hieu=sk, loai_van_ban=loai,
                        co_quan_ban_hanh=m.get("co_quan", ""), ngay_ban_hanh="", ngay_hieu_luc="",
                        tinh_trang_hieu_luc="", linh_vuc=m.get("linh_vuc", ""), title=title, dieus=dieus)
        doc.source_url = src
        doc.nguon = "vbpl.vn (tmquan) + manual"
        doc.nam = m.get("nam", "")
        chunks.extend(chunk_document(doc))

    chunks.sort(key=lambda c: len(c.text), reverse=True)
    tail = chunks[:TAIL_N]
    print(f"Tổng {len(chunks):,} chunk · embed bù {len(tail):,} chunk DÀI NHẤT "
          f"(max {len(tail[0].text)} → min {len(tail[-1].text)} ký tự)", flush=True)

    store = QdrantStore()
    n = embed_and_upsert(tail, embedder=EmbeddingClient(), store=store, show_progress=True)
    print(f"XONG bù {n:,} chunk → {store.collection} ({store.count():,} điểm tổng)", flush=True)


if __name__ == "__main__":
    main()
