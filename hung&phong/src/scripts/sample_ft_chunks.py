"""Lấy mẫu chunk ĐA DẠNG từ Qdrant để Claude sinh synthetic query (fine-tune reranker).

Phân tầng theo văn bản (so_ky_hieu) → phủ nhiều luật/NĐ/TT. Chỉ lấy Điều thực, text đủ dài
(là căn cứ pháp lý tốt để đặt câu hỏi tình huống). Mỗi mục: {idx, art, text}.
art = "so_ky_hieu|ten|Điều N" (khớp định dạng article của submission → mine hard-neg + gán positive).

Chạy: QDRANT_COLLECTION=vbpl_aiteam QDRANT_URL=http://localhost:6333 EMBED_BACKEND=st \
  EMBED_ST_MODEL=AITeamVN/Vietnamese_Embedding_v2 PYTHONUTF8=1 PYTHONPATH=. \
  python scripts/sample_ft_chunks.py --per-doc 6 --max 1200
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-doc", type=int, default=6, help="tối đa chunk/văn bản")
    ap.add_argument("--max", type=int, default=1200)
    ap.add_argument("--min-chars", type=int, default=350)
    ap.add_argument("--min-year", type=int, default=0, help="chỉ lấy văn bản năm >= (0=tắt) — bơm luật MỚI")
    ap.add_argument("--out", default="data/_ft_chunks.json")
    args = ap.parse_args()

    import re as _re
    def _year(sk: str) -> int:
        m = _re.search(r"/(19|20)(\d{2})/", sk or "")
        return int(m.group(1) + m.group(2)) if m else 0

    from backend.rag import RAGPipeline
    from tests.build_submission_v12 import _law_name

    # CHỈ lấy văn bản IN-SCOPE = đã xuất hiện trong submission (đúng domain cuộc thi),
    # tránh rác như "đề cương môn học Chủ nghĩa xã hội".
    in_scope = {d["sk"] for d in json.loads((ROOT / "data/_corpus_docs.json").read_text(encoding="utf-8"))}
    print(f"In-scope: {len(in_scope)} văn bản", flush=True)

    rag = RAGPipeline()
    client = rag.store.client
    coll = rag.settings.qdrant_collection

    by_doc: dict[str, dict[str, dict]] = defaultdict(dict)  # sk -> {art: chunk} (dedupe theo Điều)
    offset = None
    seen = 0
    while True:
        pts, offset = client.scroll(coll, limit=2000, offset=offset,
                                    with_payload=True, with_vectors=False)
        for p in pts:
            pl = p.payload or {}
            sk = str(pl.get("so_ky_hieu") or "")
            ds = pl.get("dieu_so")
            text = str(pl.get("text") or "")
            if not sk or sk not in in_scope or ds is None or int(ds) <= 0 or len(text) < args.min_chars:
                continue
            if args.min_year and _year(sk) < args.min_year:
                continue
            art = f"{sk}|{_law_name(pl)}|Điều {ds}"
            # dedupe theo Điều: giữ chunk DÀI NHẤT (đầy đủ nội dung điều)
            cur = by_doc[sk].get(art)
            if cur is None or len(text) > len(cur["text"]):
                by_doc[sk][art] = {"art": art, "text": text[:1400]}
        seen += len(pts)
        if offset is None:
            break

    # phân tầng: trải đều mỗi văn bản (đa dạng số Điều)
    chosen = []
    for sk, artmap in by_doc.items():
        items = list(artmap.values())
        step = max(1, len(items) // args.per_doc)
        for c in items[::step][: args.per_doc]:
            chosen.append(c)
    # cắt tổng, đánh idx
    chosen = chosen[: args.max]
    for i, c in enumerate(chosen):
        c["idx"] = i

    Path(ROOT / args.out).write_text(json.dumps(chosen, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Quét {seen:,} điểm | {len(by_doc):,} văn bản | chọn {len(chosen)} chunk → {args.out}", flush=True)
    print("Mẫu:", flush=True)
    for c in chosen[:3]:
        print(f"  [{c['idx']}] {c['art'][:70]} | {c['text'][:80]}...", flush=True)


if __name__ == "__main__":
    main()
