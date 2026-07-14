"""Dựng ứng viên + TEXT cho benchmark Claude-selector (top-20 pool v37 hybrid+rerank + nội dung).

Output data/_claude_cands.json = [{id, question, cands: [{art, text}]}] → workflow Claude chọn.
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    side = json.loads((ROOT / "data/v37_cands.json").read_text(encoding="utf-8"))
    qs = json.loads(Path("C:/Users/PHONG/Downloads/R2AIStage1DATA.json").read_text(encoding="utf-8"))

    from qdrant_client import QdrantClient
    c = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    coll = os.environ.get("QDRANT_COLLECTION", "vbpl_aiteam")
    print("Scroll Qdrant lấy text...", flush=True)
    lut = {}
    off = None
    while True:
        pts, off = c.scroll(coll, limit=4000, offset=off, with_payload=True, with_vectors=False)
        for p in pts:
            pl = p.payload or {}
            sk = str(pl.get("so_ky_hieu") or ""); ds = pl.get("dieu_so")
            if sk and ds is not None:
                lut[(sk, str(ds))] = (pl.get("text") or "")
        if off is None:
            break
    print(f"  lut {len(lut)} điều", flush=True)

    out = []
    for q in qs:
        qid = q["id"]
        cands = side.get(str(qid), [])[:20]
        items = []
        for ci in cands:
            art = ci["art"]; p = art.split("|")
            sk = p[0].strip(); dieu = p[-1].replace("Điều", "").strip()
            txt = lut.get((sk, dieu), "")[:700]
            items.append({"art": art, "text": txt})
        out.append({"id": qid, "question": q["question"], "cands": items})
    json.dump(out, open(ROOT / "data/_claude_cands.json", "w", encoding="utf-8"), ensure_ascii=False)
    import statistics
    print(f"XONG {len(out)} câu | TB {statistics.mean(len(r['cands']) for r in out):.1f} ứng viên/câu "
          f"| có text: {sum(1 for r in out for c in r['cands'] if c['text'])}/{sum(len(r['cands']) for r in out)}")


if __name__ == "__main__":
    main()
