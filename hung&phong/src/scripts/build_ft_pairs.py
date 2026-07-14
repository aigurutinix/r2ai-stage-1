"""Gộp query (cũ + luật-mới) → pairs hợp nhất [{query, pos_art, pos_text}] cho mining.

BỎ positive là bản SUPERSEDED (thành viên cũ trong họ phiên bản — map) → không dạy reranker
xếp bản cũ cao. Giữ bản hiện hành + văn bản không thuộc họ nào.

Chạy: PYTHONUTF8=1 PYTHONPATH=. python scripts/build_ft_pairs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

# (queries_file, chunks_file)
SOURCES = [
    ("data/_ft_queries.json", "data/_ft_chunks.json"),
    ("data/_ft_queries_new.json", "data/_ft_chunks_new.json"),
]


def main() -> None:
    # superseded = thành viên KHÔNG phải newest của mỗi họ phiên bản
    superseded = set()
    try:
        fams = json.loads((ROOT / "data/version_families.json").read_text(encoding="utf-8"))
        for f in fams:
            new = f.get("newest_sk")
            for m in f.get("members", []):
                if m != new:
                    superseded.add(m.strip())
    except Exception:  # noqa: BLE001
        pass
    print(f"Superseded sk (bỏ khỏi positive): {len(superseded)}", flush=True)

    pairs = []
    seen_q = set()
    n_drop_old = 0
    for qf, cf in SOURCES:
        qp = ROOT / qf
        cp = ROOT / cf
        if not qp.exists() or not cp.exists():
            print(f"  bỏ qua (thiếu file): {qf}", flush=True)
            continue
        chunks = {c["idx"]: c for c in json.loads(cp.read_text(encoding="utf-8"))}
        qs = json.loads(qp.read_text(encoding="utf-8"))
        if isinstance(qs, dict):
            qs = qs.get("pairs", [])
        n = 0
        for q in qs:
            ch = chunks.get(q["idx"])
            if not ch:
                continue
            sk = ch["art"].split("|")[0].strip()
            if sk in superseded:
                n_drop_old += 1
                continue
            query = q["query"].strip()
            if query in seen_q or len(query) < 15:
                continue
            seen_q.add(query)
            pairs.append({"query": query, "pos_art": ch["art"], "pos_text": ch["text"]})
            n += 1
        print(f"  {qf}: +{n} pairs", flush=True)

    (ROOT / "data/_ft_pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"XONG: {len(pairs)} pairs (bỏ {n_drop_old} positive superseded) → data/_ft_pairs.json", flush=True)


if __name__ == "__main__":
    main()
