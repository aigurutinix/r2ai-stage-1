"""Cắt relevant_articles của submission_v3 xuống top-K (đã sắp theo rerank).
Tạo submission_v3_k{K}.json + zip. Gold ~1.9 điều/câu → K nhỏ (3-5) tối ưu F2.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
SRC = Path("data/submission_v3.json")
d = json.loads(SRC.read_text(encoding="utf-8"))


def truncate(rec: dict, k: int) -> dict:
    arts = rec["relevant_articles"][:k]
    # relevant_docs = doc riêng biệt trong các điều đã giữ (giữ thứ tự)
    seen, docs = set(), []
    for a in arts:
        doc = "|".join(a.split("|")[:2])  # so_ky_hieu|tên
        if doc not in seen:
            seen.add(doc); docs.append(doc)
    return {"id": rec["id"], "question": rec["question"], "answer": rec["answer"],
            "relevant_docs": docs, "relevant_articles": arts}


for k in (int(x) for x in sys.argv[1:]) if len(sys.argv) > 1 else (3, 4, 5):
    out = [truncate(r, k) for r in d]
    p = Path(f"data/submission_v3_k{k}.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    import statistics
    print(f"K={k}: TB {statistics.mean(len(r['relevant_articles']) for r in out):.1f} điều/câu → {p.name}")
