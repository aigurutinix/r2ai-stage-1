"""IMPROVE #1: gộp bản luật cũ → giữ bản MỚI NHẤT, rồi chọn thích ứng → submission_v5.

Vá precision: corpus có nhiều thế hệ cùng 1 luật (Luật DN 13/1999, 60/2005, 68/2014,
59/2020). Reranker chấm đều ~1.0 → submit kèm bản cũ đã hết hiệu lực → precision rớt.

Quy tắc gộp (chỉ áp cho LUẬT/BỘ LUẬT/PHÁP LỆNH — bản mới thay thế hẳn bản cũ):
  - Nhóm ứng viên theo "họ luật" (tên đã bỏ năm + 'sửa đổi bổ sung').
  - Trong cùng họ, chỉ giữ ứng viên có số hiệu năm MỚI NHẤT; bỏ hẳn bản cũ hơn.
  - Nghị định/Thông tư KHÔNG gộp (bản mới không luôn thay thế toàn bộ).
"""
from __future__ import annotations
import json, re, sys, statistics
from collections import Counter
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

YEAR = re.compile(r"/(19|20)(\d{2})/")
PRIMARY = ("luật", "bộ luật", "pháp lệnh")

def year_of(sk: str):
    m = YEAR.search(sk or "")
    return int(m.group(1) + m.group(2)) if m else 0   # không có năm → coi như cũ nhất

def is_primary(name: str) -> bool:
    n = (name or "").lower().strip()
    return any(n.startswith(p) for p in PRIMARY)

def family(name: str) -> str:
    s = (name or "").lower()
    s = re.sub(r"sửa đổi,? bổ sung.*", "", s)
    s = re.sub(r"số\s.*", "", s)
    s = re.sub(r"\d{4}", "", s)
    return re.sub(r"\s+", " ", s).strip(" .,")

def collapse(cands: list[dict]) -> tuple[list[dict], int]:
    """Bỏ ứng viên thuộc họ luật-chính nếu tồn tại số hiệu MỚI HƠN cùng họ trong tập."""
    newest: dict[str, int] = {}
    for c in cands:
        parts = c["art"].split("|")
        sk, name = parts[0], (parts[1] if len(parts) >= 2 else "")
        if is_primary(name):
            fam = family(name)
            newest[fam] = max(newest.get(fam, 0), year_of(sk))
    out, dropped = [], 0
    for c in cands:
        parts = c["art"].split("|")
        sk, name = parts[0], (parts[1] if len(parts) >= 2 else "")
        if is_primary(name):
            fam = family(name)
            if year_of(sk) < newest[fam]:        # có bản mới hơn → bỏ bản cũ
                dropped += 1
                continue
        out.append(c)
    return out, dropped

def adaptive(cands, t_abs=0.45, ratio=0.85, min_k=1, max_k=3):
    if not cands: return []
    cands = sorted(cands, key=lambda c: c["rr"], reverse=True)
    cut = max(t_abs, ratio * cands[0]["rr"])
    chosen = [c for c in cands if c["rr"] >= cut][:max_k]
    return chosen or cands[:min_k]

def main():
    scored = json.loads((ROOT / "data/submission_v4_scored.json").read_text(encoding="utf-8"))
    out, total_dropped, counts, changed = [], 0, [], 0
    for r in scored:
        clean, dropped = collapse(r["candidates"])
        total_dropped += dropped
        if dropped: changed += 1
        chosen = adaptive(clean)
        arts, docs, seen = [], [], set()
        for c in chosen:
            arts.append(c["art"])
            if c["doc"] not in seen:
                seen.add(c["doc"]); docs.append(c["doc"])
        counts.append(len(arts))
        out.append({"id": r["id"], "question": r["question"], "answer": r["answer"],
                    "relevant_docs": docs, "relevant_articles": arts})
    (ROOT / "data/submission_v5.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    dist = Counter(counts)
    print(f"Gộp bản cũ: bỏ {total_dropped} ứng viên cũ ở {changed} câu")
    print(f"TB {statistics.mean(counts):.2f} điều/câu · "
          f"Phân bố: " + " ".join(f"{k}đ:{dist[k]}" for k in sorted(dist)))
    print(f"→ data/submission_v5.json")

if __name__ == "__main__":
    main()
