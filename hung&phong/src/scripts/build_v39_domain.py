"""BƯỚC 1 — Domain filter (post-process v33, FREE, no rebuild). Diệt noise_sai_domain (~25%).

Loại điều thuộc LĨNH VỰC CHUYÊN BIỆT (quốc phòng/công an/hải quan/điện lực/dầu khí/hàng hải/
thú y/y-dược/công chức...) KHI câu hỏi KHÔNG nhắc tới lĩnh vực đó. Recall-safe: chỉ bỏ khi
còn >=1 điều. Áp lên đáp án CUỐI của v33 → không retrieve/rerank lại.

Chạy: PYTHONUTF8=1 PYTHONPATH=. python scripts/build_v39_domain.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

# (keyword TRONG TÊN văn bản → lĩnh vực chuyên biệt, trigger TRONG CÂU HỎI để GIỮ lại)
DOMAINS = [
    (["quốc phòng", "quân sự", "quân đội", "cơ yếu", "sĩ quan"], ["quân", "quốc phòng", "cơ yếu", "sĩ quan", "vũ trang"]),
    (["công an", "an ninh nhân dân", "cảnh sát"], ["công an", "an ninh", "cảnh sát"]),
    (["hải quan"], ["hải quan", "xuất khẩu", "nhập khẩu", "thông quan", "xnk"]),
    (["điện lực"], ["điện lực", "điện năng", "lưới điện"]),
    (["dầu khí"], ["dầu khí", "xăng dầu"]),
    (["hàng hải", "hàng không", "đường thủy nội địa", "đường sắt"], ["hàng hải", "hàng không", "đường thủy", "đường sắt", "tàu biển", "máy bay"]),
    (["thú y", "chăn nuôi", "trồng trọt", "thủy sản", "bảo vệ thực vật", "lâm nghiệp"], ["thú y", "chăn nuôi", "trồng trọt", "thủy sản", "nông nghiệp", "vật nuôi", "cây trồng", "lâm nghiệp"]),
    (["khám bệnh, chữa bệnh", "dược", "an toàn thực phẩm"], ["khám bệnh", "chữa bệnh", "dược", "thuốc", "bệnh viện", "thực phẩm"]),
    (["cán bộ, công chức", "công chức", "viên chức"], ["công chức", "viên chức", "cán bộ", "đơn vị sự nghiệp"]),
]


def name_of(art: str) -> str:
    p = art.split("|")
    return p[1].lower() if len(p) >= 2 else art.lower()


def is_offdomain(art: str, q: str) -> str | None:
    nm = name_of(art); ql = q.lower()
    for kws, trigs in DOMAINS:
        if any(k in nm for k in kws) and not any(t in ql for t in trigs):
            return next(k for k in kws if k in nm)
    return None


def rebuild_docs(arts):
    docs, seen = [], set()
    for a in arts:
        p = a.split("|"); d = f"{p[0].strip()}|{p[1].strip()}" if len(p) >= 2 else p[0].strip()
        if d not in seen:
            seen.add(d); docs.append(d)
    return docs


def main() -> None:
    src = json.loads((ROOT / "data/submission_v33_v24map.json").read_text(encoding="utf-8"))
    out = []
    n_drop = 0; examples = []
    for r in src:
        q = r["question"]; arts = r["relevant_articles"]
        kept, dropped = [], []
        for a in arts:
            dom = is_offdomain(a, q)
            if dom and len(arts) - len(dropped) > 1:   # recall-safe: còn >=1
                dropped.append((a, dom))
            else:
                kept.append(a)
        n_drop += len(dropped)
        if dropped and len(examples) < 15:
            examples.append((q[:70], [(a.split("|")[1] if "|" in a else a, d) for a, d in dropped]))
        nr = dict(r); nr["relevant_articles"] = kept; nr["relevant_docs"] = rebuild_docs(kept)
        out.append(nr)
    json.dump(out, open(ROOT / "data/submission_v39_domain.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    import statistics
    print(f"Bỏ {n_drop} điều off-domain | TB v33→v39: "
          f"{statistics.mean(len(r['relevant_articles']) for r in src):.3f} → "
          f"{statistics.mean(len(r['relevant_articles']) for r in out):.3f} điều/câu")
    print(f"Số câu bị động: {sum(1 for s,o in zip(src,out) if len(s['relevant_articles'])!=len(o['relevant_articles']))}")
    print("\n=== VÍ DỤ điều bị bỏ (sanity-check có đúng off-domain không) ===")
    for q, drs in examples:
        print(f"Q: {q}")
        for nm, dom in drs:
            print(f"   ✗ bỏ [{dom}]: {nm}")


if __name__ == "__main__":
    main()
