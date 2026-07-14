"""v49 — ban_cu blacklist (precision, đòn lớn nhất từ mổ v47: nhiễu bản-cũ 25.7%).

Bỏ điều thuộc VB ĐÃ BỊ THAY THẾ HOÀN TOÀN (BTC chấm bản hiện hành → bản cũ luôn là nhiễu).
Recall-safe: chỉ bỏ khi còn >=1 điều. + NFKC normalize chống trùng Cyrillic/Latin số hiệu.

Chạy: PYTHONUTF8=1 PYTHONPATH=. python scripts/build_v49_blacklist.py
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

# Tiền tố "NN/YYYY" của VB ĐÃ BỊ THAY THẾ HOÀN TOÀN (bản mới hiệu lực TRƯỚC 2025 → BTC chắc chắn
# chấm bản mới). Bỏ tránh các VB rất mới (2024-2025) phòng BTC gán theo mốc cũ.
KNOWN_OLD = {
    # Luật gốc bị thay
    "60/2005",   # Luật DN cũ → 59/2020
    "68/2014",   # Luật DN → 59/2020
    "10/2012",   # BLLĐ → 45/2019
    "78/2006",   # Luật QLT → 38/2019
    "67/2014",   # Luật Đầu tư → 61/2020
    "59/2005",   # Luật Đầu tư cũ
    "70/2006",   # Luật Chứng khoán → 54/2019
    "03/2003",   # Luật Kế toán → 88/2015
    "36/2009",   # (SHTT sửa đổi cũ — cẩn thận, có thể giữ) → để ngoài
    # Nghị định / Thông tư lao động cũ
    "05/2015",   # NĐ hướng dẫn BLLĐ → 145/2020
    "44/2003", "85/2015", "49/2013", "05/2010", "03/2014", "148/2018",
    "27/2014",   # tiền lương cũ
    # Thuế cũ
    "156/2013",  # TT QLT → 80/2021
    "169/2011", "83/2013", "211/2013", "92/2015", "111/2013-cu",
    # Đăng ký DN cũ
    "20/2015", "78/2015", "43/2010", "05/2013", "96/2015",
    # Hóa đơn cũ
    "51/2010", "04/2014", "119/2018", "39/2014", "32/2011",
    # Thương mại / khác cũ
    "37/2006", "06/2006", "35/2006-cu", "154/2005",
}


def norm_sk(sk: str) -> str:
    sk = unicodedata.normalize("NFKC", sk)
    m = re.match(r"(\d+/\d{4})", sk.strip())
    return m.group(1) if m else sk.strip()


def rebuild_docs(arts):
    docs, seen = [], set()
    for a in arts:
        p = a.split("|"); d = f"{p[0].strip()}|{p[1].strip()}" if len(p) >= 2 else p[0].strip()
        if d not in seen:
            seen.add(d); docs.append(d)
    return docs


def main() -> None:
    src = json.loads((ROOT / "data/submission_v47.json").read_text(encoding="utf-8"))
    out = []; n_drop = 0; dropped_sk = {}
    for r in src:
        arts = r["relevant_articles"]
        kept = []; dropped = 0
        for a in arts:
            sk = norm_sk(a.split("|")[0])
            if sk in KNOWN_OLD and (len(arts) - dropped) > 1:   # recall-safe: còn >1
                n_drop += 1; dropped += 1; dropped_sk[sk] = dropped_sk.get(sk, 0) + 1; continue
            kept.append(a)
        if not kept:
            kept = arts[:1]
        nr = dict(r); nr["relevant_articles"] = kept; nr["relevant_docs"] = rebuild_docs(kept)
        out.append(nr)
    json.dump(out, open(ROOT / "data/submission_v49.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    import statistics
    print(f"v49: bỏ {n_drop} điều bản-cũ (blacklist) | TB v47→v49 "
          f"{statistics.mean(len(r['relevant_articles']) for r in src):.2f} → "
          f"{statistics.mean(len(r['relevant_articles']) for r in out):.2f} điều/câu")
    print("  Bỏ theo số hiệu:", dict(sorted(dropped_sk.items(), key=lambda x: -x[1])[:12]))


if __name__ == "__main__":
    main()
