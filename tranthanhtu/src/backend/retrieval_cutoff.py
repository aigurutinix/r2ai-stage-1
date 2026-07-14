"""Reranker-score cutoff + legal-validity filter — pure python, NO heavy deps.

Shared by local_rag_engine (live retrieval), generate_submission (--retrieved cache),
and scratch/sweep_cutoff.py (offline leaderboard-variant sweeping). Keeping it dependency-
free lets the submission/sweep paths import it without pulling sentence-transformers.

Cutoff policy (recall-floor): always keep the top-1 candidate, then keep each next one
while it passes an absolute floor (min_score) AND stays within `margin` of the top score,
capped at top_k. F2 favors recall 2× — prefer a gentle (loose) cutoff over an aggressive one.

Validity filter: vbpl-vn dataset has NO validity/status field, so we (a) drop a curated
list of well-known superseded laws, (b) within one candidate list, dedup same-name docs
keeping the newest year. Measured on real 2000-question retrieval: superseded laws appeared
in 735/2000 questions' top-12 — big precision drain (gold cites the in-force version).
"""
import re
from typing import List, Dict, Any, Optional

# Văn bản ĐÃ BỊ THAY THẾ (hết hiệu lực toàn bộ) → bản hiện hành. Curated, verified.
# Chỉ thêm cặp CHẮC CHẮN — gold trỏ bản hiện hành nên giữ bản cũ = sai cả docs lẫn articles.
SUPERSEDED_DOCS = {
    "60/2005/QH11": "59/2020/QH14",   # Luật Doanh nghiệp 2005 → 2020
    "68/2014/QH13": "59/2020/QH14",   # Luật Doanh nghiệp 2014 → 2020
    "35/2002/QH10": "45/2019/QH14",   # BLLĐ sửa đổi 2002 → 2019
    "10/2012/QH13": "45/2019/QH14",   # Bộ luật Lao động 2012 → 2019
    "59/2005/QH11": "61/2020/QH14",   # Luật Đầu tư 2005 → 2020
    "67/2014/QH13": "61/2020/QH14",   # Luật Đầu tư 2014 → 2020
    "13/2003/QH11": "31/2024/QH15",   # Luật Đất đai 2003 → 2024
    "45/2013/QH13": "31/2024/QH15",   # Luật Đất đai 2013 → 2024
    "65/2014/QH13": "27/2023/QH15",   # Luật Nhà ở 2014 → 2023
    "66/2014/QH13": "29/2023/QH15",   # Luật KD bất động sản 2014 → 2023
    "71/2006/QH11": "41/2024/QH15",   # Luật BHXH 2006 → 2024
    "58/2014/QH13": "41/2024/QH15",   # Luật BHXH 2014 → 2024 (hiệu lực 01/07/2025)
    "47/2010/QH12": "32/2024/QH15",   # Luật Các TCTD 2010 → 2024
    "70/2006/QH11": "54/2019/QH14",   # Luật Chứng khoán 2006 → 2019
    "52/2005/QH11": "72/2020/QH14",   # Luật BV môi trường 2005 → 2020
    "55/2014/QH13": "72/2020/QH14",   # Luật BV môi trường 2014 → 2020
    "33/2005/QH11": "91/2015/QH13",   # Bộ luật Dân sự 2005 → 2015
    "78/2015/NĐ-CP": "01/2021/NĐ-CP", # NĐ đăng ký doanh nghiệp 2015 → 2021
}


def _doc_year(doc_number: str) -> int:
    m = re.search(r"/((?:19|20)\d{2})/", doc_number or "") or re.search(r"((?:19|20)\d{2})", doc_number or "")
    return int(m.group(1)) if m else 0


def drop_superseded(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop candidates from superseded laws; dedup same-name docs keeping the newest year.

    Score order is preserved (only removals). Never returns empty when input is non-empty
    (falls back to the original list — better a stale citation than none).
    """
    if not cands:
        return cands
    # (b) trong cùng danh sách: tên chuẩn hóa trùng nhau + khác năm → chỉ giữ năm mới nhất.
    # An toàn theo ngữ cảnh: 2 văn bản trùng tên trong CÙNG top-N của 1 câu ≈ các version của nhau.
    newest: Dict[str, int] = {}
    for d in cands:
        nm = re.sub(r"\s+", " ", (d.get("clean_name") or "").strip().lower())
        y = _doc_year(d.get("doc_number", ""))
        if nm:
            newest[nm] = max(newest.get(nm, 0), y)
    out = []
    for d in cands:
        dn = d.get("doc_number", "")
        if dn in SUPERSEDED_DOCS:                       # (a) luật chắc chắn hết hiệu lực
            continue
        nm = re.sub(r"\s+", " ", (d.get("clean_name") or "").strip().lower())
        if nm and _doc_year(dn) < newest.get(nm, 0):    # (b) version cũ hơn của cùng tên
            continue
        out.append(d)
    return out or cands


def apply_cutoff(
    ranked: List[Dict[str, Any]],
    top_k: int,
    min_score: Optional[float] = None,
    margin: Optional[float] = None,
    score_key: str = "score",
) -> List[Dict[str, Any]]:
    """Trim a score-sorted candidate list. `ranked` must be sorted desc by score_key.

    - top_k     : hard cap on returned count.
    - min_score : drop candidates with score < this (None = no absolute floor).
    - margin    : drop candidates with score < (top_score - margin) (None = no relative cut).
    """
    if not ranked:
        return []
    out = [ranked[0]]
    top = ranked[0].get(score_key, 0.0)
    for d in ranked[1:top_k]:
        s = d.get(score_key, 0.0)
        if min_score is not None and s < min_score:
            break
        if margin is not None and s < top - margin:
            break
        out.append(d)
    return out
