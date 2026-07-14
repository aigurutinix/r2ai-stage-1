"""Effect-status dedup: trong 1 câu trả lời, khi có NHIỀU phiên bản cùng chủ đề (cùng
loại + cùng tiêu đề, KHÁC số ký hiệu, khác năm) → giữ bản MỚI NHẤT, bỏ bản cũ.

Judge KHÔNG xử được loại này (bản cũ vẫn "trả lời" được câu) → đây là bước bù.
An toàn: chỉ gộp khi tiêu đề-chuẩn-hoá GIỐNG HỆT; cùng số ký hiệu khác Điều → giữ hết;
số ký hiệu không có năm (VBHN/QĐ) → không bị bỏ.

Chạy: python scripts/dedup_effect_status.py --in data/submission_v14_nojudge.json \
        --out data/submission_v15_dedup.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

_YEAR = re.compile(r"/(\d{4})/")
_DIGITS = re.compile(r"\d+")
_QH = re.compile(r"QH\d+")
_WS = re.compile(r"\s+")


def _year(sk: str) -> int:
    m = _YEAR.search(sk)
    return int(m.group(1)) if m else 0


def _loai(sk: str) -> str:
    """Loại văn bản từ số ký hiệu, chuẩn hoá QH13/14/15 → QH (luật qua nhiều khoá)."""
    tail = sk.split("/")[-1] if "/" in sk else sk
    return _QH.sub("QH", tail)


def _norm_title(title: str) -> str:
    return _WS.sub(" ", _DIGITS.sub(" ", title).lower()).strip()


def _key(item: str) -> tuple[str, str]:
    """item = 'so_ky_hieu|title|Điều N' (hoặc 'so_ky_hieu|title'). Trả khoá gom phiên bản."""
    p = item.split("|")
    sk = p[0]
    title = p[1] if len(p) > 1 else ""
    return (_loai(sk), _norm_title(title))


def dedup_items(items: list[str]) -> tuple[list[str], list[str]]:
    """Trả (giữ lại, đã bỏ). Giữ thứ tự gốc."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for it in items:
        groups[_key(it)].append(it)

    drop: set[str] = set()
    for grp in groups.values():
        sks = {it.split("|")[0] for it in grp}
        if len(sks) < 2:            # cùng 1 văn bản (khác Điều) → giữ hết
            continue
        years = [_year(it.split("|")[0]) for it in grp]
        valid = [y for y in years if y > 0]
        if len(valid) < 2:          # không đủ năm để so → không động
            continue
        maxy = max(valid)
        for it in grp:
            y = _year(it.split("|")[0])
            if 0 < y < maxy:        # bản cũ hơn → bỏ
                drop.add(it)
    keep = [it for it in items if it not in drop]
    return keep, [it for it in items if it in drop]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/submission_v14_nojudge.json")
    ap.add_argument("--out", default="data/submission_v15_dedup.json")
    args = ap.parse_args()

    data = json.loads(open(args.inp, encoding="utf-8").read())
    tot_a = drop_a = tot_d = drop_d = 0
    for r in data:
        arts = r.get("relevant_articles", [])
        tot_a += len(arts)
        keep_a, dropped_a = dedup_items(arts)
        drop_a += len(dropped_a)
        r["relevant_articles"] = keep_a

        docs = r.get("relevant_docs", [])
        if docs:
            tot_d += len(docs)
            keep_d, dropped_d = dedup_items(docs)
            drop_d += len(dropped_d)
            r["relevant_docs"] = keep_d

    json.dump(data, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"ARTICLES: {tot_a} → {tot_a - drop_a} (bỏ {drop_a} bản cũ, {drop_a/tot_a*100:.1f}%)")
    if tot_d:
        print(f"DOCS:     {tot_d} → {tot_d - drop_d} (bỏ {drop_d} bản cũ, {drop_d/tot_d*100:.1f}%)")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
