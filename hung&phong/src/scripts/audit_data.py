"""Audit toàn diện chất lượng & độ phủ corpus (đọc parquet, KHÔNG đụng Qdrant).

Bắt mọi loại lỗi data có thể phát hiện từ metadata điều:
  1. Điều "khổng lồ" (nuốt điều cuối)         — char_len 1 điều quá lớn
  2. Gap số điều (skip điều giữa chừng)        — dãy dieu_so không liên tục
  3. Mất điều đầu (không bắt đầu từ Điều 1)
  4. Điều trùng (cùng so_ky_hieu + dieu_so)
  5. Điều quá ngắn (mất nội dung)
  6. Tiêu đề điều rỗng
  7. Văn bản 0 điều còn hiệu lực
  8. so_ky_hieu rỗng/null
  9. Duplicate so_ky_hieu trong documents + mismatch metadata
 10. Văn bản gốc nghi mất điều đuôi (dài nhưng ít điều)

Xuất danh sách văn bản cần sửa → data/_audit_to_fix.json

Chạy: PYTHONUTF8=1 python scripts/audit_data.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus_vbpl_v2"

# ── Ngưỡng ────────────────────────────────────────────────────────────────────
GIANT_CHAR   = 50_000   # 1 điều > ngưỡng này = nghi nuốt điều cuối
SHORT_CHAR   = 60       # điều < ngưỡng = nghi mất nội dung
LONG_DOC_CHAR = 40_000  # tổng văn bản > ngưỡng mà ít điều = nghi mất đuôi
FEW_DIEU     = 12       # số điều bị coi là "ít"
NEW_YEAR     = 2014     # văn bản từ năm này coi là "còn hiệu lực" (ưu tiên)
PRIMARY = ("Luật", "Bộ luật", "Nghị định", "Pháp lệnh")


def hr(title: str):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def main():
    arts = pd.read_parquet(CORPUS / "articles.parquet")
    docs = pd.read_parquet(CORPUS / "documents.parquet")
    arts["char_len"] = arts["char_len"].astype(int)
    arts["dieu_so"]  = arts["dieu_so"].astype(int)
    arts["sk"] = arts["so_ky_hieu"].astype(str)
    docs["sk"] = docs["so_ky_hieu"].astype(str)
    docs["nd"] = docs["n_dieu"].astype(int)

    loai_map  = dict(zip(docs["sk"], docs["loai_van_ban"].astype(str)))
    title_map = dict(zip(docs["sk"], docs["title"].astype(str)))
    nam_map   = dict(zip(docs["sk"], docs["nam"].astype(str)))

    def is_amend(sk):                       # văn bản sửa đổi/hợp nhất (điều dài tự nhiên)
        t = title_map.get(sk, "").lower()
        return ("sửa đổi" in t or "bổ sung" in t or "hợp nhất" in t
                or "danh mục" in t or "biểu thuế" in t)

    def yr(sk):
        n = nam_map.get(sk, "0")
        return int(n) if n.isdigit() else 0

    to_fix: dict[str, dict] = defaultdict(lambda: {"loi": [], "loai": "", "title": "", "nam": 0})
    def flag(sk, loi):
        to_fix[sk]["loi"].append(loi)
        to_fix[sk]["loai"]  = loai_map.get(sk, "?")
        to_fix[sk]["title"] = title_map.get(sk, "")[:60]
        to_fix[sk]["nam"]   = yr(sk)

    # Gom điều theo văn bản
    by_doc = defaultdict(list)
    for sk, ds, cl in zip(arts["sk"], arts["dieu_so"], arts["char_len"]):
        by_doc[sk].append((ds, cl))

    # ── 1. Điều khổng lồ (nuốt cuối) — văn bản GỐC ───────────────────────────
    hr("1. ĐIỀU KHỔNG LỒ — nghi nuốt điều cuối (văn bản gốc)")
    giant = arts[arts["char_len"] > GIANT_CHAR]
    n1 = 0
    for sk in giant["sk"].unique():
        if is_amend(sk):
            continue
        if loai_map.get(sk) in PRIMARY:
            mc = giant[giant["sk"] == sk]["char_len"].max()
            nd = len(by_doc[sk])
            flag(sk, f"dieu_khong_lo({mc:,}c,{nd}dieu)")
            n1 += 1
    print(f"  → {n1} văn bản gốc bị (đã loại văn bản sửa đổi/danh mục).")

    # ── 2. Gap số điều (skip giữa) ───────────────────────────────────────────
    hr("2. GAP SỐ ĐIỀU — skip điều giữa chừng")
    n2 = 0; samples2 = []
    for sk, lst in by_doc.items():
        ds_set = sorted(set(d for d, _ in lst if d > 0))
        if len(ds_set) < 3:
            continue
        gaps = [d for d in range(ds_set[0], ds_set[-1] + 1) if d not in ds_set]
        if gaps and loai_map.get(sk) in PRIMARY and not is_amend(sk):
            n2 += 1
            flag(sk, f"gap_dieu(thieu_{len(gaps)})")
            if len(samples2) < 15:
                samples2.append((sk, ds_set[0], ds_set[-1], len(gaps), loai_map.get(sk,"?"), title_map.get(sk,"")[:35]))
    print(f"  → {n2} văn bản gốc có gap (thiếu điều giữa dãy).")
    for sk, lo, hi, g, l, t in sorted(samples2, key=lambda x:-x[3]):
        print(f"     {sk:16s} {l:9s} Điều {lo}–{hi}, thiếu {g}  {t}")

    # ── 3. Mất điều đầu (không bắt đầu từ Điều 1) ─────────────────────────────
    hr("3. MẤT ĐIỀU ĐẦU — điều nhỏ nhất > 1")
    n3 = 0; samples3 = []
    for sk, lst in by_doc.items():
        ds = [d for d, _ in lst if d > 0]
        if not ds:
            continue
        mn = min(ds)
        if mn > 1 and len(ds) >= 3 and loai_map.get(sk) in PRIMARY and not is_amend(sk):
            n3 += 1
            flag(sk, f"mat_dieu_dau(bat_dau_{mn})")
            if len(samples3) < 12:
                samples3.append((sk, mn, len(ds), title_map.get(sk,"")[:38]))
    print(f"  → {n3} văn bản gốc không bắt đầu từ Điều 1.")
    for sk, mn, n, t in sorted(samples3, key=lambda x:-x[1])[:12]:
        print(f"     {sk:16s} bắt đầu Điều {mn} ({n} điều)  {t}")

    # ── 4. Điều trùng ─────────────────────────────────────────────────────────
    hr("4. ĐIỀU TRÙNG — cùng (số hiệu, điều) xuất hiện >1")
    dup = arts.groupby(["sk", "dieu_so"]).size()
    dup = dup[(dup > 1) & (dup.index.get_level_values("dieu_so") > 0)]
    n4 = len(dup)
    print(f"  → {n4} cặp (văn bản, điều) bị trùng.")
    for (sk, ds), c in dup.head(12).items():
        flag(sk, f"dieu_trung(Đ{ds}x{c})")
        print(f"     {sk:16s} Điều {ds} ×{c}  {title_map.get(sk,'')[:35]}")

    # ── 5. Điều quá ngắn ──────────────────────────────────────────────────────
    hr("5. ĐIỀU QUÁ NGẮN — nghi mất nội dung")
    short = arts[(arts["char_len"] < SHORT_CHAR) & (arts["dieu_so"] > 0)]
    n5 = len(short)
    print(f"  → {n5} điều < {SHORT_CHAR} ký tự.")
    sc = Counter(short["sk"])
    for sk, c in sc.most_common(10):
        if loai_map.get(sk) in PRIMARY:
            print(f"     {sk:16s} {c} điều ngắn  {title_map.get(sk,'')[:35]}")

    # ── 6. Tiêu đề điều rỗng ─────────────────────────────────────────────────
    hr("6. TIÊU ĐỀ ĐIỀU RỖNG")
    empty_t = arts[(arts["dieu_tieu_de"].astype(str).str.strip() == "") & (arts["dieu_so"] > 0)]
    print(f"  → {len(empty_t)} điều không có tiêu đề.")

    # ── 7. Văn bản 0 điều còn hiệu lực ───────────────────────────────────────
    hr(f"7. VĂN BẢN 0 ĐIỀU còn hiệu lực (Luật/NĐ/Pháp lệnh, năm ≥ {NEW_YEAR})")
    have = set(by_doc.keys())
    n7 = 0
    for sk in docs["sk"]:
        if sk and sk not in have and loai_map.get(sk) in PRIMARY and yr(sk) >= NEW_YEAR:
            if not is_amend(sk):
                n7 += 1
                flag(sk, "0_dieu_con_hieu_luc")
                print(f"     {sk:16s} {loai_map.get(sk):10s} {yr(sk)}  {title_map.get(sk,'')[:42]}")
    print(f"  → {n7} văn bản quan trọng còn hiệu lực bị 0 điều.")

    # ── 8. so_ky_hieu rỗng ────────────────────────────────────────────────────
    hr("8. SỐ KÝ HIỆU RỖNG/NULL")
    empty_sk_d = (docs["sk"].str.strip() == "").sum()
    empty_sk_a = (arts["sk"].str.strip() == "").sum()
    print(f"  → documents: {empty_sk_d} dòng | articles: {empty_sk_a} điều thuộc VB không số hiệu.")

    # ── 9. Duplicate so_ky_hieu trong documents + mismatch ───────────────────
    hr("9. SỐ KÝ HIỆU TRÙNG trong documents.parquet")
    vc = docs[docs["sk"].str.strip() != ""]["sk"].value_counts()
    dupd = vc[vc > 1]
    print(f"  → {len(dupd)} số hiệu xuất hiện >1 dòng (gây lệch metadata khi join).")
    mismatch = 0
    for sk in dupd.index[:200]:
        sub = docs[docs["sk"] == sk]
        if sub["loai_van_ban"].nunique() > 1:
            mismatch += 1
    print(f"     trong đó {mismatch} mã có loai_van_ban KHÁC NHAU giữa các dòng (lỗi gán).")

    # ── 10. Văn bản dài nhưng ít điều (nghi mất đuôi, kể cả không khổng lồ) ───
    hr(f"10. DÀI NHƯNG ÍT ĐIỀU — tổng > {LONG_DOC_CHAR:,}c & ≤ {FEW_DIEU} điều")
    n10 = 0; samples10 = []
    for sk, lst in by_doc.items():
        if is_amend(sk) or loai_map.get(sk) not in PRIMARY:
            continue
        total = sum(c for _, c in lst)
        nd = len([d for d, _ in lst if d > 0])
        if total > LONG_DOC_CHAR and 0 < nd <= FEW_DIEU:
            n10 += 1
            flag(sk, f"dai_it_dieu({total:,}c,{nd}dieu)")
            if len(samples10) < 15:
                samples10.append((sk, total, nd, title_map.get(sk,"")[:38]))
    print(f"  → {n10} văn bản gốc dài bất thường so với số điều.")
    for sk, tot, nd, t in sorted(samples10, key=lambda x:-x[1])[:15]:
        print(f"     {sk:16s} {tot:>9,}c {nd:3d}điều  {t}")

    # ── Tổng kết + xuất danh sách sửa ────────────────────────────────────────
    hr("TỔNG KẾT")
    out = {sk: v for sk, v in to_fix.items()}
    (ROOT / "data" / "_audit_to_fix.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    by_loi = Counter()
    for v in out.values():
        for l in v["loi"]:
            by_loi[l.split("(")[0]] += 1
    print(f"  Tổng văn bản cần xem xét sửa: {len(out)}")
    for loi, c in by_loi.most_common():
        print(f"    {c:4d}  {loi}")
    print(f"\n  → Danh sách chi tiết: data/_audit_to_fix.json")


if __name__ == "__main__":
    main()
