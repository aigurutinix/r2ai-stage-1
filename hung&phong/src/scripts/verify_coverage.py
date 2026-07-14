"""Verify độ phủ nguồn tmquan/vbpl-vn: có đủ văn bản trong phạm trù SME/DN không?

Chỉ dùng tmquan (KHÔNG dùng th1nhng0). Streaming metadata (nhẹ RAM, an toàn khi v12 chạy).

Trả lời 3 câu:
  1. Tmquan gốc có bao nhiêu VB? Sau lọc scope còn bao nhiêu?
  2. Checklist VB cốt lõi + VB nghi thiếu: có trong tmquan gốc không? Bị loại ở khâu nào?
  3. Có bao nhiêu VB khớp loại-lõi nhưng bị rớt vì từ khóa (nghi loại nhầm)?

Chạy: PYTHONUTF8=1 python scripts/verify_coverage.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset
from backend.config import get_settings
from ingest.scope import in_scope_vbpl, VBPL_CORE_TYPES, SME_KEYWORDS_VBPL

# Checklist VB cốt lõi + VB nghi thiếu/lỗi (đã phát hiện)
CHECK = {
    "59/2020/QH14": "Luật Doanh nghiệp 2020",
    "04/2017/QH14": "Luật Hỗ trợ DNNVV",
    "61/2020/QH14": "Luật Đầu tư 2020",
    "45/2019/QH14": "Bộ luật Lao động 2019",
    "38/2019/QH14": "Luật Quản lý thuế",
    "125/2020/NĐ-CP": "NĐ xử phạt thuế/hóa đơn (NGHI THIẾU)",
    "99/2013/NĐ-CP": "NĐ xử phạt sở hữu công nghiệp (NGHI THIẾU)",
    "07/2022/QH15": "Luật sửa đổi SHTT 2022 (NGHI THIẾU)",
    "122/2021/NĐ-CP": "NĐ xử phạt KH&ĐT (parse lỗi)",
    "80/2021/NĐ-CP": "NĐ hướng dẫn Luật SME",
    "01/2021/NĐ-CP": "NĐ đăng ký doanh nghiệp",
    "98/2020/NĐ-CP": "NĐ xử phạt thương mại",
    "17/2022/NĐ-CP": "NĐ xử phạt chứng khoán",
}


def main():
    s = get_settings()
    print("Nạp tmquan/vbpl-vn (streaming metadata)...", flush=True)
    ds = load_dataset("tmquan/vbpl-vn", "documents", split="train",
                      cache_dir=s.hf_cache_dir, streaming=True)

    total = 0
    n_tw = 0          # scope trung_uong
    n_core = 0        # trung_uong + loại lõi
    n_scope = 0       # qua in_scope_vbpl (đủ điều kiện vào corpus)
    n_core_no_kw = 0  # loại lõi nhưng rớt từ khóa (NGHI LOẠI NHẦM)

    found = {}        # sk -> (legal_type, scope, in_core, kw_hit, in_scope)
    check_sk = set(CHECK)

    for row in ds:
        total += 1
        sk = str(row.get("doc_number") or "").strip()
        lt = str(row.get("legal_type") or "").strip()
        sc = str(row.get("scope") or "").strip()
        blob = " ".join(str(row.get(k) or "") for k in ("title", "legal_area")).lower()
        kw_hit = any(kw in blob for kw in SME_KEYWORDS_VBPL)
        is_tw = sc == "trung_uong"
        is_core = is_tw and lt in VBPL_CORE_TYPES
        is_scope = in_scope_vbpl(row)

        if is_tw: n_tw += 1
        if is_core: n_core += 1
        if is_scope: n_scope += 1
        if is_core and not kw_hit: n_core_no_kw += 1

        if sk in check_sk and sk not in found:
            found[sk] = (lt, sc, is_core, kw_hit, is_scope, str(row.get("title") or "")[:45])

        if total % 40000 == 0:
            print(f"  ...{total:,} VB", flush=True)

    print(f"\n{'='*64}")
    print(f"  ĐỘ PHỦ tmquan/vbpl-vn")
    print(f"{'='*64}")
    print(f"  Tổng VB gốc            : {total:,}")
    print(f"  scope=trung_uong       : {n_tw:,}")
    print(f"  + loại lõi             : {n_core:,}")
    print(f"  + khớp từ khóa (corpus): {n_scope:,}")
    print(f"  >>> Loại lõi nhưng RỚT từ khóa (nghi loại nhầm): {n_core_no_kw:,}")

    print(f"\n{'='*64}")
    print(f"  CHECKLIST VĂN BẢN CỐT LÕI / NGHI THIẾU")
    print(f"{'='*64}")
    print(f"  {'số hiệu':16s} {'trong tmquan?':14s} {'lõi':4s} {'từ_khóa':8s} {'→corpus':8s} tên")
    for sk, ten in CHECK.items():
        if sk in found:
            lt, sc, core, kw, insc, title = found[sk]
            status = "CÓ"
            core_s = "✓" if core else f"✗({sc}/{lt})"
            kw_s = "✓" if kw else "✗ RỚT"
            insc_s = "✓ vào" if insc else "✗ LOẠI"
            print(f"  {sk:16s} {status:14s} {core_s:4s} {kw_s:8s} {insc_s:8s} {ten}")
        else:
            print(f"  {sk:16s} {'KHÔNG CÓ ❌':14s} {'—':4s} {'—':8s} {'—':8s} {ten}")


if __name__ == "__main__":
    main()
