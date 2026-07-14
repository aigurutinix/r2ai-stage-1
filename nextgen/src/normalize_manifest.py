"""
normalize_manifest.py
Chuẩn hoá trường `btc_standard_string` trong data/law_manifest.json về đúng format BTC:
    "<số hiệu>|<tên văn bản sạch>"

Trước khi chuẩn hoá (bị phình, lặp số hiệu + loại văn bản + đuôi 'áp dụng YYYY'):
    04/2007/QH12|Luật 04/2007/QH12 Luật thuế thu nhập cá nhân 2007 số 04/2007/QH12 áp dụng 2024
Sau khi chuẩn hoá:
    04/2007/QH12|Luật thuế thu nhập cá nhân 2007

Quy tắc làm sạch TÊN (áp lên law_name, KHÔNG ghép thêm document_type vì law_name đã có sẵn):
    - Bỏ tiền tố "Toàn văn:".
    - Bỏ tiền tố lặp "<document_type> <số hiệu> " nếu có.
    - Bỏ cụm "số <số hiệu>".
    - Bỏ đuôi "áp dụng [năm] YYYY ...".
    - Gộp khoảng trắng.

Giữ nguyên số hiệu nhúng giữa tên (vd "Nghị quyết 98/2023/QH15 về ...") vì đúng quy ước BTC.

Chạy:
    python normalize_manifest.py            # ghi đè data/law_manifest.json (có backup)
    python normalize_manifest.py --dry-run  # chỉ in thử, không ghi
"""

import os
import re
import json
import shutil
import argparse
from datetime import datetime

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "data", "law_manifest.json")


def clean_law_name(law_name: str, doc_id: str, document_type: str) -> str:
    # Dùng THẲNG law_name (đã có sẵn loại văn bản + số hiệu đúng quy ước BTC),
    # KHÔNG ghép thêm document_type/doc_id -> tránh lặp như btc cũ. Chỉ làm sạch nhiễu đuôi.
    name = law_name or ""
    # 1) Bỏ "Toàn văn:" đầu chuỗi
    name = re.sub(r"^\s*Toàn\s*văn\s*:\s*", "", name, flags=re.IGNORECASE)
    # 2) Bỏ cụm "số <doc_id>" ở bất kỳ đâu
    name = re.sub(r"\s*số\s+" + re.escape(doc_id), "", name, flags=re.IGNORECASE)
    # 3) Bỏ đuôi "áp dụng [năm] YYYY ..." (chỉ khi theo sau là 4 chữ số -> tránh cắt nhầm
    #    các tên có chữ 'áp dụng' mang nghĩa thực, vd 'Nghị quyết ... áp dụng thuế ...')
    name = re.sub(r"\s*áp\s*dụng\s+(năm\s+)?\d{4}.*$", "", name, flags=re.IGNORECASE)
    # 4) Gộp khoảng trắng + bỏ dấu phân tách thừa
    name = re.sub(r"\s+", " ", name).strip(" -–|")
    return name


def build_btc(doc_id: str, clean_name: str) -> str:
    return f"{doc_id}|{clean_name}"


def main(args):
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = []
    for doc_id, entry in manifest.items():
        if not isinstance(entry, dict):
            continue
        old_btc = entry.get("btc_standard_string", "")
        cname = clean_law_name(entry.get("law_name", ""), doc_id, entry.get("document_type", ""))
        if not cname:  # phòng trường hợp tên rỗng -> fallback loại + số hiệu
            cname = f"{entry.get('document_type','Văn bản')} {doc_id}"
        new_btc = build_btc(doc_id, cname)
        if len(samples) < 18 and old_btc != new_btc:
            samples.append((doc_id, old_btc, new_btc))
        entry["btc_standard_string"] = new_btc
        # Lưu thêm tên sạch để các module khác (verifier rule3) dùng nếu cần
        entry["clean_name"] = cname

    print("=" * 100)
    print("MẪU TRƯỚC -> SAU:")
    for doc_id, old, new in samples:
        print(f"\n[{doc_id}]")
        print(f"  CŨ : {old}")
        print(f"  MỚI: {new}")
    print("=" * 100)

    if args.dry_run:
        print("\n[dry-run] KHÔNG ghi file.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = MANIFEST_PATH + f".bak_{ts}"
    shutil.copy(MANIFEST_PATH, bak)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Đã chuẩn hoá {len(manifest)} văn bản. Backup: {os.path.basename(bak)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args())
