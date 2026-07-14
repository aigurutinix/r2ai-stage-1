"""
scripts/build_url_map.py

Xây map { "<số hiệu>": {"url": ..., "name": ...} } từ file số hiệu + URL (crawl thuvienphapluat)
→ data/doc_url_map.json. App demo (app/service.py) dùng map này để gắn link vào mỗi văn bản trích dẫn.

Đầu vào rất linh hoạt — MỖI DÒNG chỉ cần chứa 1 số hiệu và 1 URL, theo BẤT KỲ thứ tự/định dạng nào:
  - ngăn cách bằng TAB hoặc dấu phẩy (tự nhận);
  - CÓ hoặc KHÔNG có dòng tiêu đề (dòng tiêu đề tự bị bỏ qua vì không khớp mẫu số hiệu/URL);
  - cột "tên văn bản" KHÔNG cần có — tên hiển thị lấy từ law_manifest.json; thiếu thì suy ra từ URL.

Chạy:
    python scripts/build_url_map.py                       # đọc mặc định data/law_urls.csv
    python scripts/build_url_map.py --csv path/to/file.tsv
"""
import os
import re
import json
import argparse
import logging

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import Settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_url_map")

# Số hiệu văn bản Việt Nam, 3 dạng (URL dùng "-" thay "/" nên không nhầm với slug):
#   1) số/năm/loại   vd 108/2025/QH15, 65/2023/NĐ-CP
#   2) số/loại       vd 644/QĐ-TTg, 15/NQ-CP  (không có năm ở giữa)
#   3) số-loại       vd 74-CP, 39-CP          (kiểu cũ)
DOC_NUMBER_RE = re.compile(
    r"\d{1,4}/\d{4}/[A-Za-zĐđ0-9/\-]+"
    r"|\d{1,4}/[A-Za-zĐđ][A-Za-zĐđ0-9/\-]*"
    r"|\d{1,4}-[A-Za-zĐđ]{1,6}"
)
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def normalize_doc_number(raw: str) -> str:
    """Chuẩn hóa số hiệu về khóa thống nhất: lấy đúng cụm khớp regex, uppercase, bỏ khoảng trắng."""
    m = DOC_NUMBER_RE.search(str(raw or ""))
    return m.group(0).strip().upper().replace(" ", "") if m else ""


def name_from_url(url: str) -> str:
    """
    Suy ra tên văn bản (không dấu) từ slug URL thuvienphapluat, dùng làm DỰ PHÒNG khi số hiệu
    không có trong law_manifest.json. Ví dụ:
      .../Luat-Quan-ly-thue-2025-so-108-2025-QH15-675268.aspx
      → "Luat Quan ly thue 2025 so 108 2025 QH15"
    """
    seg = url.rstrip("/").split("/")[-1]
    seg = re.sub(r"\.aspx$", "", seg, flags=re.IGNORECASE)
    seg = re.sub(r"-\d+$", "", seg)          # bỏ đuôi số định danh (vd -675268)
    name = seg.replace("-", " ").strip()
    return name


def iter_cells(path: str):
    """Yield từng danh sách ô của mỗi dòng, tự nhận delimiter tab/phẩy."""
    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read()
    delimiter = "\t" if text.count("\t") >= text.count(",") else ","
    for line in text.splitlines():
        if line.strip():
            yield [c.strip().strip('"') for c in line.split(delimiter)]


def main():
    default_csv = os.path.join(Settings.DATA_DIR, "law_urls.csv")
    ap = argparse.ArgumentParser(description="Build số hiệu → URL map từ file số hiệu + URL")
    ap.add_argument("--csv", default=default_csv,
                    help=f"File đầu vào (tab hoặc phẩy; mặc định: {default_csv})")
    ap.add_argument("--out", default=Settings.DOC_URL_MAP_PATH, help="File JSON đầu ra")
    args = ap.parse_args()

    mapping = {}
    total = skipped = 0
    for cells in iter_cells(args.csv):
        total += 1
        # Quét các ô: ô nào khớp số hiệu → key, ô nào khớp URL → url (không phụ thuộc thứ tự cột).
        key = url = ""
        for c in cells:
            if not key:
                key = normalize_doc_number(c)
            if not url:
                m = URL_RE.search(c)
                if m:
                    url = m.group(0)
        if not key or not url:
            skipped += 1
            continue
        mapping.setdefault(key, {"url": url, "name": name_from_url(url)})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2, sort_keys=True)

    logger.info(f"✅ {len(mapping)} số hiệu → URL vào {args.out} "
                f"(tổng {total} dòng, bỏ qua {skipped} dòng không có đủ số hiệu+URL).")
    for k in list(mapping)[:5]:
        logger.info(f"    {k} -> {mapping[k]['url']}")


if __name__ == "__main__":
    main()
