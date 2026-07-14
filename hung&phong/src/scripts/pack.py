"""Đóng gói bài nộp đúng format BTC (project.md 5.2): 1 file `results.json`, zip PHẲNG.

Dùng: PYTHONUTF8=1 python scripts/pack.py data/submission_xxx.json [data/submission_xxx.zip]
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path


def main() -> None:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".zip")
    data = json.loads(src.read_text(encoding="utf-8"))
    assert isinstance(data, list), "results phải là MẢNG"
    empty = [r["id"] for r in data if not r.get("relevant_articles")]
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("results.json", json.dumps(data, ensure_ascii=False))
    with zipfile.ZipFile(dst) as z:
        names = z.namelist()
    assert names == ["results.json"], f"zip sai: {names}"
    print(f"OK → {dst} | {len(data)} câu | {len(empty)} câu rỗng relevant_articles | zip={names}")


if __name__ == "__main__":
    main()
