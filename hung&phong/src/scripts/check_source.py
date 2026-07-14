"""Kiểm nguồn: BLDS 91/2015 có thật không + liệt kê Bộ luật + sample Luật mới."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from backend.config import get_settings
from datasets import load_dataset

s = get_settings()
ds = load_dataset("tmquan/vbpl-vn", "documents", split="train", cache_dir=s.hf_cache_dir)

# 1. BLDS 91/2015 đích danh (mọi scope)
print("=== Tìm BLDS 91/2015/QH13 (mọi scope) ===")
found = [r for r in ds if str(r.get("doc_number"))=="91/2015/QH13"]
for r in found:
    print(f"  {r['doc_number']} | {r['legal_type']} | {r['title'][:50]} | scope={r['scope']}")
if not found:
    print("  KHÔNG có 91/2015/QH13 trong nguồn.")

# 2. Mọi 'Bộ luật' trong nguồn
print("\n=== Tất cả legal_type='Bộ luật' trong nguồn ===")
for r in ds:
    if str(r.get("legal_type"))=="Bộ luật":
        print(f"  {r['doc_number']} | {r['title'][:45]} | {r['year']} | scope={r['scope']}")

# 3. Tìm title bắt đầu 'Bộ luật Dân sự' (mọi số hiệu)
print("\n=== Title chứa 'bộ luật dân sự' ===")
for r in ds:
    t=str(r.get("title") or "").lower()
    if "bộ luật dân sự" in t:
        print(f"  {r['doc_number']} | {r['title'][:50]} | {r['year']} | type={r['legal_type']}")
