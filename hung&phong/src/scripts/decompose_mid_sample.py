"""Soi thử phân rã "VỪA" trên các câu mà bản SÂU (v22) đã trôi dạt.

In song song: CÂU GỐC / NÔNG (subqueries.json) / SÂU (subqueries_deep.json) / VỪA (mới)
để mắt người đối chiếu xem "vừa" có giữ neo pháp lý không. Sequential (1 LLM) cho gọn.

Chạy: PYTHONUTF8=1 PYTHONPATH=. python scripts/decompose_mid_sample.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.decompose_mid import decompose_mid
from backend.llm import LLMClient

QFILE = Path("C:/Users/PHONG/Downloads/R2AIStage1DATA.json")
NONG = ROOT / "data/subqueries.json"
SAU = ROOT / "data/subqueries_deep.json"
OUT = ROOT / "data/subqueries_mid_sample.json"

# 20 câu bản SÂU đã trôi dạt (từ phân tích 4 sub-agent Review #5)
SAMPLE_IDS = [1719, 884, 950, 554, 707, 861, 954, 1425, 1600, 916,
              863, 784, 1118, 1013, 1838, 292, 561, 865, 1629, 1881]


def fmt(subs: list[str], label: str) -> str:
    if not subs:
        return f"  {label}: (không có)"
    lines = [f"  {label} ({len(subs)} vế):"]
    for i, s in enumerate(subs, 1):
        lines.append(f"     {i}. {s}")
    return "\n".join(lines)


def main() -> None:
    qs = {int(q["id"]): q["question"] for q in json.loads(QFILE.read_text(encoding="utf-8"))}
    nong = json.loads(NONG.read_text(encoding="utf-8"))
    sau = json.loads(SAU.read_text(encoding="utf-8"))

    llm = LLMClient()
    result: dict[str, list[str]] = {}

    for qid in SAMPLE_IDS:
        q = qs.get(qid, "")
        if not q:
            print(f"\n### id {qid}: (KHÔNG tìm thấy câu hỏi)\n", flush=True)
            continue
        mid = decompose_mid(q, llm)
        result[str(qid)] = mid
        print("\n" + "=" * 100, flush=True)
        print(f"### id {qid}", flush=True)
        print(f"  GỐC: {q}", flush=True)
        print(fmt(nong.get(str(qid), []), "NÔNG"), flush=True)
        print(fmt(sau.get(str(qid), []), "SÂU "), flush=True)
        print(fmt(mid, ">>> VỪA"), flush=True)

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 100, flush=True)
    print(f"XONG {len(result)} câu → {OUT}", flush=True)


if __name__ == "__main__":
    main()
