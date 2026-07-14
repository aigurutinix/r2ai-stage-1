"""Sinh bộ đánh giá có nhãn vàng (gold) bám sát corpus.

Vì BTC giữ kín đáp án, ta tự dựng eval set **bám corpus**: lấy các Điều có thật
trong những luật trọng tâm DN/SME, rồi dùng LLM (qwen) sinh một câu hỏi tự nhiên
mà Điều đó trả lời được. Gold = (so_ky_hieu, dieu_so) của chính Điều nguồn →
nhãn đúng theo cấu trúc, không phải đoán số điều.

Câu hỏi KHÔNG nhắc "Điều X" / số hiệu luật → buộc hệ thống phải tự truy hồi.

Usage:
    python -m tests.build_eval_set [--per-law 6] [--out data/eval_set.json]

Lưu ý: chạy SAU khi ingest xong (tránh tranh GPU với bge-m3).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pyarrow.parquet as pq

from backend.config import get_settings
from backend.llm import LLMClient
from ingest.parse import parse_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("build_eval_set")

ROOT = Path(__file__).resolve().parents[1]

# Luật trọng tâm DN/SME — định danh bằng so_ky_hieu (tin cậy hơn title vì
# dataset có title lộn xộn + bản rỗng trùng số). Đây là các bản ĐÃ XÁC NHẬN
# có nội dung đầy đủ trong corpus (xem scripts/_list_laws.py).
TARGET_LAWS: list[tuple[str, str]] = [
    ("Luật DN 2020", "59/2020/QH14"),
    ("Hỗ trợ DNNVV", "04/2017/QH14"),
    ("Luật Đầu tư 2020", "61/2020/QH14"),
    ("Bộ luật Lao động", "10/2012/QH13"),
    ("Quản lý thuế", "78/2006/QH11"),
    ("Luật Kế toán", "88/2015/QH13"),
    ("Luật Cạnh tranh", "23/2018/QH14"),
    ("Kinh doanh bảo hiểm", "08/2022/QH15"),
    ("Luật Chứng khoán", "70/2006/QH11"),
    ("Luật Phá sản", "21/2004/QH11"),
]
_TARGET_SK = {sk: topic for topic, sk in TARGET_LAWS}

GEN_SYSTEM = (
    "Bạn là chuyên gia pháp lý Việt Nam. Nhiệm vụ: từ nội dung MỘT điều luật, "
    "đặt đúng MỘT câu hỏi tự nhiên mà người dân/doanh nghiệp có thể hỏi và điều "
    "luật này trả lời được. Yêu cầu: KHÔNG nhắc tới số điều, số hiệu hay tên văn "
    "bản; câu hỏi tự đứng độc lập, rõ ràng, tiếng Việt; chỉ in đúng câu hỏi."
)
GEN_USER = "Nội dung điều luật:\n\"\"\"\n{article}\n\"\"\"\n\nCâu hỏi:"

def _clean_question(raw: str) -> str:
    q = raw.strip().strip('"').strip()
    q = re.sub(r"^(Câu hỏi|Question)\s*[:\-]\s*", "", q, flags=re.I).strip()
    # nếu model trả nhiều dòng, lấy dòng câu hỏi đầu tiên có dấu ?
    for line in q.splitlines():
        line = line.strip().strip('"')
        if "?" in line:
            return line
    return q.splitlines()[0].strip() if q else q


def _n_usable_dieu(row: dict) -> int:
    parsed = parse_document(row)
    if not parsed:
        return 0
    return sum(1 for x in parsed.dieus if x.dieu_so > 0 and x.char_len >= 200)


def _pick_docs(parquet_path: Path) -> dict[str, dict]:
    """Mỗi luật trọng tâm (theo so_ky_hieu) → chọn ROW có nhiều Điều dùng được
    nhất. Dataset có bản rỗng trùng so_ky_hieu nên KHÔNG chọn theo năm/đầu tiên,
    mà theo số Điều thực sự parse được → luôn lấy bản đầy đủ nội dung."""
    pf = pq.ParquetFile(str(parquet_path))
    best: dict[str, dict] = {}
    for batch in pf.iter_batches(batch_size=512):
        for row in batch.to_pylist():
            sk = str(row.get("so_ky_hieu") or "")
            topic = _TARGET_SK.get(sk)
            if topic is None:
                continue
            n = _n_usable_dieu(row)
            cur = best.get(topic)
            if cur is None or n > cur["_n_dieu"]:
                best[topic] = {"_n_dieu": n, "_topic": topic, "row": row}
    return best


def build(parquet_path: Path, per_law: int, out_path: Path) -> None:
    docs = _pick_docs(parquet_path)
    logger.info("Matched %d luật trọng tâm: %s", len(docs),
                ", ".join(f"{t}({d['_n_dieu']}đ)" for t, d in docs.items()))

    llm = LLMClient()
    cases: list[dict] = []
    cid = 0
    for topic, d in docs.items():
        parsed = parse_document(d["row"])
        if not parsed:
            logger.warning("  %s: parse fail", topic)
            continue
        # Điều có nội dung đủ dài, bỏ Điều 0 (preamble) → sample đều
        dieus = [x for x in parsed.dieus if x.dieu_so > 0 and x.char_len >= 200]
        if not dieus:
            continue
        step = max(1, len(dieus) // per_law)
        picked = dieus[::step][:per_law]
        logger.info("  %s [%s]: %d điều khả dụng → lấy %d",
                    topic, parsed.so_ky_hieu, len(dieus), len(picked))
        for dieu in picked:
            article = f"Điều {dieu.dieu_so}. {dieu.dieu_tieu_de}\n{dieu.text}"[:3000]
            try:
                q = _clean_question(llm.complete(GEN_SYSTEM, GEN_USER.format(article=article)))
            except Exception as e:
                logger.warning("    gen fail Điều %d: %s", dieu.dieu_so, e)
                continue
            if not q or "?" not in q:
                continue
            cid += 1
            cases.append({
                "id": cid,
                "question": q,
                "gold": [{"so_ky_hieu": parsed.so_ky_hieu, "dieu_so": dieu.dieu_so}],
                "topic": topic,
                "source_text": (dieu.dieu_tieu_de or dieu.text[:120]),
            })

    out_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved %d eval cases → %s", len(cases), out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-law", type=int, default=6)
    ap.add_argument("--out", default="data/eval_set.json")
    args = ap.parse_args()

    s = get_settings()
    cache_dir = Path(s.hf_cache_dir).resolve()
    parquet_path = cache_dir / "vbpl_scope.parquet"
    if not parquet_path.exists():
        parquet_path = cache_dir / "vbpl_full.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Không thấy parquet ở {cache_dir}")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    build(parquet_path, args.per_law, out_path)


if __name__ == "__main__":
    main()
