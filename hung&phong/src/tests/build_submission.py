"""Sinh file nộp bài cuộc thi từ R2AIStage1DATA.json.

Mỗi câu xuất ra:
  - id, question
  - answer        : câu trả lời THẬT do chatbot sinh (chứa các MÃ TRÍCH DẪN [Điều X, Luật ...])
  - relevant_docs : ["<so_ky_hieu>|<tên VB>", ...]      (các VB chatbot dẫn trong answer)
  - relevant_articles: ["<so_ky_hieu>|<tên VB>|Điều X", ...]

relevant_* lấy từ các NGUỒN truy hồi mà câu trả lời THỰC SỰ dẫn (khớp cả so_ky_hieu
lẫn "Điều X" trong answer) → định danh chuẩn theo metadata, không bịa.

Usage:
    python -m tests.build_submission --in <path> [--out data/submission.json]
                                     [--limit N] [--offset 0] [--top-k 10]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

from backend.rag import RAGPipeline

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
logger = logging.getLogger("build_submission")
logger.setLevel(logging.INFO)
ROOT = Path(__file__).resolve().parents[1]


def _law_name(p: dict) -> str:
    loai = (p.get("loai_van_ban") or "").strip()
    title = (p.get("title") or "").strip()
    # vbpl-vn title hay có hậu tố " số" thừa (vd "Luật Đấu thầu số") → bỏ.
    title = re.sub(r"\s+số:?\s*$", "", title).strip()
    # Tránh lặp loại nếu title đã bắt đầu bằng loại (vd "Luật Đấu thầu" → khỏi thêm "Luật").
    if loai and title.lower().startswith(loai.lower()):
        return title
    return f"{loai} {title}".strip() if loai else title


def build_relevant(answer: str, hits: list[dict]) -> tuple[list[str], list[str]]:
    """Trả về (relevant_docs, relevant_articles) — VB/điều mà answer thực sự dẫn."""
    docs: list[str] = []
    arts: list[str] = []
    seen_doc: set[str] = set()
    seen_art: set[str] = set()

    def _add(h: dict) -> None:
        p = h.get("payload", {})
        sk = p.get("so_ky_hieu")
        ds = p.get("dieu_so")
        if not sk or ds is None or int(ds) <= 0:
            return
        sk = str(sk)
        name = _law_name(p)
        doc = f"{sk}|{name}"
        art = f"{sk}|{name}|Điều {ds}"
        if doc not in seen_doc:
            seen_doc.add(doc)
            docs.append(doc)
        if art not in seen_art:
            seen_art.add(art)
            arts.append(art)

    def _dieu_in(answer: str, ds) -> bool:
        return f"Điều {ds}" in answer

    # Tier 1: answer dẫn cả so_ky_hieu LẪN "Điều X" (chắc chắn nhất).
    for h in hits:
        p = h.get("payload", {})
        sk, ds = p.get("so_ky_hieu"), p.get("dieu_so")
        if sk and ds is not None and str(sk) in answer and _dieu_in(answer, ds):
            _add(h)
    if arts:
        return docs, arts

    # Tier 2: answer dẫn "Điều X" (không thấy số hiệu) → khớp theo số điều.
    for h in hits:
        p = h.get("payload", {})
        ds = p.get("dieu_so")
        if ds is not None and _dieu_in(answer, ds):
            _add(h)
    if arts:
        return docs, arts

    # Tier 3: answer không dẫn rõ → lấy top-3 nguồn truy hồi (đảm bảo có căn cứ).
    for h in hits[:3]:
        _add(h)
    return docs, arts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="data/submission.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    questions = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    sel = questions[args.offset:] if args.limit is None else questions[args.offset: args.offset + args.limit]
    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Tổng %d câu, chạy %d (offset %d) → %s", len(questions), len(sel), args.offset, out_path)

    rag = RAGPipeline()
    out: list[dict] = []
    for i, q in enumerate(sel, 1):
        t0 = time.perf_counter()
        hits = rag.retrieve(q["question"], top_k=args.top_k)
        system, user = rag.build_prompt(q["question"], hits)
        answer = rag.llm.complete(system, user)
        docs, arts = build_relevant(answer, hits)
        out.append({
            "id": q["id"],
            "question": q["question"],
            "answer": answer,
            "relevant_docs": docs,
            "relevant_articles": arts,
        })
        if i % 5 == 0 or i == len(sel):
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[%d/%d] id=%d docs=%d arts=%d %.1fs",
                    i, len(sel), q["id"], len(docs), len(arts), time.perf_counter() - t0)

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("XONG %d câu → %s", len(out), out_path)


if __name__ == "__main__":
    main()
