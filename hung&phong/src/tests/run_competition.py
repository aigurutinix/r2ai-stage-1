"""Chạy bộ câu hỏi cuộc thi (R2AIStage1) qua RAG, lưu answer + nguồn để verify.

BTC chỉ cung cấp câu hỏi (không có đáp án) → script này sinh câu trả lời thật của
hệ thống, kèm các điều luật trích dẫn + nguồn truy hồi, để con người/LLM-judge
đọc và chấm theo thể lệ §4.2 (Căn cứ chính xác pháp luật, v.v.).

Usage:
    python -m tests.run_competition --in <path> [--limit 60] [--offset 0] [--top-k 10]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from backend.rag import RAGPipeline
from tests.metrics import extract_dieu_numbers

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
logger = logging.getLogger("run_competition")
logger.setLevel(logging.INFO)
ROOT = Path(__file__).resolve().parents[1]


def _src(h: dict) -> dict:
    p = h.get("payload", {})
    return {
        "so_ky_hieu": p.get("so_ky_hieu"),
        "dieu_so": p.get("dieu_so"),
        "dieu_tieu_de": p.get("dieu_tieu_de"),
        "loai_van_ban": p.get("loai_van_ban"),
        "title": p.get("title"),
        "tinh_trang": p.get("tinh_trang_hieu_luc"),
        "score": round(h.get("score") or 0, 4),
        "snippet": (p.get("text") or "")[:200].replace("\n", " "),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="data/competition_answers.json")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    questions = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    sel = questions[args.offset: args.offset + args.limit]
    logger.info("Loaded %d questions, running %d (offset %d)", len(questions), len(sel), args.offset)

    rag = RAGPipeline()
    out = []
    for i, q in enumerate(sel, 1):
        t0 = time.perf_counter()
        hits = rag.retrieve(q["question"], top_k=args.top_k)
        system, user = rag.build_prompt(q["question"], hits)
        answer = rag.llm.complete(system, user)
        dt = time.perf_counter() - t0
        rec = {
            "id": q["id"],
            "question": q["question"],
            "answer": answer,
            "answer_dieu": sorted(extract_dieu_numbers(answer)),
            "n_citations": len(extract_dieu_numbers(answer)),
            "sources": [_src(h) for h in hits],
            "latency_sec": round(dt, 2),
        }
        out.append(rec)
        logger.info("[%d/%d] id=%d cites=%d %.1fs", i, len(sel), q["id"],
                    rec["n_citations"], dt)
        # ghi tăng dần để không mất nếu dừng giữa chừng
        Path(ROOT / args.out).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    n_cited = sum(1 for r in out if r["n_citations"] > 0)
    logger.info("DONE %d câu | có trích dẫn Điều: %d (%.0f%%) | avg cites/answer %.1f",
                len(out), n_cited, 100 * n_cited / len(out) if out else 0,
                sum(r["n_citations"] for r in out) / len(out) if out else 0)


if __name__ == "__main__":
    main()
