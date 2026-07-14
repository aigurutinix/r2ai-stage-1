"""Đánh giá hệ thống theo thể lệ cuộc thi — Precision / Recall / F2 macro.

Chạy IN-PROCESS qua RAGPipeline (không cần dựng HTTP server). Hai lớp đo:

  • Track A — END-TO-END (đúng cách chấm của BTC):
      rút 'Điều X' từ `answer` do LLM sinh  →  so với gold 'Điều X'.
  • Track B — RETRIEVAL ceiling (trần truy hồi, không phụ thuộc LLM):
      tập 'Điều X' / (so_ky_hieu, dieu_so) trong top-k hit  →  so với gold.
      Kèm Recall@k và MRR theo cặp (so_ky_hieu, dieu_so).

Track B cho biết giới hạn trên mà khâu sinh có thể đạt; Track A cho biết điểm
thực tế nộp thi. Khoảng cách A↔B = phần LLM bỏ sót / không trích dẫn.

Usage:
    python -m tests.f2_eval --eval-set data/eval_set.json [--top-k 10]
                            [--retrieval-only] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from backend.rag import RAGPipeline
from tests.metrics import PRF, extract_dieu_numbers, macro_average, prf2

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
logger = logging.getLogger("f2_eval")
logger.setLevel(logging.INFO)

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GoldRef:
    so_ky_hieu: str
    dieu_so: int


@dataclass
class EvalCase:
    id: int
    question: str
    gold: list[GoldRef]
    topic: str = ""
    source_text: str = ""

    @staticmethod
    def from_dict(d: dict) -> "EvalCase":
        return EvalCase(
            id=d["id"],
            question=d["question"],
            gold=[GoldRef(g["so_ky_hieu"], int(g["dieu_so"])) for g in d.get("gold", [])],
            topic=d.get("topic", ""),
            source_text=d.get("source_text", ""),
        )


@dataclass
class CaseResult:
    id: int
    topic: str
    question: str
    gold_dieu: list[int]
    # Track A (end-to-end answer)
    answer_dieu: list[int]
    a_prf: dict
    # Track B (retrieval ceiling)
    retr_dieu: list[int]
    b_prf: dict
    # cặp (so_ky_hieu, dieu_so) chặt
    pair_recall: float
    rank_first_gold: int | None  # vị trí gold đầu tiên trong hit (1-based) → MRR
    latency_sec: float
    answer_preview: str = ""


def _gold_dieu_set(case: EvalCase) -> set[str]:
    return {str(g.dieu_so) for g in case.gold}


def _gold_pair_set(case: EvalCase) -> set[tuple[str, str]]:
    return {(g.so_ky_hieu, str(g.dieu_so)) for g in case.gold}


def _hit_pairs(hits: list[dict]) -> list[tuple[str, str]]:
    out = []
    for h in hits:
        p = h.get("payload", {})
        sk, ds = p.get("so_ky_hieu"), p.get("dieu_so")
        if sk is not None and ds is not None:
            out.append((str(sk), str(ds)))
    return out


def evaluate_case(rag: RAGPipeline, case: EvalCase, top_k: int, retrieval_only: bool) -> CaseResult:
    t0 = time.perf_counter()
    hits = rag.retrieve(case.question, top_k=top_k)

    gold_d = _gold_dieu_set(case)
    gold_pairs = _gold_pair_set(case)
    hit_pairs = _hit_pairs(hits)

    # Track B — trần truy hồi
    retr_dieu = {ds for _, ds in hit_pairs}
    b = prf2(retr_dieu, gold_d)

    # cặp chặt: recall theo (so_ky_hieu, dieu_so)
    retr_pairs = set(hit_pairs)
    pair_correct = retr_pairs & gold_pairs
    pair_recall = len(pair_correct) / len(gold_pairs) if gold_pairs else 1.0

    # MRR: hạng của gold-pair đầu tiên trong danh sách hit (theo thứ tự score)
    rank_first = None
    for idx, pr in enumerate(hit_pairs, start=1):
        if pr in gold_pairs:
            rank_first = idx
            break

    # Track A — end-to-end (sinh answer)
    if retrieval_only:
        answer = ""
        a = PRF(0.0, 0.0, 0.0, 0, 0, len(gold_d))
        ans_d: set[int] = set()
    else:
        system, user = rag.build_prompt(case.question, hits)
        answer = rag.llm.complete(system, user)
        ans_d = extract_dieu_numbers(answer)
        a = prf2(ans_d, gold_d)

    return CaseResult(
        id=case.id, topic=case.topic, question=case.question,
        gold_dieu=sorted(gold_d), answer_dieu=sorted(ans_d),
        a_prf=asdict(a), retr_dieu=sorted(retr_dieu), b_prf=asdict(b),
        pair_recall=round(pair_recall, 4), rank_first_gold=rank_first,
        latency_sec=round(time.perf_counter() - t0, 2),
        answer_preview=answer[:240].replace("\n", " "),
    )


def summarize(results: list[CaseResult], retrieval_only: bool) -> dict:
    a_macro = macro_average([PRF(**r.a_prf) for r in results])
    b_macro = macro_average([PRF(**r.b_prf) for r in results])
    ranks = [r.rank_first_gold for r in results]
    mrr = sum((1.0 / r) for r in ranks if r) / len(results) if results else 0.0
    pair_recall_macro = sum(r.pair_recall for r in results) / len(results) if results else 0.0
    lats = [r.latency_sec for r in results]
    return {
        "n_queries": len(results),
        "retrieval_only": retrieval_only,
        "track_A_end_to_end": {  # điểm thi thực tế
            "precision": round(a_macro.precision, 4),
            "recall": round(a_macro.recall, 4),
            "f2": round(a_macro.f2, 4),
        },
        "track_B_retrieval_ceiling": {  # trần truy hồi
            "precision": round(b_macro.precision, 4),
            "recall": round(b_macro.recall, 4),
            "f2": round(b_macro.f2, 4),
        },
        "pair_recall_macro": round(pair_recall_macro, 4),  # (so_ky_hieu, dieu_so) chặt
        "mrr_pair": round(mrr, 4),
        "latency_avg_sec": round(sum(lats) / len(lats), 2) if lats else None,
    }


def render_markdown(results: list[CaseResult], summary: dict, top_k: int) -> str:
    A, B = summary["track_A_end_to_end"], summary["track_B_retrieval_ceiling"]
    L = [
        f"# F2 Eval — corpus-grounded ({summary['n_queries']} truy vấn, top_k={top_k})",
        "",
        "## Kết quả macro",
        "| Track | Precision | Recall | **F2** |",
        "|---|---:|---:|---:|",
        f"| **A. End-to-end** (điểm thi) | {A['precision']} | {A['recall']} | **{A['f2']}** |",
        f"| B. Retrieval ceiling | {B['precision']} | {B['recall']} | **{B['f2']}** |",
        "",
        f"- Pair-recall (so_ky_hieu, Điều) chặt: **{summary['pair_recall_macro']}**",
        f"- MRR (theo cặp): **{summary['mrr_pair']}**",
        f"- Latency trung bình: **{summary['latency_avg_sec']}s**",
        "",
        "> Track B là trần truy hồi (LLM lý tưởng sẽ trích hết). Khoảng cách A↔B = "
        "phần LLM bỏ sót khi sinh câu trả lời.",
        "",
        "## Chi tiết",
        "| ID | Topic | Gold Điều | Retr Điều (top-k) | Answer Điều | A.F2 | rank | Lat |",
        "|---:|---|---|---|---|---:|---:|---:|",
    ]
    for r in results:
        gold = ",".join(map(str, r.gold_dieu)) or "-"
        retr = ",".join(map(str, r.retr_dieu))[:40] or "-"
        ans = ",".join(map(str, r.answer_dieu)) or "-"
        L.append(
            f"| {r.id} | {r.topic} | {gold} | {retr} | {ans} | "
            f"{r.a_prf['f2']:.2f} | {r.rank_first_gold or '-'} | {r.latency_sec}s |"
        )
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="data/eval_set.json")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    eval_path = ROOT / args.eval_set if not Path(args.eval_set).is_absolute() else Path(args.eval_set)
    cases = [EvalCase.from_dict(d) for d in json.loads(eval_path.read_text(encoding="utf-8"))]
    if args.limit:
        cases = cases[: args.limit]
    logger.info("Loaded %d eval cases from %s", len(cases), eval_path)

    rag = RAGPipeline()
    results: list[CaseResult] = []
    for i, c in enumerate(cases, 1):
        r = evaluate_case(rag, c, top_k=args.top_k, retrieval_only=args.retrieval_only)
        results.append(r)
        logger.info(
            "[%d/%d] Q%d %s | A.F2=%.2f B.F2=%.2f rank=%s %.1fs",
            i, len(cases), c.id, c.topic, r.a_prf["f2"], r.b_prf["f2"],
            r.rank_first_gold, r.latency_sec,
        )

    summary = summarize(results, args.retrieval_only)

    out_dir = ROOT / "data"
    (out_dir / "f2_eval_results.json").write_text(
        json.dumps({"summary": summary, "results": [asdict(r) for r in results]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "f2_eval_report.md").write_text(
        render_markdown(results, summary, args.top_k), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info("Report → data/f2_eval_report.md")


if __name__ == "__main__":
    main()
