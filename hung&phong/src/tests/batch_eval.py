"""Batch evaluator — chạy 50 testcase MOJ → /chat_sync, đo retrieval + latency.

Usage:
    python -m tests.batch_eval [--limit 50] [--concurrency 3] [--base http://127.0.0.1:8000]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

# Cho phép chạy `python tests/batch_eval.py` lẫn `python -m tests.batch_eval`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.query_analyzer import analyze
from tests.sample_queries import SAMPLE_QUERIES, TestCase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("batch_eval")


@dataclass
class EvalResult:
    id: int
    category: str
    query: str
    expected_so_ky_hieu: str | None
    expected_in_corpus: bool
    # Tín hiệu detect được trước khi gọi backend:
    detected_filter: dict
    # Kết quả từ backend:
    ok: bool
    latency_sec: float
    error: str | None
    sources_count: int
    top_source_so_ky_hieu: str | None
    top_source_score: float | None
    top_source_ten: str | None
    answer_preview: str
    expected_hit: bool  # top_source khớp expected_so_ky_hieu?


async def run_one(
    client: httpx.AsyncClient,
    tc: TestCase,
    sem: asyncio.Semaphore,
) -> EvalResult:
    filters = analyze(tc.query)
    detected = {
        "so_ky_hieu": filters.so_ky_hieu,
        "dieu_so": filters.dieu_so,
        "loai_van_ban": filters.loai_van_ban,
    }
    async with sem:
        t0 = time.perf_counter()
        try:
            r = await client.post(
                "/chat_sync",
                json={"query": tc.query, "top_k": 9},
                timeout=120.0,
            )
            r.raise_for_status()
            data = r.json()
            elapsed = time.perf_counter() - t0
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.warning("Q%-2d FAIL (%.2fs): %s", tc.id, elapsed, e)
            return EvalResult(
                id=tc.id, category=tc.category, query=tc.query,
                expected_so_ky_hieu=tc.expected_so_ky_hieu,
                expected_in_corpus=tc.expected_in_corpus,
                detected_filter=detected,
                ok=False, latency_sec=round(elapsed, 2), error=str(e)[:300],
                sources_count=0, top_source_so_ky_hieu=None,
                top_source_score=None, top_source_ten=None,
                answer_preview="", expected_hit=False,
            )

    sources = data.get("sources", []) or []
    answer = data.get("answer", "") or ""
    top = sources[0] if sources else {}
    top_sk = top.get("so_ky_hieu")
    expected_hit = bool(
        tc.expected_so_ky_hieu
        and top_sk
        and tc.expected_so_ky_hieu.lower() in str(top_sk).lower()
    )
    logger.info(
        "Q%-2d %s lat=%.2fs src=%d top=%s hit=%s",
        tc.id, tc.category, elapsed, len(sources), top_sk, expected_hit,
    )
    return EvalResult(
        id=tc.id, category=tc.category, query=tc.query,
        expected_so_ky_hieu=tc.expected_so_ky_hieu,
        expected_in_corpus=tc.expected_in_corpus,
        detected_filter=detected,
        ok=True, latency_sec=round(elapsed, 2), error=None,
        sources_count=len(sources),
        top_source_so_ky_hieu=top_sk,
        top_source_score=top.get("score"),
        top_source_ten=top.get("ten_van_ban"),
        answer_preview=answer[:300].replace("\n", " "),
        expected_hit=expected_hit,
    )


async def main_async(base: str, concurrency: int, limit: int | None) -> list[EvalResult]:
    queries = SAMPLE_QUERIES[:limit] if limit else SAMPLE_QUERIES
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=base) as client:
        tasks = [run_one(client, tc, sem) for tc in queries]
        results = await asyncio.gather(*tasks)
    return results


def summarize(results: list[EvalResult]) -> dict:
    by_cat: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)

    total = len(results)
    ok = sum(1 for r in results if r.ok)
    hits = sum(1 for r in results if r.expected_hit)
    expected_in_corpus = sum(1 for r in results if r.expected_in_corpus)
    hits_when_in_corpus = sum(
        1 for r in results if r.expected_in_corpus and r.expected_hit
    )
    latencies = [r.latency_sec for r in results if r.ok]

    cat_summary = {}
    for cat, items in sorted(by_cat.items()):
        cat_total = len(items)
        cat_ok = sum(1 for r in items if r.ok)
        cat_hits = sum(1 for r in items if r.expected_hit)
        cat_in_corpus = sum(1 for r in items if r.expected_in_corpus)
        cat_lat = [r.latency_sec for r in items if r.ok]
        cat_summary[cat] = {
            "total": cat_total,
            "ok": cat_ok,
            "expected_in_corpus": cat_in_corpus,
            "top_source_hit": cat_hits,
            "avg_latency": round(statistics.mean(cat_lat), 2) if cat_lat else None,
        }

    return {
        "total": total,
        "ok": ok,
        "failed": total - ok,
        "expected_in_corpus": expected_in_corpus,
        "top_source_hit_total": hits,
        "top_source_hit_when_expected_in_corpus": hits_when_in_corpus,
        "latency": {
            "avg": round(statistics.mean(latencies), 2) if latencies else None,
            "p50": round(statistics.median(latencies), 2) if latencies else None,
            "p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 2)
            if len(latencies) >= 5 else None,
            "max": round(max(latencies), 2) if latencies else None,
        },
        "by_category": cat_summary,
    }


def render_markdown(results: list[EvalResult], summary: dict) -> str:
    lines = ["# Batch Eval — 50 testcase MOJ"]
    lines.append("")
    lines.append("## Tổng quan")
    lines.append(f"- Total: **{summary['total']}**  ·  OK: **{summary['ok']}**  ·  Failed: **{summary['failed']}**")
    lines.append(f"- Expected có trong corpus: **{summary['expected_in_corpus']}**")
    lines.append(f"- Top-source hit (toàn bộ): **{summary['top_source_hit_total']}/{summary['total']}**")
    lines.append(
        f"- Top-source hit (chỉ tính câu kỳ vọng có trong corpus): "
        f"**{summary['top_source_hit_when_expected_in_corpus']}/{summary['expected_in_corpus']}**"
    )
    lat = summary["latency"]
    lines.append(
        f"- Latency: avg **{lat['avg']}s** · p50 **{lat['p50']}s** · p95 **{lat['p95']}s** · max **{lat['max']}s**"
    )
    lines.append("")
    lines.append("## Theo category")
    lines.append("| Category | Total | OK | Expect-in-corpus | Top-source hit | Avg latency |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cat, s in summary["by_category"].items():
        lines.append(
            f"| {cat} | {s['total']} | {s['ok']} | {s['expected_in_corpus']} | "
            f"{s['top_source_hit']} | {s['avg_latency']}s |"
        )
    lines.append("")
    lines.append("## Chi tiết")
    lines.append("| ID | Cat | Query | Expect | Detected | Top src | Score | Hit | Lat | Answer preview |")
    lines.append("|---:|---|---|---|---|---|---:|---|---:|---|")
    for r in results:
        q = r.query.replace("|", "\\|")[:80]
        ans = r.answer_preview.replace("|", "\\|")[:120]
        det = ";".join(f"{k}={v}" for k, v in r.detected_filter.items() if v) or "-"
        top = (r.top_source_so_ky_hieu or "-")
        score = f"{r.top_source_score:.3f}" if r.top_source_score is not None else "-"
        hit = "✅" if r.expected_hit else ("⚪" if not r.expected_so_ky_hieu else "❌")
        lines.append(
            f"| {r.id} | {r.category} | {q} | {r.expected_so_ky_hieu or '-'} | {det} | "
            f"{top} | {score} | {hit} | {r.latency_sec}s | {ans} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    results = asyncio.run(main_async(args.base, args.concurrency, args.limit))
    total_elapsed = time.perf_counter() - t0

    summary = summarize(results)
    summary["total_wall_time_sec"] = round(total_elapsed, 2)

    json_path = out_dir / "batch_eval_results.json"
    md_path = out_dir / "batch_eval_report.md"

    json_path.write_text(
        json.dumps(
            {"summary": summary, "results": [asdict(r) for r in results]},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(results, summary), encoding="utf-8")

    logger.info("Wall time: %.1fs", total_elapsed)
    logger.info("JSON:     %s", json_path)
    logger.info("Markdown: %s", md_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
