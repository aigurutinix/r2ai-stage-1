"""Generate competition submission (results.json + flat submission.zip).

Design: RETRIEVAL is authoritative for citations. relevant_docs/relevant_articles
are filled directly from retrieved articles (not parsed back from the LLM), then
every cited "Điều N" is guaranteed to appear in the answer text so the auto-grader
extracts them. This decouples the IR score from LLM citation compliance.

Run:
    python generate_submission.py                # full (needs Ollama + ChromaDB)
    python generate_submission.py --no-llm       # IR-only answers (no Ollama)
    python generate_submission.py --top-k 6
"""
import argparse
import json
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

import local_models_config as cfg
from retrieval_cutoff import apply_cutoff   # pure-python, không kéo theo deps nặng
# local_rag_engine imported lazily (only when retrieving locally) — `--retrieved` mode
# (đọc cache từ Colab) không cần sentence-transformers/rank_bm25 cài ở local.

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(REPO_ROOT, "results.json")
ZIP_PATH = os.path.join(REPO_ROOT, "submission.zip")


def build_citation_fields(retrieved):
    """From retrieved articles → (relevant_docs, relevant_articles) competition strings."""
    docs, arts = [], []
    for d in retrieved:
        code, name, art = d.get("doc_number", ""), d.get("clean_name", ""), d.get("article", "")
        if not code or not art:
            continue
        ds = f"{code}|{name}"
        as_ = f"{code}|{name}|{art}"
        if ds not in docs:
            docs.append(ds)
        if as_ not in arts:
            arts.append(as_)
    return docs, arts


def ensure_citations_in_answer(answer, retrieved):
    """Guarantee every retrieved 'Điều N' appears in answer text (grader parses it)."""
    present = set(re.findall(r"Điều\s+(\d+)", answer))
    missing = {}
    for d in retrieved:
        m = re.match(r"Điều\s+(\d+)", d.get("article", ""))
        if m and m.group(1) not in present:
            missing.setdefault(d.get("clean_name", ""), []).append(d["article"])
    if not missing:
        return answer
    parts = [f"{', '.join(arts)} ({name})" for name, arts in missing.items()]
    return answer.rstrip() + "\n\nCăn cứ pháp lý áp dụng: " + "; ".join(parts) + "."


def template_answer(question, retrieved):
    """Deterministic answer for --no-llm mode (still contains every Điều N)."""
    if not retrieved:
        return ("Chưa tìm thấy căn cứ pháp lý phù hợp trong cơ sở dữ liệu. "
                "Khuyến nghị tham vấn luật sư chuyên nghiệp.")
    lines = [f"- {d['article']} {d.get('clean_name','')}: {d.get('title','')}" for d in retrieved]
    return "Căn cứ pháp lý liên quan đến câu hỏi:\n" + "\n".join(lines)


def _checkpoint(results):
    json.dump(sorted(results, key=lambda r: r["id"]), open(RESULTS_PATH, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=cfg.RETRIEVE_TOP_K)
    ap.add_argument("--no-llm", action="store_true", help="skip LLM; answer = template liệt kê điều luật")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--questions", default=cfg.TEST_QUESTIONS,
                    help="questions json (mặc định 20 mock; truyền stage1_questions.json cho bài thật)")
    ap.add_argument("--retrieved", default=None,
                    help="dùng retrieved.json (từ Colab) → bỏ qua retrieval local, CHỈ chạy LLM")
    args = ap.parse_args()

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[:args.limit]

    # Nguồn retrieval: cache từ Colab (--retrieved) HOẶC engine local
    cache, engine = None, None
    if args.retrieved:
        cache = {int(k): v for k, v in json.load(open(args.retrieved, encoding="utf-8")).items()}
        print(f"Dùng {args.retrieved}: {len(cache)} câu → bỏ qua retrieval local (chỉ chạy LLM).")
    else:
        from local_rag_engine import LocalLegalRAGEngine
        engine = LocalLegalRAGEngine(use_reranker=True)

    llm = None
    if not args.no_llm:
        from local_llm_client import LocalLLMClient
        llm = LocalLLMClient()

    # Resume từ checkpoint (results.json) — quan trọng cho LLM 2000 câu chạy dài
    results, done = [], set()
    if os.path.exists(RESULTS_PATH):
        try:
            results = json.load(open(RESULTS_PATH, encoding="utf-8"))
            done = {r["id"] for r in results}
            if done:
                print(f"resume: đã có {len(done)} câu trong results.json")
        except Exception:
            results, done = [], set()

    for q in questions:
        qid, qtext = int(q["id"]), q["question"]
        if qid in done:
            continue
        if cache is not None:   # cache giàu (top-N + score) → áp cutoff tại đây
            retrieved = apply_cutoff(cache.get(qid, []), args.top_k,
                                     cfg.RETRIEVE_MIN_SCORE, cfg.RETRIEVE_MARGIN, score_key="score")
        else:
            retrieved = engine.retrieve(qtext, top_k=args.top_k)
        rel_docs, rel_arts = build_citation_fields(retrieved)

        if llm and retrieved:
            system, user = llm.build_prompt(qtext, retrieved)
            try:
                answer = llm.chat(system, user)
            except Exception as e:
                print(f"  [q{qid}] LLM error ({e}); template.")
                answer = template_answer(qtext, retrieved)
        else:
            answer = template_answer(qtext, retrieved)

        answer = ensure_citations_in_answer(answer, retrieved)
        results.append({
            "id": qid, "question": qtext, "answer": answer,
            "relevant_docs": rel_docs, "relevant_articles": rel_arts,
        })
        if len(results) % 50 == 0:
            _checkpoint(results)
            print(f"  checkpoint {len(results)}/{len(questions)}")

    results.sort(key=lambda r: r["id"])
    if len(results) != len(questions):
        print(f"⚠ Cảnh báo: {len(results)} câu / {len(questions)} câu hỏi (thiếu = bài không hợp lệ).")

    _checkpoint(results)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(RESULTS_PATH, arcname="results.json")   # flat: results.json at root

    print(f"\n✓ {RESULTS_PATH} + {ZIP_PATH} | {len(results)} câu.")


if __name__ == "__main__":
    main()
