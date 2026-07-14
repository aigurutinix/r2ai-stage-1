"""v12: Pipeline đầy đủ — retrieve + rerank + multi-query + collapse + adaptive + judge.

Khác v9/v10:
  - Retrieve TOÀN BỘ câu (kể cả không phân rã) bằng AITeamVN-v2 mới nhất
    thay vì dùng v8_scored (v8_scored vẫn đúng model nhưng v12 retrieve trực tiếp → sạch hơn)
  - Judge tích hợp NGAY TRONG pipeline (không cần chạy offline 2 bước như v10)
  - Judge chỉ áp dụng khi có >= 2 điều sau adaptive → tránh lọc câu chỉ có 1 điều

Khác v11 (sai):
  - GIỮ reranker (v11 bỏ → precision sụp)
  - judge_topk nhỏ hơn (chỉ judge sau adaptive, không judge 10 điều raw)

Chạy:
  QDRANT_COLLECTION=vbpl_aiteam EMBED_BACKEND=st EMBED_ST_MODEL=AITeamVN/Vietnamese_Embedding_v2
  HYBRID_SEARCH=true USE_HNSW=true USE_RERANKER=true BM25_INDEX_PATH=data/bm25_vbpl_v2.pkl
  python -m tests.build_submission_v12 --out data/submission_v12.json
"""
from __future__ import annotations

import argparse, json, logging, os, re, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("USE_TF", "0")

ROOT = Path(__file__).resolve().parents[1]
JUDGE_CACHE = ROOT / "data" / "judge_cache_v12.json"

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
logger = logging.getLogger("sub_v12")
logger.setLevel(logging.INFO)

# ── Collapse + adaptive (giữ nguyên từ v9) ────────────────────────────────────
_YEAR    = re.compile(r"/(19|20)(\d{2})/")
_PRIMARY = ("luật", "bộ luật", "pháp lệnh")
_TRAILING_SO = re.compile(r"\s+số:?\s*$")


def _year_of(sk: str) -> int:
    m = _YEAR.search(sk or "")
    return int(m.group(1) + m.group(2)) if m else 0


def _is_primary(name: str) -> bool:
    return any((name or "").lower().strip().startswith(p) for p in _PRIMARY)


def _family(name: str) -> str:
    s = (name or "").lower()
    s = re.sub(r"sửa đổi,? bổ sung.*", "", s)
    s = re.sub(r"số\s.*", "", s)
    s = re.sub(r"\d{4}", "", s)
    return re.sub(r"\s+", " ", s).strip(" .,")


def _law_name(p: dict) -> str:
    loai  = (p.get("loai_van_ban") or "").strip()
    title = _TRAILING_SO.sub("", (p.get("title") or "").strip())
    if loai and title.lower().startswith(loai.lower()):
        return title
    return f"{loai} {title}".strip() if loai else title


def collapse(cands: list[dict]) -> list[dict]:
    newest: dict[str, int] = {}
    for c in cands:
        p = c["art"].split("|"); sk, name = p[0], (p[1] if len(p) >= 2 else "")
        if _is_primary(name):
            fam = _family(name)
            newest[fam] = max(newest.get(fam, 0), _year_of(sk))
    return [c for c in cands
            if not (_is_primary(c["art"].split("|")[1] if len(c["art"].split("|")) >= 2 else "")
                    and _year_of(c["art"].split("|")[0]) < newest.get(
                        _family(c["art"].split("|")[1] if len(c["art"].split("|")) >= 2 else ""), 0))]


def adaptive(cands: list[dict], t_abs=0.40, ratio=0.80, min_k=1, max_k=4) -> list[dict]:
    if not cands:
        return []
    cands = sorted(cands, key=lambda c: c["rr"], reverse=True)
    cut = max(t_abs, ratio * cands[0]["rr"])
    chosen = [c for c in cands if c["rr"] >= cut][:max_k]
    return chosen or cands[:min_k]


# ── Candidate builder ─────────────────────────────────────────────────────────
def cand_from_hit(h: dict) -> dict | None:
    p  = h.get("payload", {})
    sk = p.get("so_ky_hieu")
    ds = p.get("dieu_so")
    if not sk or ds is None or int(ds) <= 0:
        return None
    name = _law_name(p)
    rr   = float(h.get("rerank_score") or h.get("adj_score") or h.get("score") or 0.0)
    return {
        "art":  f"{sk}|{name}|Điều {ds}",
        "doc":  f"{sk}|{name}",
        "rr":   rr,
        "_key": f"{sk}#{ds}",
    }


def merge_cands(lists: list[list[dict]]) -> list[dict]:
    pool: dict[str, dict] = {}
    for lst in lists:
        for c in lst:
            if c is None:
                continue
            k = c["_key"]
            if k not in pool or c["rr"] > pool[k]["rr"]:
                pool[k] = c
    return sorted(pool.values(), key=lambda c: c["rr"], reverse=True)


def hits_to_cands(hits: list[dict]) -> list[dict]:
    return [c for h in hits if (c := cand_from_hit(h)) is not None]


# ── Text lookup (dùng chung Qdrant client của RAGPipeline) ────────────────────
def load_text_lookup(collection: str, client) -> dict[str, str]:
    print(f"Scroll {collection} để lấy text...", flush=True)
    total = client.count(collection, exact=True).count
    lookup: dict[str, str] = {}
    offset, done = None, 0
    while True:
        pts, offset = client.scroll(collection, limit=2000, offset=offset,
                                    with_payload=True, with_vectors=False)
        for p in pts:
            pl  = p.payload or {}
            sk  = str(pl.get("so_ky_hieu") or "")
            ds  = pl.get("dieu_so")
            if not sk or ds is None:
                continue
            key = f"{sk}|{_law_name(pl)}|Điều {ds}"
            lookup[key] = str(pl.get("text") or "")
        done += len(pts)
        if done % 40000 < 2000:
            print(f"  {done:,}/{total:,}", flush=True)
        if offset is None:
            break
    print(f"Đã nạp {len(lookup):,} điều.", flush=True)
    return lookup


# ── LLM-judge (CoT, tích hợp inline) ─────────────────────────────────────────
_J_SYSTEM = """
Bạn là bộ lọc căn cứ pháp lý cho hệ thống truy hồi văn bản pháp luật Việt Nam.
Nhiệm vụ của bạn KHÔNG phải là trả lời câu hỏi, mà là quyết định từng điều luật có nên GIỮ làm căn cứ hay không.

Nguyên tắc GIỮ = trả lời "CÓ":
1. Điều luật trực tiếp quy định quy tắc, điều kiện, thủ tục, hồ sơ, thời hạn, mức phạt, biện pháp khắc phục, quyền, nghĩa vụ hoặc trách nhiệm được hỏi.
2. Điều luật chỉ trả lời một phần của câu hỏi nhiều vế nhưng phần đó là vế được hỏi rõ ràng.
3. Điều luật là "luật khung": định nghĩa, phạm vi áp dụng, quyền/nghĩa vụ, hành vi bị cấm, điều kiện chung; cần kết hợp với điều khác để trả lời đúng.
4. Điều luật thuộc nghị định/thông tư/quyết định hướng dẫn chi tiết luật khi câu hỏi hỏi hồ sơ, thủ tục, biểu mẫu, mức, thời hạn, biện pháp hoặc cách thực hiện.
5. Điều luật sửa đổi/bổ sung được GIỮ nếu nội dung trong điều trực tiếp thay đổi quy định liên quan.

Nguyên tắc LOẠI = trả lời "KHÔNG":
1. Chỉ cùng lĩnh vực rộng nhưng không xử lý đúng vấn đề trong câu hỏi.
2. Sai phạm vi đặc thù so với câu hỏi, ví dụ: công an, quân đội/quốc phòng, hải quan/xuất nhập khẩu, ngân hàng/NHNN, chứng khoán, điện lực, karaoke, địa phương cụ thể, cơ quan/tổ chức chuyên ngành khác.
3. Sai đối tượng, sai thủ tục, sai loại hành vi, sai quan hệ pháp luật hoặc chỉ là quy định quản lý nội bộ không được hỏi.
4. Phiên bản cũ/bị thay thế khi có ứng viên mới hơn tương đương và câu hỏi không hỏi giai đoạn lịch sử.
5. Điều về xử phạt/khắc phục hậu quả khi câu hỏi chỉ hỏi quyền, điều kiện hoặc thủ tục bình thường, trừ khi câu hỏi có dấu hiệu vi phạm, xử phạt, trách nhiệm, hậu quả, bồi thường, thu hồi.

Lưu ý quan trọng:
- Đừng loại luật khung chỉ vì đã có điều xử phạt hoặc điều hướng dẫn chi tiết.
- Đừng loại nghị định/thông tư chỉ vì đã có luật, nếu văn bản dưới luật chứa chi tiết cần cho câu hỏi.
- Với câu hỏi nhiều vế, một điều chỉ cần khớp chắc một vế cũng có thể là "CÓ".
- Nếu không đủ chắc, ưu tiên "CÓ" cho ứng viên có nội dung thật sự gần câu hỏi; chỉ "KHÔNG" khi thấy sai phạm vi/sai đối tượng rõ.
- Trả lời đúng định dạng, không thêm đoạn văn ngoài danh sách.
""".strip()
_J_USER = """Câu hỏi: {question}

Đọc từng ứng viên bên dưới và phân loại GIỮ/LOẠI.

Cách đối chiếu:
1. Xác định câu hỏi đang hỏi về đối tượng nào, hành vi/thủ tục/quyền-nghĩa vụ nào, và có hỏi xử phạt/hậu quả hay không.
2. So khớp điều luật với đúng đối tượng, đúng phạm vi, đúng loại vấn đề.
3. Nếu là điều nền tảng cần để hiểu/áp dụng điều khác thì vẫn chọn CÓ.
4. Nếu chỉ trùng vài từ khóa nhưng lệch phạm vi hoặc lệch thủ tục thì chọn KHÔNG.

Ví dụ cách chấm:
- Hỏi "mức phạt hành vi X": điều định nghĩa/hành vi bị cấm về X = CÓ; điều mức phạt X = CÓ; điều về ngành đặc thù khác = KHÔNG.
- Hỏi "hồ sơ/thủ tục đăng ký": điều nghị định/thông tư nêu thành phần hồ sơ/trình tự = CÓ; điều chỉ nói nguyên tắc chung không giúp xác định thủ tục = KHÔNG, trừ khi câu hỏi cũng hỏi điều kiện/quyền chung.
- Hỏi nhiều vế "có được làm A không và bị xử lý thế nào": điều về quyền/điều kiện A = CÓ; điều xử lý vi phạm A = CÓ.

Ứng viên:
{block}

Trả lời ĐÚNG {n} dòng, mỗi dòng đúng mẫu:
1: CÓ — [lý do rất ngắn: trực tiếp / luật khung / hướng dẫn chi tiết / trả lời một vế]
2: KHÔNG — [lý do rất ngắn: sai phạm vi / sai đối tượng / sai thủ tục / chỉ liên quan rộng / bản cũ]
..."""

_VERDICT_RE = re.compile(r"^\s*(\d+)\s*[:\.]\s*(CÓ|KHÔNG|CO|KHONG)", re.IGNORECASE)


def _parse_verdicts(text: str, n: int) -> list[bool]:
    v: dict[int, bool] = {}
    for line in (text or "").splitlines():
        m = _VERDICT_RE.match(line)
        if m:
            v[int(m.group(1))] = m.group(2).upper() in ("CÓ", "CO")
    return [v.get(i + 1, True) for i in range(n)]


def judge(question: str, cands: list[dict], texts: dict[str, str], llm) -> list[bool]:
    if len(cands) <= 1:
        return [True] * len(cands)
    parts = []
    for i, c in enumerate(cands, 1):
        txt = (texts.get(c["art"]) or "")[:1600]
        parts.append(f"Điều {i} — {c['art']}\n{txt}")
    block = "\n\n".join(parts)
    user  = _J_USER.format(question=question, block=block, n=len(cands))
    try:
        out = llm.complete(_J_SYSTEM, user, think=True)
        return _parse_verdicts(out, len(cands))
    except Exception as e:
        logger.warning("LLM lỗi: %s", e)
        return [True] * len(cands)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",       default="data/submission_v12.json")
    ap.add_argument("--orig-topk", type=int, default=8,  help="top-k câu gốc (reranked)")
    ap.add_argument("--sub-topk",  type=int, default=8,  help="top-k mỗi sub-query (reranked)")
    ap.add_argument("--max-k",     type=int, default=4,  help="adaptive max_k")
    ap.add_argument("--limit",     type=int, default=0)
    ap.add_argument("--no-judge",  action="store_true", help="Bỏ LLM-judge (nhanh, đo recall)")
    ap.add_argument("--subq",      default="data/subqueries.json", help="File phân rã sub-query")
    args = ap.parse_args()

    qs = json.loads(
        Path("C:/Users/PHONG/Downloads/R2AIStage1DATA.json").read_text(encoding="utf-8"))
    if args.limit:
        qs = qs[:args.limit]

    subq = json.loads((ROOT / args.subq).read_text(encoding="utf-8"))
    v8   = {r["id"]: r for r in json.loads(
        (ROOT / "data/submission_v8.json").read_text(encoding="utf-8"))}

    cache: dict[str, list[bool]] = (
        json.loads(JUDGE_CACHE.read_text(encoding="utf-8"))
        if JUDGE_CACHE.exists() else {}
    )

    from backend.rag import RAGPipeline

    rag  = RAGPipeline()
    llm = texts = None
    if not args.no_judge:
        from backend.llm import LLMClient
        llm  = LLMClient()
        texts = load_text_lookup(rag.settings.qdrant_collection, client=rag.store.client)

    logger.info("orig_topk=%d sub_topk=%d max_k=%d | subq=%d",
                args.orig_topk, args.sub_topk, args.max_k, len(subq))

    out_path = ROOT / args.out
    out: list[dict] = []
    n_judged = n_filtered = n_fallback = 0
    t0 = time.time()

    for i, q in enumerate(qs, 1):
        qid      = q["id"]
        question = q["question"]
        subs     = subq.get(str(qid)) or []

        # ── 1. Retrieve + rerank câu gốc ─────────────────────────────────────
        orig_hits  = rag.retrieve(question, top_k=args.orig_topk)
        orig_cands = hits_to_cands(orig_hits)

        # ── 2. Multi-query sub (chỉ khi >= 2 câu con) ────────────────────────
        sub_cands_list: list[list[dict]] = []
        if len(subs) >= 2:
            for s in subs:
                h = rag.retrieve(s, top_k=args.sub_topk)
                sub_cands_list.append(hits_to_cands(h))

        # ── 3. Merge max-rr + collapse bản cũ ────────────────────────────────
        all_lists = [orig_cands] + sub_cands_list
        merged    = merge_cands(all_lists)
        clean     = collapse(merged)

        # ── 4. Adaptive select ────────────────────────────────────────────────
        max_k  = min(len(subs) + 1, args.max_k) if len(subs) >= 2 else args.max_k
        chosen = adaptive(clean, t_abs=0.40, ratio=0.80, min_k=1, max_k=max_k)

        if not chosen:
            out.append(v8[qid])
            continue

        # ── 5. LLM-judge (chỉ khi >= 2 điều, resumable cache) ────────────────
        ckey = str(qid)
        if args.no_judge:
            kept = chosen
        elif len(chosen) >= 2:
            if ckey not in cache:
                verdicts = judge(question, chosen, texts, llm)
                cache[ckey] = verdicts
                n_judged += 1
                if n_judged % 20 == 0:
                    JUDGE_CACHE.write_text(
                        json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            else:
                verdicts = cache[ckey]
                if len(verdicts) != len(chosen):
                    verdicts = [True] * len(chosen)

            kept = [c for c, v in zip(chosen, verdicts) if v]
            if not kept:
                kept = chosen[:1]
                n_fallback += 1
            elif len(kept) < len(chosen):
                n_filtered += 1
        else:
            kept = chosen     # chỉ 1 điều → bỏ qua judge

        # ── 6. Build output ───────────────────────────────────────────────────
        arts: list[str] = [c["art"] for c in kept]
        docs: list[str] = []
        seen: set = set()
        for c in kept:
            if c["doc"] not in seen:
                seen.add(c["doc"]); docs.append(c["doc"])

        answer = v8.get(qid, {}).get("answer", "")
        out.append({"id": qid, "question": question,
                    "answer": answer,
                    "relevant_docs": docs, "relevant_articles": arts})

        # Dọn GPU cache định kỳ — chống fragment/degradation sau nhiều lần rerank
        # (giữ reranker luôn trên GPU, không rớt xuống CPU → tránh "treo giả").
        if i % 20 == 0:
            import torch, gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        if i % 50 == 0:
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            dt = time.time() - t0
            logger.info("[%d/%d] %.2f q/s | %.1fs/cau gan day | GPU %s",
                        i, len(qs), i / dt, dt / max(i, 1),
                        f"{__import__('torch').cuda.memory_allocated()/1e9:.1f}GB"
                        if __import__('torch').cuda.is_available() else "cpu")

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    JUDGE_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    logger.info("XONG %d câu | judged=%d filtered=%d fallback=%d (%.0fs) → %s",
                len(out), n_judged, n_filtered, n_fallback, time.time() - t0, out_path)
    print(f"Hoàn thành: {len(out)} câu → {out_path}", flush=True)


if __name__ == "__main__":
    main()
