"""PASS 2: hợp nhất multi-query → submission_v6.

Với câu đã phân rã (subqueries.json): truy hồi câu GỐC + từng câu CON → gộp ứng viên
theo (số hiệu, điều), điểm = MAX rerank qua các truy vấn → gộp bản luật cũ (IMPROVE#1)
→ chọn thích ứng (max_k co giãn theo số vế). Câu KHÔNG phân rã: bê nguyên từ v5.

Chạy: HYBRID_SEARCH=true USE_RERANKER=true USE_HNSW=true USE_TF=0
  python -m tests.build_submission_v6 --ids-from-cache --out data/submission_v6.json
"""
from __future__ import annotations
import argparse, json, logging, re, time
from pathlib import Path

from backend.rag import RAGPipeline
from tests.build_submission import _law_name

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
logger = logging.getLogger("sub_v6"); logger.setLevel(logging.INFO)
ROOT = Path(__file__).resolve().parents[1]

# --- gộp bản luật cũ (IMPROVE#1) + chọn thích ứng — inline để khỏi import chéo ---
_YEAR = re.compile(r"/(19|20)(\d{2})/")
_PRIMARY = ("luật", "bộ luật", "pháp lệnh")

def _year_of(sk: str) -> int:
    m = _YEAR.search(sk or ""); return int(m.group(1)+m.group(2)) if m else 0

def _is_primary(name: str) -> bool:
    n = (name or "").lower().strip(); return any(n.startswith(p) for p in _PRIMARY)

def _family(name: str) -> str:
    s = (name or "").lower()
    s = re.sub(r"sửa đổi,? bổ sung.*", "", s)
    s = re.sub(r"số\s.*", "", s)
    s = re.sub(r"\d{4}", "", s)
    return re.sub(r"\s+", " ", s).strip(" .,")

def collapse(cands: list[dict]) -> tuple[list[dict], int]:
    newest: dict[str, int] = {}
    for c in cands:
        p = c["art"].split("|"); sk, name = p[0], (p[1] if len(p) >= 2 else "")
        if _is_primary(name):
            fam = _family(name); newest[fam] = max(newest.get(fam, 0), _year_of(sk))
    out, dropped = [], 0
    for c in cands:
        p = c["art"].split("|"); sk, name = p[0], (p[1] if len(p) >= 2 else "")
        if _is_primary(name) and _year_of(sk) < newest[_family(name)]:
            dropped += 1; continue
        out.append(c)
    return out, dropped

def adaptive(cands, t_abs=0.45, ratio=0.85, min_k=1, max_k=3):
    if not cands: return []
    cands = sorted(cands, key=lambda c: c["rr"], reverse=True)
    cut = max(t_abs, ratio * cands[0]["rr"])
    chosen = [c for c in cands if c["rr"] >= cut][:max_k]
    return chosen or cands[:min_k]


def cand_from_hit(h: dict) -> dict | None:
    p = h.get("payload", {})
    sk, ds = p.get("so_ky_hieu"), p.get("dieu_so")
    if not sk or ds is None or int(ds) <= 0:
        return None
    name = _law_name(p)
    return {"art": f"{sk}|{name}|Điều {ds}", "doc": f"{sk}|{name}",
            "rr": float(h.get("rerank_score", 0.0)), "nam": p.get("nam"),
            "loai": p.get("loai_van_ban"), "_key": f"{sk}#{ds}"}


def key_of(c: dict) -> str:
    if "_key" in c:
        return c["_key"]
    sk = c["art"].split("|")[0]
    ds = c["art"].rsplit("Điều ", 1)[-1].strip()
    return f"{sk}#{ds}"


def merge(orig: list[dict], sub_hits: list[list[dict]]) -> list[dict]:
    """Gộp ứng viên gốc + các câu con; điểm rr = MAX qua mọi truy vấn."""
    pool: dict[str, dict] = {}
    def add(c):
        k = key_of(c)
        if k not in pool or c["rr"] > pool[k]["rr"]:
            pool[k] = {**c, "_key": k}
    for c in orig:
        add(c)
    for hits in sub_hits:
        for h in hits:
            c = cand_from_hit(h)
            if c:
                add(c)
    return sorted(pool.values(), key=lambda c: c["rr"], reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/submission_v9.json")
    ap.add_argument("--sub-topk", type=int, default=8)
    args = ap.parse_args()

    qs = json.loads(Path("C:/Users/PHONG/Downloads/R2AIStage1DATA.json").read_text(encoding="utf-8"))
    v8 = {r["id"]: r for r in json.loads((ROOT/"data/submission_v8.json").read_text(encoding="utf-8"))}
    scored = {r["id"]: r for r in json.loads((ROOT/"data/submission_v8_scored.json").read_text(encoding="utf-8"))}
    subq = json.loads((ROOT/"data/subqueries.json").read_text(encoding="utf-8"))

    rag = RAGPipeline()
    logger.info("hnsw=%s | %d câu phân rã / %d tổng", rag.settings.use_hnsw, len(subq), len(qs))

    out, t0, n_done = [], time.time(), 0
    out_path = ROOT / args.out
    for i, q in enumerate(qs, 1):
        qid = q["id"]
        subs = subq.get(str(qid))
        if not subs or len(subs) == 1:
            out.append(v8[qid])                  # không phân rã → giữ v8
            continue
        orig = scored.get(qid, {}).get("candidates", [])
        sub_hits = [rag.retrieve(s, top_k=args.sub_topk) for s in subs]
        merged = merge(orig, sub_hits)
        clean, _ = collapse(merged)
        max_k = min(len(subs) + 1, 4)
        chosen = adaptive(clean, t_abs=0.40, ratio=0.80, min_k=1, max_k=max_k)
        arts, docs, seen = [], [], set()
        for c in chosen:
            arts.append(c["art"])
            if c["doc"] not in seen:
                seen.add(c["doc"]); docs.append(c["doc"])
        out.append({"id": qid, "question": q["question"], "answer": v8[qid]["answer"],
                    "relevant_docs": docs, "relevant_articles": arts})
        n_done += 1
        if n_done % 20 == 0:
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("[%d phân rã | %d/%d tổng] %.2f câu-phânrã/s", n_done, i, len(qs),
                        n_done / (time.time() - t0))
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("XONG %d câu (%d phân rã, %.0fs) → %s", len(out), n_done, time.time()-t0, out_path)


if __name__ == "__main__":
    main()
