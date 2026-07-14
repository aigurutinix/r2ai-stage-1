"""Đo TRẦN COVERAGE recall: gold (mã VB, số Điều) có trong corpus không.

Recall_max = gold_article_có_trong_corpus / tổng_gold. Rerank/fine-tune KHÔNG vượt được trần này.
So corpus MỚI (vbpl) vs CŨ (hf93k) trên cùng gold → trả lời "rebuild có mất điều luật không".
Tách universe-gap (doc ngoài 8020 ids) vs parse-loss (doc trong universe nhưng noart/expired).

Run: python scratch/audit_coverage.py
"""
import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = re.compile(r"(\d+)")


def norm_doc(s):
    return re.sub(r"\s+", "", (s or "").strip())          # "59/2019/NĐ-CP" chuẩn hóa khoảng trắng


def art_int(s):
    m = ART.search(s or "")
    return m.group(1) if m else None                       # "Điều 24" -> "24"


def corpus_keys(path):
    """{(mã chuẩn, số điều)} từ một corpus jsonl."""
    keys, docs = set(), set()
    if not os.path.exists(path):
        return keys, docs
    for ln in open(path, encoding="utf-8"):
        if not ln.strip():
            continue
        r = json.loads(ln)
        dn = norm_doc(r.get("doc_number"))
        ai = art_int(r.get("article"))
        if dn and ai:
            keys.add((dn, ai)); docs.add(dn)
    return keys, docs


def parse_gold(item):
    """['59/2019/NĐ-CP|Điều 24', ...] -> [(mã chuẩn, số điều)]."""
    out = []
    for g in item.get("gold", []):
        if "|" not in g:
            continue
        mvb, dieu = g.split("|", 1)
        dn, ai = norm_doc(mvb), art_int(dieu)
        if dn and ai:
            out.append((dn, ai))
    return out


def audit(gold_items, keys, docs, universe):
    """Trả về dict thống kê coverage trên một gold set + một corpus."""
    g_tot = g_hit = 0                                       # article-level
    q_any = q_all = q_with_gold = 0                         # question-level
    gold_docs, doc_in_corpus, doc_in_univ = set(), 0, 0
    for it in gold_items:
        gold = parse_gold(it)
        if not gold:
            continue
        q_with_gold += 1
        hits = [k in keys for k in gold]
        g_tot += len(gold); g_hit += sum(hits)
        if any(hits): q_any += 1
        if all(hits): q_all += 1
        for dn, _ in gold:
            gold_docs.add(dn)
    for dn in gold_docs:
        if dn in docs: doc_in_corpus += 1
        if dn in universe: doc_in_univ += 1
    return {
        "q": q_with_gold, "art_ceiling": g_hit / max(g_tot, 1),
        "cover_any": q_any / max(q_with_gold, 1), "cover_all": q_all / max(q_with_gold, 1),
        "gold_docs": len(gold_docs),
        "doc_in_corpus": doc_in_corpus / max(len(gold_docs), 1),
        "doc_in_univ": doc_in_univ / max(len(gold_docs), 1),
    }


def show(tag, s):
    print(f"\n[{tag}]  ({s['q']} câu có gold, {s['gold_docs']} văn bản gold duy nhất)")
    print(f"  TRẦN recall (article)   : {s['art_ceiling']:.3f}   <-- rerank/fine-tune KHÔNG vượt số này")
    print(f"  cover_any (≥1 điều đúng): {s['cover_any']:.3f}")
    print(f"  cover_all (đủ mọi điều) : {s['cover_all']:.3f}")
    print(f"  doc gold ∈ corpus       : {s['doc_in_corpus']:.3f}")
    print(f"  doc gold ∈ universe 8020: {s['doc_in_univ']:.3f}   (gap = universe-gap, ngoài tầm rebuild)")


def main():
    universe = set(norm_doc(d) for d in json.load(open(os.path.join(ROOT, "data/sme_doc_ids_all.json"))))
    new_keys, new_docs = corpus_keys(os.path.join(ROOT, "data/corpus_articles.jsonl"))
    old_keys, old_docs = corpus_keys(os.path.join(ROOT, "backup/corpus_articles_hf93k.jsonl"))
    print(f"corpus MỚI(vbpl): {len(new_keys)} (mã,điều) / {len(new_docs)} doc")
    print(f"corpus CŨ (hf)  : {len(old_keys)} (mã,điều) / {len(old_docs)} doc")
    print(f"universe ids    : {len(universe)} doc")

    zalo = json.load(open(os.path.join(ROOT, "data/zalo_eval.json")))
    print("\n===== ZALO eval (gold người-gán THẬT, 3196 câu — lưu ý lệch scope: hỏi cả luật ngoài SME) =====")
    show("MỚI vbpl", audit(zalo, new_keys, new_docs, universe))
    show("CŨ  hf93k", audit(zalo, old_keys, old_docs, universe))


if __name__ == "__main__":
    main()
