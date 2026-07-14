"""Rebuild corpus SME từ API CHÍNH THỐNG vbpl (thay HF dataset) — coverage + hiệu lực + parse sạch.

3 cái lợi so với corpus HF hiện tại (93K điều, regex-parse, heuristic hiệu lực):
  1. PHỤC HỒI 7611 doc null-markdown bị HF bỏ → recall ceiling (đòn chính, top-1 R0.72 vs ta 0.55).
  2. effStatus CHÍNH THỨC → chỉ giữ "Còn hiệu lực" (bỏ doc hết hiệu lực như 02/2016/TT-BCT) → precision.
  3. prov-article/clause/item → parse Điều chuẩn (khỏi regex đoán).

Chạy trên KAGGLE (mạng nhanh; fetch ~12K doc). KHÔNG chạy local (máy user lag + mạng yếu).

Quy trình:
    python rebuild_corpus_vbpl.py ids       # stream HF keywords → data/sme_doc_ids_all.json {docNum: vbpl_id}
    python rebuild_corpus_vbpl.py rebuild    # fetch tất cả id (đa luồng) → data/corpus_articles.jsonl (mới)
→ rồi embed_corpus.py để sinh corpus_emb.npy mới (corpus đã đổi → PHẢI re-embed).
"""
import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
import local_models_config as cfg
from legal_text_parser import parse_legal_name
from vbpl_fetch import fetch_doc, parse_articles, extract_id, is_in_force
import build_corpus as bc   # tái dùng _keep / _pick_doc_number (DRY)

IDS_ALL = os.path.join(cfg.DATA_DIR, "sme_doc_ids_all.json")


def collect_ids():
    """Stream HF keywords → {docNum: vbpl_id} cho MỌI doc khớp SME (kể cả null-markdown)."""
    from datasets import load_dataset
    ds = load_dataset(cfg.HF_VBPL, "documents", split="train", streaming=True)
    try:
        ds = ds.select_columns(["doc_number", "title", "legal_type", "source_url", "summary"])
    except Exception as e:
        print("(select_columns skip:", e, ")")
    ids, n, miss = {}, 0, 0
    for row in ds:
        n += 1
        if n % 20000 == 0:
            print(f"  scanned {n} | matched {len(ids)} | no-id {miss}")
        if not bc._keep(row, "keywords"):
            continue
        code = bc._pick_doc_number(row.get("doc_number"))
        if not code or code in ids:
            continue
        vid = extract_id(row.get("source_url", ""))
        if vid:
            ids[code] = vid
        else:
            miss += 1
    json.dump(ids, open(IDS_ALL, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\nMatched SME docs: {len(ids)} có id (+{miss} thiếu id) / {n} scanned → {IDS_ALL}")


def rebuild(ids_path, workers=8, out_path=None, flush_every=200):
    """Fetch đa luồng → GHI DẦN ra đĩa (RAM thấp, an toàn 16GB) + checkpoint + resume.

    Resume: doc đã có trong out_path → bỏ qua (chạy lại an toàn nếu mạng hụt).
    RAM chỉ giữ set id (vài chục MB), row ghi thẳng file → không phình bộ nhớ.
    """
    out_path = out_path or cfg.CORPUS_JSONL
    done_path = out_path + ".done"                     # sidecar: MỌI docNum đã xử lý (kể cả expired/noart/fail)
    ids = json.load(open(ids_path, encoding="utf-8"))

    done_docs, seen_ids, n_rows = set(), set(), 0
    if os.path.exists(out_path):                       # RESUME: điều đã GHI (doc còn hiệu lực có điều)
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                done_docs.add(r["doc_number"]); seen_ids.add(r["id"]); n_rows += 1
    if os.path.exists(done_path):                      # + doc đã xử lý nhưng KHÔNG ghi (expired/noart/fail)
        with open(done_path, encoding="utf-8") as f:
            done_docs.update(ln.strip() for ln in f if ln.strip())
    if done_docs:
        print(f"resume: {len(done_docs)} doc đã xử lý ({n_rows} điều) → bỏ qua, không fetch lại")

    items = [(dn, vid) for dn, vid in ids.items() if dn not in done_docs]
    print(f"fetch {len(items)}/{len(ids)} doc còn lại ({workers} luồng, ghi dần)...")
    stat = {"ok": 0, "expired": 0, "fail": 0, "noart": 0, "proc": 0}
    lock = threading.Lock()
    fout = open(out_path, "a", encoding="utf-8")       # append → giữ phần đã resume
    dfout = open(done_path, "a", encoding="utf-8")     # sidecar đồng hành (log mọi doc đã xử lý)

    def work(pair):
        docNum, vid = pair
        d = fetch_doc(vid)
        if not d:
            with lock: stat["fail"] += 1
        elif not is_in_force(d.get("effStatus")):      # GROUND-TRUTH hiệu lực
            with lock: stat["expired"] += 1
        else:
            arts = parse_articles(d.get("content", ""))
            if not arts:
                with lock: stat["noart"] += 1
            else:
                meta = parse_legal_name(d.get("title", ""), docNum, d.get("docType", "") or "")
                m = re.search(r"(\d{4})", d.get("issueDate", "") or "")
                yr, slug = (m.group(1) if m else ""), re.sub(r"[^A-Za-z0-9]", "", docNum)
                with lock:
                    for a in arts:
                        aid = f"{slug}_{a['article'].replace(' ', '')}"
                        while aid in seen_ids:
                            aid += "x"
                        seen_ids.add(aid)
                        fout.write(json.dumps({
                            "id": aid, "doc_number": docNum, "clean_name": meta["clean_name"],
                            "legal_type": meta["type"], "year": yr, "article": a["article"],
                            "title": a["title"], "text": a["text"][:4000],
                            "source_url": f"https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/{vid}",
                        }, ensure_ascii=False) + "\n")
                    stat["ok"] += 1
        with lock:
            stat["proc"] += 1; p = stat["proc"]
            dfout.write(docNum + "\n")                  # log MỌI doc đã xử lý → lần sau resume bỏ qua
        if p % flush_every == 0:
            fout.flush(); dfout.flush()
            print(f"  {p}/{len(items)} | ok {stat['ok']} | hết hiệu lực {stat['expired']} | "
                  f"noart {stat['noart']} | fail {stat['fail']}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, items))
    fout.flush(); fout.close(); dfout.flush(); dfout.close()

    total_arts, inforce = 0, set()                      # đếm lại từ file: điều + văn bản CÒN HIỆU LỰC duy nhất
    with open(out_path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                total_arts += 1; inforce.add(json.loads(ln)["doc_number"])
    print(f"\n✓ {out_path}: {total_arts} điều / {len(inforce)} văn bản CÒN HIỆU LỰC")
    print(f"  fetch: ok {stat['ok']} | hết hiệu lực (bỏ) {stat['expired']} | no-article {stat['noart']} | fail {stat['fail']}")
    print("  → upload corpus_articles.jsonl lên Kaggle → embed_corpus.py (GPU) sinh corpus_emb.npy MỚI.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ids", "rebuild"])
    ap.add_argument("--ids", default=IDS_ALL, help="file id cho mode rebuild")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="đường dẫn corpus ra (mặc định data/corpus_articles.jsonl)")
    a = ap.parse_args()
    if a.mode == "ids":
        collect_ids()
    else:
        rebuild(a.ids, a.workers, a.out)
