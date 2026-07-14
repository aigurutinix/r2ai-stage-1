"""Build the article-level legal corpus from organizer-recommended HF datasets.

Streams `tmquan/vbpl-vn` (158k docs), keeps SME-relevant laws, parses each
document's markdown into Điều-level chunks, writes `data/corpus_articles.jsonl`.

Usage:
    python build_corpus.py                 # allowlist mode (fast, precise)
    python build_corpus.py --mode keywords # broader sweep (more recall, slower)
    python build_corpus.py --limit 5000    # cap scanned docs (debug)

Output JSONL row:
    {id, doc_number, clean_name, legal_type, year, article, title, text, source_url}
"""
import argparse
import json
import os
import re
import sys

import local_models_config as cfg
from legal_text_parser import parse_legal_name, split_into_articles


def _norm_code(c: str) -> str:
    return re.sub(r"\s+", "", (c or "")).upper()


_ALLOW_NORM = {_norm_code(c) for c in cfg.SME_LAW_ALLOWLIST}


def _pick_doc_number(raw) -> str:
    """vbpl-vn doc_number is a list; return first non-empty element."""
    if isinstance(raw, list):
        for x in raw:
            if x and str(x).strip():
                return str(x).strip()
        return ""
    return str(raw).strip() if raw else ""


def _keep(row, mode: str) -> bool:
    legal_type = (row.get("legal_type") or "").lower()
    if not any(t in legal_type for t in cfg.KEEP_LEGAL_TYPES):
        # allowlist match can still rescue a doc with odd legal_type
        if mode != "allowlist":
            return False
    if mode == "allowlist":
        code = _norm_code(_pick_doc_number(row.get("doc_number")))
        return code in _ALLOW_NORM
    # keywords mode — search descriptive fields (doc_name is a stable ID, not text)
    blob = f"{row.get('title') or ''} {row.get('summary') or ''}".lower()
    return any(k in blob for k in cfg.SME_TITLE_KEYWORDS)


def build(mode: str, limit: int, append: bool = False):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Missing 'datasets'. Run: pip install -r backend/requirements-local.txt")

    print(f"Streaming {cfg.HF_VBPL} (config=documents) ... mode={mode}")
    ds = load_dataset(cfg.HF_VBPL, "documents", split="train", streaming=True)
    # Only pull the columns we use → skips heavy structure_json/extracted_json/file_paths_json,
    # dramatically reducing download while streaming 158k docs.
    try:
        ds = ds.select_columns(
            ["doc_number", "title", "legal_type", "markdown", "source_url", "year", "doc_name", "summary"]
        )
    except Exception as e:
        print(f"(select_columns skipped: {e})")

    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    seen_ids = set()
    existing_codes = set()     # doc_numbers already in corpus → skip in append mode
    found_codes = set()
    dropped_no_articles = []   # (code) kept-by-filter but markdown null / no Điều parsed
    n_docs = n_kept = n_articles = 0

    if append and os.path.exists(cfg.CORPUS_JSONL):
        with open(cfg.CORPUS_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                seen_ids.add(r["id"])
                existing_codes.add(_norm_code(r.get("doc_number", "")))
        print(f"APPEND: corpus đã có {len(seen_ids)} điều / {len(existing_codes)} văn bản → bỏ qua các văn bản này.")

    with open(cfg.CORPUS_JSONL, "a" if append else "w", encoding="utf-8") as out:
        for row in ds:
            n_docs += 1
            if limit and n_docs > limit:
                break
            if n_docs % 5000 == 0:
                print(f"  scanned {n_docs} docs | kept {n_kept} | articles {n_articles}")

            if not _keep(row, mode):
                continue

            code = _pick_doc_number(row.get("doc_number"))
            if not code:
                continue
            if _norm_code(code) in existing_codes:   # append: văn bản đã có → bỏ qua
                continue
            meta = parse_legal_name(row.get("title", ""), code, row.get("legal_type", ""))
            articles = split_into_articles(row.get("markdown") or "")
            if not articles:
                # kept by filter but no parsable articles (null markdown / odd format)
                dropped_no_articles.append(code)
                continue

            n_kept += 1
            found_codes.add(_norm_code(code))
            code_slug = re.sub(r"[^A-Za-z0-9]", "", code)
            for art in articles:
                art_id = f"{code_slug}_{art['article'].replace(' ', '')}"
                base, k = art_id, 1
                while art_id in seen_ids:
                    art_id = f"{base}_{k}"
                    k += 1
                seen_ids.add(art_id)
                out.write(json.dumps({
                    "id": art_id,
                    "doc_number": code,
                    "clean_name": meta["clean_name"],
                    "legal_type": meta["type"],
                    "year": row.get("year", ""),
                    "article": art["article"],
                    "title": art["title"],
                    "text": art["text"][:4000],
                    "source_url": row.get("source_url", ""),
                }, ensure_ascii=False) + "\n")
                n_articles += 1

    print("\n" + "=" * 60)
    print(f"Corpus written: {cfg.CORPUS_JSONL}")
    print(f"Docs scanned={n_docs} kept={n_kept} articles={n_articles}")
    if dropped_no_articles:
        print(f"⚠ {len(dropped_no_articles)} matched docs had NO parsable articles "
              f"(null markdown / odd format): {sorted(set(dropped_no_articles))[:20]}")
    if mode == "allowlist":
        missing = _ALLOW_NORM - found_codes - existing_codes   # existing đã có (append) không tính thiếu
        print(f"Allowlist found {len(found_codes)} mới + {len(existing_codes)} sẵn có / {len(_ALLOW_NORM)}")
        if missing:
            print(f"⚠ MISSING from dataset (recall capped for these): {sorted(missing)}")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["allowlist", "keywords"], default="allowlist")
    ap.add_argument("--limit", type=int, default=0, help="cap scanned docs (0=all)")
    ap.add_argument("--append", action="store_true",
                    help="thêm luật mới vào corpus hiện có (bỏ qua văn bản đã có), không ghi đè")
    args = ap.parse_args()
    build(args.mode, args.limit, args.append)
