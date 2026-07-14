import json
import pickle
import re
import unicodedata
from pathlib import Path

from rank_bm25 import BM25Okapi


ARTICLES_PATH = Path("articles_phapdien64_web_canonical.jsonl")
BM25_OUT_PATH = Path("bm25_dual_with_docs_phapdien64_web_canonical.pkl")


def strip_accents(text):
    text = str(text or "")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn").replace("đ", "d").replace("Đ", "D")


def bm25_tokenize(text, remove_diacritics=False):
    text = str(text or "").lower()
    if remove_diacritics:
        text = strip_accents(text)
    return re.findall(r"\w+", text, flags=re.UNICODE)


def load_docs(path):
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            docs.append(json.loads(line))
            if i % 25000 == 0:
                print(f"Loaded rows: {i}")
    return docs


def main():
    print(f"Loading docs from: {ARTICLES_PATH}")
    docs = load_docs(ARTICLES_PATH)
    print(f"Total docs: {len(docs)}")

    print("Tokenizing normal corpus...")
    bm25_corpus_normal = [
        bm25_tokenize(doc.get("bm25_search", ""), remove_diacritics=False)
        for doc in docs
    ]
    print("Tokenizing no-diac corpus...")
    bm25_corpus_no_diac = [
        bm25_tokenize(doc.get("bm25_search", ""), remove_diacritics=True)
        for doc in docs
    ]

    print("Building BM25 normal...")
    bm25_normal = BM25Okapi(bm25_corpus_normal)
    print("Building BM25 no-diac...")
    bm25_no_diac = BM25Okapi(bm25_corpus_no_diac)

    doc_ids = [doc.get("doc_id") or doc.get("id") for doc in docs]
    bundle = {
        "bm25_normal": bm25_normal,
        "bm25_no_diac": bm25_no_diac,
        "docs": docs,
        "doc_ids": doc_ids,
    }
    with BM25_OUT_PATH.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved BM25 bundle to: {BM25_OUT_PATH}")
    print(f"Total docs: {len(docs)}")


if __name__ == "__main__":
    main()
