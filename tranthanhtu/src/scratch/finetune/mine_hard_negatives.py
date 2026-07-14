#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mine_hard_negatives.py — Đào hard negatives cho fine-tune reranker (arXiv 2412.00657).

Đọc synth_pairs.jsonl (do gen_synthetic_pairs.py sinh) + corpus_articles.jsonl + corpus_emb.npy.
Với mỗi câu hỏi tổng hợp: BGE-M3 (AITeamVN/Vietnamese_Embedding) lấy top-K=100, positive =
điều luật nguồn (match theo pos_id), hard negs lấy từ cửa sổ rank [10,60) loại trừ positive +
trùng (doc_number,article) + adjacency ±2. Xuất train_reranker.jsonl đúng schema FlagEmbedding
encoder-only reranker {query, pos:[str], neg:[str]} (KHÔNG pos_scores/neg_scores/prompt khi KD=False).

Split held-out 90/10 theo pos_doc_number (tránh leakage) → eval_pairs.jsonl cho eval_reranker.py.

DATA-FLOW CONTRACT (đồng bộ với gen_synthetic_pairs.py / train_reranker.py / eval_reranker.py):
  READS  synth_pairs.jsonl : {q, pos_id, pos_doc_number, pos_article, pos_text, aspect}
  READS  corpus_articles.jsonl : {id, doc_number, clean_name, legal_type, year, article, title, text, ...}
  READS  corpus_emb.npy : row i ↔ corpus jsonl line i (chỉ các dòng có doc_number — KHỚP retrieval cell 10)
  WRITES train_reranker.jsonl : {"query": str, "pos": [str], "neg": [str x N]}
  WRITES eval_pairs.jsonl     : {"query", "pos": [str], "neg": [str x N],
                                 "pos_id", "pos_doc_number", "pos_article"}  (held-out, cho eval base vs FT)
         ↑ eval_reranker.load_flagembedding_eval ĐỌC r["pos"]/r["neg"]; các field pos_id/doc_number/article
           giữ thêm cho debug (loader bỏ qua) — BẮT BUỘC có "pos" nếu không eval skip toàn bộ sample.

Resumable: append-only vào train/val, keyed theo (pos_id, q-hash) qua sidecar .done.
Determinism: random seed 42 cố định. Tái dùng corpus_emb.npy an toàn (mirror retrieval cell 10).
Chạy: Kaggle 2×T4 hoặc local. ~0.5h cho ~25-40K cặp.
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys

import numpy as np

SEED = 42
EMBED_ID = "AITeamVN/Vietnamese_Embedding"   # BGE-M3, KHỚP retrieval cell 10
MAX_SEQ = 1024                               # KHỚP MAX_SEQ retrieval cell 10
EMB_BATCH = 128
WORK = "/kaggle/working"

# ---- Hyperparams mining (theo spec.hard_neg_strategy / arXiv 2412.00657) ----
TOPK = 100              # K = top-100 BGE-M3
WIN_LO, WIN_HI = 10, 60  # cửa sổ rank [10,60): bỏ top-10 (false-neg paraphrase), cap 60 (giữ "hard")
ADJ_GUARD = 2           # loại ±2 rank quanh positive nếu positive rơi trong cửa sổ
NUM_NEG = 15            # N=15 hard negs / query (trainer sẽ sample 7 khi train_group_size=8)
TOPUP_HI = 100          # nếu thiếu sau loại trừ → bù từ [60,100) rồi fallback random corpus
VAL_FRAC = 0.10         # held-out ~10% theo pos_doc_number (tránh leakage)


def find_file(name, explicit=None):
    """Tìm file: đường dẫn chỉ định → /kaggle/working → cwd → /kaggle/input/**. (mirror notebook find_file)."""
    if explicit and os.path.exists(explicit):
        return explicit
    cands = [os.path.join(WORK, name), name] + glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Không thấy {name}. Truyền --flag hoặc đặt vào /kaggle/working hoặc Add Input.")


def load_corpus(path):
    """Đọc corpus — CHỈ giữ dòng có doc_number, đúng thứ tự như retrieval cell 10 (để row i ↔ emb i)."""
    corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("doc_number"):
                corpus.append(r)
    return corpus


def passage_text(row):
    """Chuỗi passage = f'{title}\\n{text}' — BYTE-IDENTICAL với inference (docs_text cell 10) → no train/serve skew."""
    return f"{row.get('title', '')}\n{row.get('text', '')}"


def load_corpus_emb(corpus, emb_arg):
    """Tái dùng corpus_emb.npy nếu CÓ và KHỚP độ dài corpus, nếu không thì encode + lưu.

    Mirror y hệt safe-reuse pattern của retrieval cell 10: row i ↔ corpus line i.
    """
    emb_path = emb_arg or os.path.join(WORK, "corpus_emb.npy")
    emb_in = emb_path if os.path.exists(emb_path) else (
        glob.glob("/kaggle/input/**/corpus_emb.npy", recursive=True) or [None])[0]
    corpus_emb = None
    if emb_in and os.path.exists(emb_in):
        _e = np.load(emb_in).astype("float32")
        if len(_e) == len(corpus):
            corpus_emb = _e
            print(f"Dùng corpus_emb.npy có sẵn ({emb_in}) khớp {len(corpus)} dòng → BỎ QUA embed corpus")
        else:
            print(f"⚠ corpus_emb.npy ({len(_e)}) ≠ corpus ({len(corpus)}) → BỎ emb cũ, embed lại")
    if corpus_emb is None:
        emb = _load_embedder()
        docs_text = [passage_text(r) for r in corpus]
        print(f"Embedding {len(docs_text)} điều luật (batch {EMB_BATCH}, max_seq {MAX_SEQ})…")
        corpus_emb = emb.encode(docs_text, batch_size=EMB_BATCH, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=True).astype("float32")
        save_to = emb_path if os.path.isdir(os.path.dirname(emb_path) or ".") else os.path.join(WORK, "corpus_emb.npy")
        try:
            np.save(save_to, corpus_emb)
            print(f"Đã lưu {save_to} ({len(corpus_emb)} vec)")
        except OSError as e:
            print(f"⚠ Không lưu được corpus_emb.npy ({e}) — tiếp tục với emb trong RAM.")
    return corpus_emb


_EMBEDDER = None


def _load_embedder():
    """Lazy-load BGE-M3 (chỉ 1 instance, dùng lại cho cả corpus + queries — tránh nạp 2.3GB lần hai)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        import torch
        from sentence_transformers import SentenceTransformer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Nạp embedder {EMBED_ID} trên {dev}…")
        _EMBEDDER = SentenceTransformer(EMBED_ID, device=dev)
        _EMBEDDER.max_seq_length = MAX_SEQ
    return _EMBEDDER


def read_pairs(path):
    """Đọc synth_pairs.jsonl. Bỏ qua dòng lỗi/thiếu field bắt buộc."""
    pairs, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if p.get("q") and p.get("pos_id"):
                pairs.append(p)
            else:
                bad += 1
    if bad:
        print(f"⚠ Bỏ qua {bad} dòng synth_pairs hỏng/thiếu field.")
    return pairs


def pair_key(p):
    """Khóa duy nhất cho 1 cặp (resumable): pos_id + hash câu hỏi."""
    h = hashlib.sha1(p["q"].encode("utf-8")).hexdigest()[:12]
    return f"{p['pos_id']}::{h}"


def load_done(done_path):
    if not os.path.exists(done_path):
        return set()
    with open(done_path, encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def count_lines(path):
    """Đếm số dòng đã ghi (cho resume) — đóng file handle tường minh (không leak)."""
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def split_train_val(pairs, val_frac):
    """Chia 90/10 THEO pos_doc_number để không leak (cùng văn bản không nằm cả hai bên)."""
    rng = random.Random(SEED)
    docs = sorted({p.get("pos_doc_number", "") for p in pairs})
    rng.shuffle(docs)
    n_val = max(1, int(len(docs) * val_frac)) if docs else 0
    val_docs = set(docs[:n_val])
    train = [p for p in pairs if p.get("pos_doc_number", "") not in val_docs]
    val = [p for p in pairs if p.get("pos_doc_number", "") in val_docs]
    return train, val


def mine_one(p, q_vec, corpus_emb, corpus, id2idx, rng, topk, num_neg):
    """Đào hard negs cho 1 cặp. Trả (neg_idxs:list[int], n_neg:int) hoặc None nếu drop."""
    pos_idx = id2idx.get(p["pos_id"])
    if pos_idx is None:
        return None  # pos_id không có trong corpus → bỏ (defensive)

    # 1. top-K theo cosine (corpus_emb đã normalize, q_vec normalize) → argpartition rồi sort
    scores = corpus_emb @ q_vec
    k = min(topk, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]  # giảm dần
    ranking = top.tolist()

    # 2. rank của positive trong top-K
    try:
        r_pos = ranking.index(pos_idx)
    except ValueError:
        return None  # defensive: positive KHÔNG trong top-100 → drop (gate kế thừa từ gen)

    # khóa loại trừ same-(doc_number, article) của positive
    pos_dn = (p.get("pos_doc_number") or corpus[pos_idx].get("doc_number") or "")
    pos_art = (p.get("pos_article") or corpus[pos_idx].get("article") or "")

    def excluded(rank, idx):
        if idx == pos_idx:                                   # (a) chính positive
            return True
        row = corpus[idx]
        if row.get("doc_number") == pos_dn and row.get("article") == pos_art:  # (b) trùng (dn,art)
            return True
        if WIN_LO <= r_pos < WIN_HI and abs(rank - r_pos) <= ADJ_GUARD:        # (c) adjacency ±2
            return True
        return False

    # 3+4. cửa sổ [10,60) sau loại trừ
    window = [(rk, ranking[rk]) for rk in range(WIN_LO, min(WIN_HI, len(ranking)))
              if not excluded(rk, ranking[rk])]
    cand = [idx for _, idx in window]

    # 5. sample N; nếu thiếu → bù [60,100) rồi fallback random corpus
    if len(cand) >= num_neg:
        negs = rng.sample(cand, num_neg)
    else:
        negs = list(cand)
        topup = [ranking[rk] for rk in range(WIN_HI, min(TOPUP_HI, len(ranking)))
                 if not excluded(rk, ranking[rk]) and ranking[rk] not in negs]
        rng.shuffle(topup)
        negs.extend(topup[:num_neg - len(negs)])
        if len(negs) < num_neg:  # fallback random corpus (rất hiếm)
            chosen = set(negs) | {pos_idx}
            n_corpus = len(corpus)
            guard = 0
            while len(negs) < num_neg and guard < num_neg * 50:
                j = rng.randrange(n_corpus)
                guard += 1
                row = corpus[j]
                if j in chosen:
                    continue
                if row.get("doc_number") == pos_dn and row.get("article") == pos_art:
                    continue
                chosen.add(j)
                negs.append(j)
    return negs, len(negs)


def embed_queries(pairs, batch=EMB_BATCH):
    """Encode tất cả câu hỏi (normalize, max_seq 1024) — KHỚP retrieval cell 10."""
    emb = _load_embedder()
    qs = [p["q"] for p in pairs]
    print(f"Embedding {len(qs)} câu hỏi tổng hợp…")
    return emb.encode(qs, batch_size=batch, normalize_embeddings=True,
                      convert_to_numpy=True, show_progress_bar=True).astype("float32")


def write_split(pairs, q_vecs, corpus_emb, corpus, id2idx, out_path, is_val,
                topk, num_neg, done_path):
    """Đào + ghi 1 split (append-only, resumable). Trả thống kê."""
    rng = random.Random(SEED)
    done = load_done(done_path)
    n_prior = count_lines(out_path)        # dòng đã có từ phiên trước (resume)
    n_new = n_skip_done = n_drop = n_neg_total = 0
    fout = open(out_path, "a", encoding="utf-8")
    dfout = open(done_path, "a", encoding="utf-8")
    try:
        for i, p in enumerate(pairs):
            key = pair_key(p)
            if key in done:
                n_skip_done += 1
                continue
            res = mine_one(p, q_vecs[i], corpus_emb, corpus, id2idx, rng, topk, num_neg)
            if res is None:
                n_drop += 1
                dfout.write(key + "\n")  # đánh dấu đã xử lý (kể cả drop) → không retry
                continue
            neg_idxs, n_neg = res
            neg_texts = [passage_text(corpus[j]) for j in neg_idxs]
            n_neg_total += n_neg
            pos_text = p.get("pos_text") or passage_text(corpus[id2idx[p["pos_id"]]])
            if is_val:
                # eval_pairs.jsonl — PHẢI có "pos" (eval_reranker.load_flagembedding_eval đọc r["pos"]);
                # nếu thiếu "pos" thì loader skip TOÀN BỘ sample. Giữ thêm pos_id/doc_number/article (debug).
                rec = {"query": p["q"], "pos": [pos_text], "neg": neg_texts,
                       "pos_id": p["pos_id"],
                       "pos_doc_number": p.get("pos_doc_number", ""),
                       "pos_article": p.get("pos_article", "")}
            else:
                # train_reranker.jsonl — ĐÚNG schema FlagEmbedding (KHÔNG pos_scores/neg_scores/prompt)
                rec = {"query": p["q"], "pos": [pos_text], "neg": neg_texts}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            dfout.write(key + "\n")
            n_new += 1
            if (i + 1) % 500 == 0:
                fout.flush(); dfout.flush()
                print(f"  …{i + 1}/{len(pairs)} | new {n_new} | drop {n_drop} | skip-done {n_skip_done}")
    finally:
        fout.close(); dfout.close()
    avg_neg = round(n_neg_total / n_new, 2) if n_new else 0
    total_written = n_prior + n_new
    return {"written": total_written, "new": n_new, "drop": n_drop,
            "skip_done": n_skip_done, "avg_neg": avg_neg}


def main():
    ap = argparse.ArgumentParser(description="Đào hard negatives → FlagEmbedding train jsonl.")
    ap.add_argument("--pairs", default=None, help="synth_pairs.jsonl (mặc định: tự tìm)")
    ap.add_argument("--corpus", default=None, help="corpus_articles.jsonl (mặc định: tự tìm)")
    ap.add_argument("--emb", default=None, help="corpus_emb.npy (mặc định: tự tìm/encode+lưu)")
    ap.add_argument("--out-train", default=f"{WORK}/ft/train_reranker.jsonl")
    ap.add_argument("--out-val", default=f"{WORK}/ft/eval_pairs.jsonl")
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--num-neg", type=int, default=NUM_NEG)
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    pairs_path = find_file("synth_pairs.jsonl", args.pairs)
    corpus_path = find_file("corpus_articles.jsonl", args.corpus)
    print(f"pairs : {pairs_path}\ncorpus: {corpus_path}")

    corpus = load_corpus(corpus_path)
    id2idx = {r["id"]: i for i, r in enumerate(corpus)}
    print(f"corpus: {len(corpus)} điều (có doc_number)")

    pairs = read_pairs(pairs_path)
    print(f"synth pairs: {len(pairs)}")
    if not pairs:
        print("Không có cặp nào để xử lý — thoát."); sys.exit(1)

    corpus_emb = load_corpus_emb(corpus, args.emb)
    if len(corpus_emb) != len(corpus):
        print(f"FATAL: corpus_emb ({len(corpus_emb)}) ≠ corpus ({len(corpus)}) sau khi nạp."); sys.exit(2)

    train_pairs, val_pairs = split_train_val(pairs, args.val_frac)
    print(f"split (theo pos_doc_number): train {len(train_pairs)} | val {len(val_pairs)}")

    # embed queries CHO TỪNG split theo đúng thứ tự (giữ index khớp khi đào)
    q_train = embed_queries(train_pairs) if train_pairs else np.zeros((0, corpus_emb.shape[1]), "float32")
    q_val = embed_queries(val_pairs) if val_pairs else np.zeros((0, corpus_emb.shape[1]), "float32")

    for out in (args.out_train, args.out_val):
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print("\n=== Đào TRAIN ===")
    st = write_split(train_pairs, q_train, corpus_emb, corpus, id2idx, args.out_train,
                     is_val=False, topk=args.topk, num_neg=args.num_neg,
                     done_path=args.out_train + ".done")
    print("train:", st, "→", args.out_train)

    print("\n=== Đào VAL (held-out) ===")
    sv = write_split(val_pairs, q_val, corpus_emb, corpus, id2idx, args.out_val,
                     is_val=True, topk=args.topk, num_neg=args.num_neg,
                     done_path=args.out_val + ".done")
    print("val  :", sv, "→", args.out_val)

    print(f"\nXONG. train_rows={st['written']} val_rows={sv['written']} "
          f"avg_neg(train)={st['avg_neg']} avg_neg(val)={sv['avg_neg']}")


if __name__ == "__main__":
    main()
