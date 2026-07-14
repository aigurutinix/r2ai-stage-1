#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_synthetic_pairs.py — Sinh câu hỏi tiếng Việt tổng hợp cho mỗi điều luật (aspect-guided)
bằng Qwen2.5-7B-Instruct, lọc chất lượng (độ dài / echo / dedup) rồi gate bằng BGE-M3 top-40
self-retrieval. Resumable + checkpointed, low-RAM. Chạy trên Kaggle 2×T4.

Pipeline (theo arXiv 2412.00657):
  1. Đọc corpus_articles.jsonl (lấy mẫu phân tầng theo legal_type để không thiên Thông tư).
  2. Qwen sinh 1-5 khía cạnh + 1 câu hỏi/khía cạnh (batch GPU, left-pad, fp16, device_map=auto).
  3. Lọc: len, regex echo ("Điều N" / số hiệu VB / generic stems), dedup trong cùng pos_id.
  4. Gate: load AITeamVN/Vietnamese_Embedding 1 lần, encode mọi câu hỏi, chấm với corpus_emb.npy,
     GIỮ pair chỉ khi pos_id nằm trong top-40 retrieved cho q.
  5. Ghi synth_pairs.jsonl {q, pos_id, pos_doc_number, pos_article, pos_text, aspect} + .done sidecar.

Hợp đồng field (pairs_format): gen GHI {q, pos_id, pos_doc_number, pos_article, pos_text, aspect};
mine_hard_negatives.py ĐỌC đúng các field này.

Resumable: phần Qwen sinh được checkpoint qua raw_questions.jsonl + .done sidecar (id corpus đã xử lý);
restart 12h-session bỏ qua điều đã sinh. Bước gate BGE-M3 chạy lại trên toàn bộ raw (rẻ, ~vài phút).

Usage (Kaggle):
  python gen_synthetic_pairs.py --max-articles 20000 --per-article 3
  python gen_synthetic_pairs.py --corpus /kaggle/input/.../corpus_articles.jsonl \
      --out /kaggle/working/ft/synth_pairs.jsonl --limit 500   # smoke test
"""
import argparse
import glob
import json
import os
import random
import re
import sys

# ----------------------------- Cấu hình mặc định (theo spec) -----------------------------
GEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
EMBED_ID = "AITeamVN/Vietnamese_Embedding"  # BGE-M3, dùng đúng như retrieval cell 10
DEFAULT_OUT = "/kaggle/working/ft/synth_pairs.jsonl"
MAX_ARTICLES_DEFAULT = 20000   # cap mẫu phân tầng (~50-60K câu thô), --max-articles để chỉnh
GEN_BATCH = 8                  # batch articles cho Qwen trên 2×T4
ARTICLE_BODY_CAP = 3000        # cắt body điều luật cho prompt
MAX_NEW_TOKENS = 512
TOP40 = 40                     # ngưỡng self-retrieval gate
EMBED_MAX_SEQ = 1024           # khớp retrieval cell 10
SEED = 42

# Generic stems cần loại (echo guard) — câu hỏi rỗng nghĩa / hỏi vòng vo về "quy định nào".
GENERIC_STEMS = [
    "quy định nào",
    "như thế nào về việc theo",
    "theo quy định nào",
    "được quy định ở đâu",
    "nằm ở điều nào",
]
RE_DIEU = re.compile(r"Điều\s*\d+", re.IGNORECASE)
RE_DOCNUM = re.compile(r"\d+\s*/\s*\d+\s*/\s*[A-ZĐ]")  # số hiệu VB: 04/2017/QH14

SYSTEM_PROMPT = (
    "Bạn là chuyên gia pháp lý Việt Nam. Cho một điều luật, hãy xác định 1-5 KHÍA CẠNH "
    "riêng biệt mà điều luật này trả lời, và với MỖI khía cạnh viết MỘT câu hỏi tự nhiên "
    "bằng tiếng Việt mà một người dân/doanh nghiệp thực sự sẽ hỏi. Câu hỏi PHẢI trả lời "
    "được CHỈ bằng điều luật này. TUYỆT ĐỐI KHÔNG nhắc số điều ('Điều N'), không nhắc số "
    "hiệu văn bản, không dùng từ 'theo quy định nào'. Trả về JSON thuần: "
    '{"pairs":[{"aspect":"...","question":"..."}]}.'
)


# ----------------------------- IO helpers (theo notebook find_file) -----------------------------
def find_corpus(explicit):
    """Tìm corpus_articles.jsonl giống find_file() trong notebook."""
    if explicit and os.path.exists(explicit):
        return explicit
    name = "corpus_articles.jsonl"
    cands = [os.path.join("/kaggle/working", name), name, os.path.join("data", name)]
    cands += glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Không thấy {name}. Truyền --corpus hoặc Add Input / đặt vào /kaggle/working."
    )


def find_corpus_emb():
    """Tìm corpus_emb.npy (row i ↔ corpus line i đã lọc doc_number).
    ƯU TIÊN /kaggle/working (file do cell 10 ghi TRONG session này — đảm bảo khớp
    embedder + thứ tự corpus hiện tại). corpus_emb từ /kaggle/input có thể là bản CŨ
    (vd HF 93K) → length-guard trong gate_top40 sẽ bắt lỗi độ dài; nhưng nếu CÙNG độ dài
    mà KHÁC thứ tự dòng thì align SAI âm thầm → cảnh báo to khi phải fallback sang input."""
    name = "corpus_emb.npy"
    preferred = [os.path.join("/kaggle/working", name), name, os.path.join("data", name)]
    for p in preferred:
        if os.path.exists(p):
            return p
    fallback = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if fallback:
        print(
            f"⚠ Dùng corpus_emb.npy từ /kaggle/input ({fallback[0]}) — KHÔNG phải bản "
            "/kaggle/working do cell 10 ghi. Nếu corpus đã rebuild khác thứ tự, gate sẽ "
            "align SAI (chỉ length-guard bắt được mismatch độ dài). Khuyến nghị: chạy cell "
            "10 trước để sinh /kaggle/working/corpus_emb.npy, hoặc truyền --corpus-emb.",
            file=sys.stderr,
            flush=True,
        )
        return fallback[0]
    return None


def load_corpus(path):
    """Đọc corpus. CHỈ giữ row có doc_number truthy — KHỚP HỆT bộ lọc retrieval cell 10
    để index của corpus_emb.npy (row i) trùng với corpus[i] ở bước gate top-40."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("doc_number"):
                rows.append(r)
    return rows


def passage_text(row):
    """Chuỗi passage HỆT lúc inference (docs_text trong cell 10): f'{title}\\n{text}'."""
    return f"{row.get('title', '')}\n{row.get('text', '')}"


# ----------------------------- Lấy mẫu phân tầng theo legal_type -----------------------------
def stratified_sample(corpus, max_articles, seed):
    """Lấy mẫu cân bằng theo legal_type để không thiên về Thông tư (20K/40K).
    Phân bổ tỉ lệ căn (sqrt) → kéo các loại ít lên, nén loại nhiều xuống; không vượt số có sẵn."""
    if max_articles is None or max_articles <= 0 or len(corpus) <= max_articles:
        return corpus

    rng = random.Random(seed)
    by_type = {}
    for r in corpus:
        by_type.setdefault(r.get("legal_type", "?"), []).append(r)

    # Phân bổ quota theo sqrt(count) để cân bằng đại diện các nhóm hiếm.
    import math
    weights = {t: math.sqrt(len(v)) for t, v in by_type.items()}
    wsum = sum(weights.values()) or 1.0
    quota = {t: int(round(max_articles * w / wsum)) for t, w in weights.items()}

    sampled = []
    for t, rows in by_type.items():
        rng.shuffle(rows)
        q = min(quota.get(t, 0), len(rows))
        sampled.extend(rows[:q])

    # Bù/cắt cho khớp max_articles (do làm tròn).
    rng.shuffle(sampled)
    if len(sampled) > max_articles:
        sampled = sampled[:max_articles]
    elif len(sampled) < max_articles:
        chosen = {id(r) for r in sampled}
        pool = [r for r in corpus if id(r) not in chosen]
        rng.shuffle(pool)
        sampled.extend(pool[: max_articles - len(sampled)])

    rng.shuffle(sampled)
    return sampled


# ----------------------------- Lọc chất lượng câu hỏi -----------------------------
def question_is_valid(q):
    """Áp các luật lọc trong spec (mục FILTERING 1-2). Trả True nếu giữ."""
    if not q:
        return False
    q = q.strip()
    if len(q) < 12 or len(q) > 300:
        return False
    if RE_DIEU.search(q):           # echo "Điều N"
        return False
    if RE_DOCNUM.search(q):         # echo số hiệu văn bản
        return False
    low = q.lower()
    if any(stem in low for stem in GENERIC_STEMS):
        return False
    return True


# ----------------------------- Qwen prompt + parse -----------------------------
def build_user_prompt(row):
    body = (row.get("text", "") or "")[:ARTICLE_BODY_CAP]
    return (
        f"Tên văn bản: {row.get('clean_name', '')}\n"
        f"Loại: {row.get('legal_type', '')}\n"
        f"{row.get('article', '')} {row.get('title', '')}\n"
        f"Nội dung:\n{body}"
    )


def extract_json_pairs(reply):
    """JSON-parse reply của Qwen → list[(aspect, question)]. Robust: bóc khối {...} đầu tiên."""
    if not reply:
        return []
    # bóc khối JSON ngoài cùng (Qwen đôi khi kèm rào ```json)
    start = reply.find("{")
    end = reply.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return []
    blob = reply[start : end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return []
    pairs = data.get("pairs") if isinstance(data, dict) else None
    if not isinstance(pairs, list):
        return []
    out = []
    for p in pairs:
        if not isinstance(p, dict):
            continue
        aspect = str(p.get("aspect", "")).strip()
        question = str(p.get("question", "")).strip()
        if question:
            out.append((aspect or "general", question))
    return out


# ----------------------------- Qwen batch generation -----------------------------
def load_generator():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading generator {GEN_MODEL_ID} (fp16, device_map=auto) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
    tok.padding_side = "left"  # left-pad cho decoder-only generate theo batch
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return tok, model


def qwen_generate(tok, model, rows, temperature, top_p):
    """Sinh 1 batch. Trả list[reply str] song song rows. OOM → lùi từng câu."""
    import torch

    texts = []
    for r in rows:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(r)},
        ]
        texts.append(
            tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        )

    @torch.no_grad()
    def _run(batch_texts):
        enc = tok(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)
        do_sample = temperature and temperature > 0
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tok.pad_token_id,
        )
        gen = out[:, enc.input_ids.shape[1]:]
        res = [tok.decode(g, skip_special_tokens=True).strip() for g in gen]
        del enc, out, gen
        return res

    try:
        return _run(texts)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # device_map="auto" đôi khi báo OOM dưới dạng RuntimeError("CUDA out of memory")
        # thay vì OutOfMemoryError có kiểu → bắt cả hai, nhưng RuntimeError khác phải ném lại.
        if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        res = []
        for t in texts:
            try:
                res.append(_run([t])[0])
            except Exception:
                res.append("")
            torch.cuda.empty_cache()
        return res


# ----------------------------- Resume state -----------------------------
def load_done(done_path):
    done = set()
    if os.path.exists(done_path):
        with open(done_path, encoding="utf-8") as f:
            done.update(ln.strip() for ln in f if ln.strip())
    return done


def load_raw(raw_path):
    """Đọc câu hỏi thô đã sinh (checkpoint trước bước gate). Mỗi dòng:
    {q, pos_id, pos_doc_number, pos_article, pos_text, aspect}.
    DEDUP theo (pos_id, q): nếu một điều bị xử lý lại sau crash (raw có câu của nó nhưng
    .done chưa kịp ghi → resume sinh lại), các dòng trùng sẽ bị gộp về 1 — tránh phình
    train data + giữ đúng khoá logic (pos_id, q) như pairs_format mô tả."""
    items = []
    seen = set()
    n_dup = 0
    if os.path.exists(raw_path):
        with open(raw_path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    it = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                key = (it.get("pos_id"), (it.get("q") or "").strip().lower())
                if key in seen:
                    n_dup += 1
                    continue
                seen.add(key)
                items.append(it)
    if n_dup:
        print(f"load_raw: bỏ {n_dup} dòng trùng (pos_id,q) do resume.", flush=True)
    return items


# ----------------------------- Bước 1-3: sinh + lọc thô (checkpointed) -----------------------------
def generate_raw_questions(corpus_sample, raw_path, done_path, per_article, progress_every):
    """Sinh câu hỏi qua Qwen, lọc len/echo/dedup, append vào raw_path. Resumable qua done_path."""
    done_ids = load_done(done_path)
    todo = [r for r in corpus_sample if r["id"] not in done_ids]
    print(
        f"Qwen gen: {len(corpus_sample)} điều mẫu | đã xong {len(done_ids)} | còn {len(todo)}",
        flush=True,
    )
    if not todo:
        print("Tất cả điều đã sinh xong (resume) → bỏ qua bước Qwen.", flush=True)
        return

    tok, model = load_generator()

    raw_f = open(raw_path, "a", encoding="utf-8")
    done_f = open(done_path, "a", encoding="utf-8")

    n_articles = 0
    n_kept = 0
    try:
        for i in range(0, len(todo), GEN_BATCH):
            batch = todo[i : i + GEN_BATCH]

            # Lần 1: temp 0.7 / top_p 0.9. Lần 2 (retry article fail-parse): temp 0.4.
            replies = qwen_generate(tok, model, batch, temperature=0.7, top_p=0.9)

            retry_rows, retry_pos = [], []
            for j, (row, reply) in enumerate(zip(batch, replies)):
                if not extract_json_pairs(reply):
                    retry_rows.append(row)
                    retry_pos.append(j)
            if retry_rows:
                retry_replies = qwen_generate(
                    tok, model, retry_rows, temperature=0.4, top_p=0.9
                )
                for pos, rep in zip(retry_pos, retry_replies):
                    replies[pos] = rep

            batch_done_ids = []
            for row, reply in zip(batch, replies):
                pairs = extract_json_pairs(reply)
                ptext = passage_text(row)
                rid = row.get("id")
                if not rid:
                    continue  # phòng thủ: corpus row thiếu id → bỏ (data hiện đảm bảo có)
                seen_q = set()
                kept_for_row = 0
                for aspect, question in pairs:
                    if kept_for_row >= per_article:
                        break
                    if not question_is_valid(question):
                        continue
                    qnorm = question.strip().lower()
                    if qnorm in seen_q:        # dedup trong cùng pos_id
                        continue
                    seen_q.add(qnorm)
                    rec = {
                        "q": question.strip(),
                        "pos_id": rid,
                        "pos_doc_number": row.get("doc_number", ""),
                        "pos_article": row.get("article", ""),
                        "pos_text": ptext,
                        "aspect": aspect,
                    }
                    raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept_for_row += 1
                    n_kept += 1
                # đánh dấu done dù 0 câu (skip lần sau, kể cả fail/garbled)
                batch_done_ids.append(rid)
                n_articles += 1

            # FLUSH MỖI BATCH + thứ tự AN TOÀN: persist raw TRƯỚC, rồi mới ghi+persist .done.
            # → .done KHÔNG BAO GIỜ vượt trước câu hỏi đã ghi đĩa; crash giữa chừng chỉ khiến
            #   điều bị sinh lại (dedup ở load_raw gộp), không mất align/không bỏ sót.
            raw_f.flush()
            os.fsync(raw_f.fileno())
            for rid in batch_done_ids:
                done_f.write(rid + "\n")
            done_f.flush()
            os.fsync(done_f.fileno())

            if (n_articles) % progress_every < GEN_BATCH:
                print(
                    f"  gen {n_articles}/{len(todo)} điều | {n_kept} câu hỏi thô giữ lại",
                    flush=True,
                )
    finally:
        raw_f.flush()
        raw_f.close()
        done_f.flush()
        done_f.close()
        # giải phóng VRAM trước bước gate BGE-M3
        try:
            import gc
            import torch

            del model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"Qwen gen xong: {n_articles} điều mới, {n_kept} câu hỏi thô.", flush=True)


# ----------------------------- Bước 4: gate BGE-M3 top-40 self-retrieval -----------------------------
def gate_top40(raw_items, corpus, corpus_emb_path, out_path, progress_every):
    """Giữ (q, pos_id) chỉ khi pos_id ∈ top-40 BGE-M3 retrieval của q. Ghi synth_pairs.jsonl."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import torch

    if not raw_items:
        print("Không có câu hỏi thô nào để gate.", flush=True)
        open(out_path, "w", encoding="utf-8").close()
        return 0

    corpus_emb = np.load(corpus_emb_path).astype("float32")
    if len(corpus_emb) != len(corpus):
        raise ValueError(
            f"corpus_emb.npy ({len(corpus_emb)}) ≠ corpus đã lọc doc_number ({len(corpus)}). "
            "Phải dùng corpus_emb khớp corpus hiện tại (xem retrieval cell 10)."
        )
    id_to_idx = {r["id"]: i for i, r in enumerate(corpus)}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Gate BGE-M3 top-{TOP40}: load {EMBED_ID} on {dev} ...", flush=True)
    emb = SentenceTransformer(EMBED_ID, device=dev)
    emb.max_seq_length = EMBED_MAX_SEQ

    queries = [it["q"] for it in raw_items]
    q_emb = emb.encode(
        queries,
        batch_size=128,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    out_f = open(out_path, "w", encoding="utf-8")
    kept = 0
    missing_pos = 0
    try:
        for n, it in enumerate(raw_items):
            pos_idx = id_to_idx.get(it["pos_id"])
            if pos_idx is None:
                missing_pos += 1
                continue
            scores = corpus_emb @ q_emb[n]
            # top-40 chỉ số (đủ kiểm tra membership, rẻ hơn full argsort cho mỗi câu)
            top_idx = np.argpartition(scores, -TOP40)[-TOP40:]
            if pos_idx in top_idx:
                out_f.write(json.dumps(it, ensure_ascii=False) + "\n")
                kept += 1
            if (n + 1) % progress_every == 0:
                out_f.flush()
                rate = 100.0 * kept / (n + 1)
                print(
                    f"  gate {n + 1}/{len(raw_items)} | giữ {kept} ({rate:.1f}%)",
                    flush=True,
                )
    finally:
        out_f.flush()
        out_f.close()

    keep_rate = 100.0 * kept / max(1, len(raw_items))
    print(
        f"Gate xong: giữ {kept}/{len(raw_items)} ({keep_rate:.1f}%) | pos_id missing {missing_pos}",
        flush=True,
    )
    if keep_rate < 40.0:
        print(
            "⚠ keep-rate < 40% — prompt Qwen hoặc cap body cần tinh chỉnh (xem risk GENERATION QUALITY).",
            flush=True,
        )
    return kept


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Sinh câu hỏi tổng hợp aspect-guided (Qwen + BGE-M3 gate).")
    ap.add_argument("--corpus", default=None, help="đường dẫn corpus_articles.jsonl")
    ap.add_argument("--out", default=DEFAULT_OUT, help="đường dẫn synth_pairs.jsonl đầu ra")
    ap.add_argument("--limit", type=int, default=None,
                    help="giới hạn TỔNG số điều xử lý (smoke test); None = dùng --max-articles")
    ap.add_argument("--max-articles", type=int, default=MAX_ARTICLES_DEFAULT,
                    help="cap mẫu phân tầng theo legal_type (mặc định 20000)")
    ap.add_argument("--per-article", type=int, default=3,
                    help="số câu hỏi tối đa giữ lại mỗi điều (mặc định 3)")
    ap.add_argument("--progress-every", type=int, default=200, help="in tiến độ mỗi N")
    ap.add_argument("--corpus-emb", default=None, help="đường dẫn corpus_emb.npy (mặc định auto-find)")
    args = ap.parse_args()

    random.seed(SEED)

    corpus_path = find_corpus(args.corpus)
    print(f"corpus: {corpus_path}", flush=True)
    corpus = load_corpus(corpus_path)
    print(f"corpus rows (có doc_number): {len(corpus)}", flush=True)

    # Mẫu: --limit ưu tiên (smoke test) > --max-articles (phân tầng).
    if args.limit is not None:
        cap = min(args.limit, len(corpus))
        sample = stratified_sample(corpus, cap, SEED)
        print(f"--limit {args.limit} → mẫu phân tầng {len(sample)} điều", flush=True)
    else:
        sample = stratified_sample(corpus, args.max_articles, SEED)
        print(f"--max-articles {args.max_articles} → mẫu phân tầng {len(sample)} điều", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out_path = args.out
    raw_path = out_path + ".raw.jsonl"     # checkpoint câu hỏi thô (trước gate)
    done_path = out_path + ".done"         # sidecar: id corpus đã sinh xong

    # Bước 1-3: Qwen sinh + lọc thô (resumable, low-RAM append).
    generate_raw_questions(
        sample, raw_path, done_path, args.per_article, args.progress_every
    )

    # Bước 4: gate BGE-M3 top-40 → synth_pairs.jsonl.
    corpus_emb_path = args.corpus_emb or find_corpus_emb()
    if not corpus_emb_path:
        print(
            "⚠ Không thấy corpus_emb.npy → BỎ QUA gate top-40, ghi raw làm synth_pairs.jsonl. "
            "Truyền --corpus-emb để bật gate (mạnh khuyến nghị).",
            file=sys.stderr,
            flush=True,
        )
        raw_items = load_raw(raw_path)
        with open(out_path, "w", encoding="utf-8") as f:
            for it in raw_items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"Ghi {len(raw_items)} pair (CHƯA gate) → {out_path}", flush=True)
        return

    raw_items = load_raw(raw_path)
    print(f"Đọc {len(raw_items)} câu hỏi thô từ checkpoint để gate.", flush=True)
    kept = gate_top40(raw_items, corpus, corpus_emb_path, out_path, args.progress_every)
    print(f"\n>>> Hoàn tất: {kept} synth pairs → {out_path}", flush=True)
    print(">>> Bước tiếp: mine_hard_negatives.py đọc file này.", flush=True)


if __name__ == "__main__":
    main()
