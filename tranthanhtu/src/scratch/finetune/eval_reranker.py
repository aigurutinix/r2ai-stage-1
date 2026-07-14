"""OFFLINE gate: base vs fine-tuned reranker trên held-out split (TRƯỚC khi tốn lượt leaderboard).

Cho MỖI val query: chấm điểm cả candidate set (positive + hard-negative đã mine), xếp hạng theo
điểm reranker, tính MRR@10 + recall@{1,3,5,10} của positive. Chạy CẢ base (AITeamVN/Vietnamese_Reranker)
LẪN fine-tuned (/kaggle/working/ft_reranker) trên CÙNG candidate set → in bảng so sánh + delta tuyệt đối.

Scoring contract = khớp inference (backend/local_reranker.py + retrieval cell 10):
  pair = [query, passage_str]; logits.view(-1).float() (raw logit, cao = tốt); max_length = RERANK_MAX = 512.
Passage_str ở đây = f"{title}\\n{text}" — KHỚP dạng pos/neg mà mine_hard_negatives.py + train phát ra
(no train/serve skew giữa train↔eval). CHÚ Ý: backend/local_reranker.py hiện join bằng DẤU CÁCH
(f"{title} {text}"), KHÔNG phải '\\n' → muốn 0 skew lúc PHỤC VỤ thật thì sửa local_reranker.py về '\\n'.

HAI nguồn val (auto-detect theo --val):
  A) eval_pairs.jsonl  — schema THỰC TẾ do mine_hard_negatives.py phát ra:
       {"query": str, "pos_id", "pos_doc_number", "pos_article",
        "pos_text": str, "neg": [str, ...]}
     (cũng chấp nhận FlagEmbedding train-format {"query","pos":[str,...],"neg":[...]})
     candidate set = positive(s) + neg; positive = pos_text (1) HOẶC mọi text trong pos.
  B) data/gold_dev.json (+ --retrieved retrieved.json) — 20-item gold thật:
       gold_dev: {"items":[{"id", "question", "gold_articles":["<doc>|<Điều>", ...]}]}
       retrieved: {"<id>": [{"doc_number","clean_name","article","title","text"}, ...]}
     candidate passage = f"{title}\\n{text}"; positive = candidate có (doc_number, article) ∈ gold_articles.

Run:
    python eval_reranker.py --val /kaggle/working/ft/eval_pairs.jsonl \
        --base AITeamVN/Vietnamese_Reranker --ft /kaggle/working/ft_reranker
    # hoặc gold thật:
    python eval_reranker.py --val data/gold_dev.json --retrieved backup/retrieved.json \
        --base AITeamVN/Vietnamese_Reranker --ft /kaggle/working/ft_reranker
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Phải khớp RERANK_MAX_LEN của inference (backend/local_models_config.py) + passage_max_len+query_max_len
# dùng khi train (64 + 448) → KHÔNG train/serve skew.
RERANK_MAX = 512
BATCH = 8          # RR_BATCH=8 đã chứng minh an toàn trên T4 (xem risks trong design spec).
RANKS = (1, 3, 5, 10)
MRR_K = 10


# ---------------------------------------------------------------------------
# Load val: trả về list[dict] đồng nhất {"query": str, "cands": [str], "pos_idx": set[int]}
# ---------------------------------------------------------------------------
def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] bỏ dòng {ln} (JSON lỗi): {e}", file=sys.stderr)
    return rows


def load_flagembedding_eval(path):
    """jsonl val split → samples. positive(s) đặt TRƯỚC rồi đến neg.

    Chấp nhận HAI biến thể positive (auto, không cần config):
      • mine_hard_negatives.py eval_pairs.jsonl:  {"query", "pos_text": str, "neg": [...]}
        (ĐÂY là file thực tế emit cho eval — pos là 1 chuỗi 'pos_text', KHÔNG có key 'pos')
      • FlagEmbedding train-format:                {"query", "pos": [str, ...], "neg": [...]}
    """
    samples = []
    n_no_q = n_no_pos = 0
    for r in _read_jsonl(path):
        q = r.get("query") or r.get("q")  # 'q' phòng khi trỏ nhầm vào synth_pairs.jsonl
        # gom positive: ưu tiên 'pos' (list FlagEmbedding); nếu không có thì dùng 'pos_text' (string).
        pos = r.get("pos")
        if isinstance(pos, str):
            pos = [pos]
        elif not pos:                      # None / [] / missing → thử pos_text
            pt = r.get("pos_text")
            pos = [pt] if pt else []
        pos = [p for p in pos if p]        # bỏ chuỗi rỗng
        neg = [c for c in (r.get("neg") or []) if c]
        if not q:
            n_no_q += 1
            continue
        if not pos:
            n_no_pos += 1
            continue
        # candidate set: pos trước rồi neg → pos_idx = 0..len(pos)-1
        cands = list(pos) + list(neg)
        pos_idx = set(range(len(pos)))
        samples.append({"query": q, "cands": cands, "pos_idx": pos_idx})
    if n_no_q or n_no_pos:
        print(f"[warn] bỏ {n_no_q} dòng thiếu query, {n_no_pos} dòng thiếu positive "
              f"(không có 'pos' lẫn 'pos_text').", file=sys.stderr)
    return samples


def load_gold_dev_eval(gold_path, retrieved_path):
    """gold_dev.json + retrieved.json → samples. positive = candidate có (doc_number, article) ∈ gold."""
    if not retrieved_path or not os.path.exists(retrieved_path):
        sys.exit(f"[fatal] --val là gold_dev cần --retrieved <retrieved.json>; không thấy: {retrieved_path}")
    with open(gold_path, "r", encoding="utf-8") as f:
        gold = json.load(f)
    with open(retrieved_path, "r", encoding="utf-8") as f:
        retrieved = json.load(f)

    samples = []
    for item in gold.get("items", []):
        qid = item.get("id")
        q = item.get("question")
        gold_keys = set(item.get("gold_articles", []))  # "<doc_number>|<article>"
        cand_list = retrieved.get(str(qid)) or retrieved.get(qid) or []
        if not q or not cand_list:
            continue
        cands, pos_idx = [], set()
        for i, c in enumerate(cand_list):
            # passage text = f"{title}\n{text}" (inference-identical, khớp dạng pos/neg lúc mine/train)
            cands.append(f"{c.get('title', '')}\n{c.get('text', '')}")
            key = f"{c.get('doc_number', '')}|{c.get('article', '')}"
            if key in gold_keys:
                pos_idx.add(i)
        if not pos_idx:
            # gold không nằm trong candidate set → recall trần của retrieval, reranker không cứu được; bỏ.
            continue
        samples.append({"query": q, "cands": cands, "pos_idx": pos_idx})
    return samples


def load_val(path, retrieved_path):
    if not os.path.exists(path):
        sys.exit(f"[fatal] không thấy --val: {path}")
    # auto-detect: .json (gold_dev) vs .jsonl (FlagEmbedding eval split)
    is_gold = path.endswith(".json") and not path.endswith(".jsonl")
    if is_gold:
        # phân biệt thêm bằng nội dung phòng khi đặt tên nhầm
        try:
            with open(path, "r", encoding="utf-8") as f:
                head = json.load(f)
            if isinstance(head, dict) and "items" in head:
                return load_gold_dev_eval(path, retrieved_path)
        except json.JSONDecodeError:
            pass
    return load_flagembedding_eval(path)


# ---------------------------------------------------------------------------
# Reranker: load + score (contract khớp backend/local_reranker.py)
# ---------------------------------------------------------------------------
def load_reranker(model_path, device):
    # model_path = HF hub id (vd 'AITeamVN/Vietnamese_Reranker') HOẶC local dir
    # (vd /kaggle/working/ft_reranker). from_pretrained xử lý cả hai.
    print(f"  load reranker: {model_path} on {device} ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    if device.type == "cuda":
        model = model.half()   # fp16 như inference
    return tok, model


@torch.no_grad()
def score_pairs(tok, model, query, cands, device):
    """Raw logit cho mỗi (query, cand). Higher = better. Batched, truncation tới RERANK_MAX."""
    pairs = [[query, c] for c in cands]
    scores = []
    for i in range(0, len(pairs), BATCH):
        batch = pairs[i:i + BATCH]
        inputs = tok(
            batch, padding=True, truncation=True,
            max_length=RERANK_MAX, return_tensors="pt",
        ).to(device)
        logits = model(**inputs).logits.view(-1).float()
        scores.extend(logits.cpu().tolist())
    return scores


# ---------------------------------------------------------------------------
# Metrics: MRR@10 + recall@k. Rank theo điểm giảm dần (tie-break ổn định theo index gốc).
# ---------------------------------------------------------------------------
def eval_model(tok, model, samples, device, tag):
    n = len(samples)
    rr_sum = 0.0
    rec_hits = {k: 0 for k in RANKS}
    for si, s in enumerate(samples, 1):
        scores = score_pairs(tok, model, s["query"], s["cands"], device)
        # order = chỉ số candidate xếp hạng giảm dần theo score; -score để stable-sort giữ tie theo index gốc
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        pos_idx = s["pos_idx"]
        # rank (1-based) của positive ĐẦU TIÊN xuất hiện trong thứ hạng
        first_pos_rank = next((rk for rk, ci in enumerate(order, 1) if ci in pos_idx), None)
        if first_pos_rank is not None:
            if first_pos_rank <= MRR_K:
                rr_sum += 1.0 / first_pos_rank
            for k in RANKS:
                if first_pos_rank <= k:
                    rec_hits[k] += 1
        if si % 50 == 0 or si == n:
            print(f"    [{tag}] {si}/{n}", file=sys.stderr)
    return {
        "mrr@10": rr_sum / n if n else 0.0,
        **{f"recall@{k}": rec_hits[k] / n if n else 0.0 for k in RANKS},
    }


# ---------------------------------------------------------------------------
# Bảng so sánh
# ---------------------------------------------------------------------------
def print_report(base_m, ft_m, n_samples, val_path):
    cols = ["mrr@10"] + [f"recall@{k}" for k in RANKS]
    w = 12
    print()
    print("=" * (w * 4 + 4))
    print(f"  RERANKER OFFLINE GATE  —  {n_samples} val queries  ({os.path.basename(val_path)})")
    print("=" * (w * 4 + 4))
    header = f"{'metric':<{w}}{'base':>{w}}{'fine-tuned':>{w}}{'Δ (ft-base)':>{w}}"
    print(header)
    print("-" * len(header))
    for c in cols:
        b, f = base_m[c], ft_m[c]
        d = f - b
        arrow = "↑" if d > 1e-9 else ("↓" if d < -1e-9 else "·")
        print(f"{c:<{w}}{b:>{w}.4f}{f:>{w}.4f}{('%+.4f %s' % (d, arrow)):>{w}}")
    print("-" * len(header))
    avg_delta = sum(ft_m[c] - base_m[c] for c in cols) / len(cols)
    verdict = "LIFT ✓ (đáng nộp leaderboard)" if avg_delta > 0 else "NO LIFT ✗ (chưa nên tốn lượt)"
    print(f"avg Δ across metrics: {avg_delta:+.4f}  →  {verdict}")
    print("=" * (w * 4 + 4))


def main():
    ap = argparse.ArgumentParser(description="Base vs fine-tuned reranker offline eval.")
    ap.add_argument("--val", required=True,
                    help="eval_pairs.jsonl (FlagEmbedding format) HOẶC data/gold_dev.json")
    ap.add_argument("--base", default="AITeamVN/Vietnamese_Reranker",
                    help="base reranker (HF id hoặc local dir)")
    ap.add_argument("--ft", default="/kaggle/working/ft_reranker",
                    help="fine-tuned checkpoint dir")
    ap.add_argument("--retrieved", default=None,
                    help="retrieved.json (CHỈ cần khi --val là gold_dev.json)")
    ap.add_argument("--json-out", default=None,
                    help="ghi report ra file JSON (tùy chọn)")
    args = ap.parse_args()

    samples = load_val(args.val, args.retrieved)
    if not samples:
        sys.exit("[fatal] 0 val sample sau khi load/filter — kiểm tra --val (và --retrieved nếu gold_dev).")
    print(f"[info] loaded {len(samples)} val queries; "
          f"avg cands/query = {sum(len(s['cands']) for s in samples) / len(samples):.1f}", file=sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}", file=sys.stderr)

    # BASE
    print("[info] scoring BASE ...", file=sys.stderr)
    tok_b, model_b = load_reranker(args.base, device)
    base_m = eval_model(tok_b, model_b, samples, device, "base")
    del model_b, tok_b
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # FINE-TUNED. --ft theo contract là 1 local dir (vd /kaggle/working/ft_reranker). Nếu nó TRÔNG
    # như local path (tuyệt đối hoặc có './') mà chưa tồn tại → fail sớm với thông điệp rõ. Nếu là
    # bare HF hub id (vd 'org/model', không phải path) thì để from_pretrained tự resolve như --base.
    looks_local = os.path.isabs(args.ft) or args.ft.startswith(".") or os.path.exists(args.ft)
    if looks_local and not os.path.exists(args.ft):
        sys.exit(f"[fatal] không thấy fine-tuned checkpoint: {args.ft} "
                 f"(chạy train_reranker.py trước, hoặc trỏ --ft tới Added-Input dataset).")
    print("[info] scoring FINE-TUNED ...", file=sys.stderr)
    tok_f, model_f = load_reranker(args.ft, device)
    ft_m = eval_model(tok_f, model_f, samples, device, "ft")
    del model_f, tok_f
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print_report(base_m, ft_m, len(samples), args.val)

    if args.json_out:
        report = {
            "val": args.val,
            "n_samples": len(samples),
            "base": base_m,
            "fine_tuned": ft_m,
            "delta": {k: ft_m[k] - base_m[k] for k in base_m},
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[info] report → {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
