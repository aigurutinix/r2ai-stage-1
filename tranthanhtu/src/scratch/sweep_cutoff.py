"""Sweep cutoff configs OFFLINE → nhiều biến thể results.json cho leaderboard (không cần GPU).

Đọc `retrieved.json` GIÀU (top-N + score, từ Kaggle Phase A bản mới) + `results.json` gốc
(đã có answer LLM) → với mỗi cấu hình cutoff: dựng lại relevant_docs/articles + gắn lại khối
"Căn cứ pháp lý áp dụng" vào answer (giữ prose LLM) → results_<tag>.json + submission_<tag>.zip.

Nộp lần lượt các zip lên leaderboard (10 bài/ngày vòng public) để chọn cutoff tốt nhất —
KHÔNG chạy lại retrieval/LLM. (Cutoff chặt hơn mặc định: prose có thể còn "Điều N" ngoài tập
cutoff → tính vào predicted, hơi giảm precision; sweep chủ yếu để NỚI nên không sao.)

Run:
    python scratch/sweep_cutoff.py --retrieved backup/retrieved.json --base results.json
"""
import argparse
import json
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
from retrieval_cutoff import apply_cutoff, drop_superseded

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CIT_MARKER = "\n\nCăn cứ pháp lý áp dụng:"

# (tag, top_k, margin, min_score, validity_filter, sibling_expand, llm_verified) — lưới VÒNG 7.
# Vòng 6 (nới mù): F2 ~0.49, P sập 0.3, R chỉ 0.6 → trần recall CACHE top-12 ≈ 0.62-0.65
# (top-1 leaderboard R 0.7253 = ngoài cache ta). Nới mù chết vì nhặt 4 đá / 1 vàng →
# vòng 7 NỚI CÓ KIỂM SOÁT: chỉ giữ candidate được Qwen chấm CÓ (verified.json, --verified).
# Luôn giữ top-1 (sàn recall). v_k6 ≈ điểm cân bằng kỳ vọng; t3m15 = anchor đã đo 0.5371.
GRID = [
    ("t3m15",  3,  1.5,  None, True, False, False),  # anchor (so 0.5371 — phễu-80 có đổi tight-cutoff?)
    ("t5m3",   5,  3.0,  None, True, False, False),  # nới vừa — pool sâu có làm wide khả thi hơn round 6?
    ("t6m4",   6,  4.0,  None, True, False, False),  # nới hơn
    ("v_k6",   6,  None, None, True, False, True),   # verify (cần --verified từ Phase V)
    ("v_k8",   8,  None, None, True, False, True),
    ("v_k10",  10, None, None, True, False, True),
]

# Sibling expand: sau cutoff, với mỗi văn bản đã giữ → thêm tối đa 1 điều TỐT NHẤT còn lại
# của cùng văn bản (từ top-12 đã lọc hiệu lực) nếu điểm >= top - SIB_MARGIN. Cap tổng SIB_CAP.
SIB_MARGIN, SIB_CAP = 5.0, 5


def expand_siblings(ctx, cands):
    if not ctx:
        return ctx
    top = ctx[0].get("score", 0.0)
    have = {(d.get("doc_number"), d.get("article")) for d in ctx}
    kept_docs = [d.get("doc_number") for d in ctx]
    out = list(ctx)
    for dn in kept_docs:
        if len(out) >= SIB_CAP:
            break
        sib = next((c for c in cands
                    if c.get("doc_number") == dn
                    and (c.get("doc_number"), c.get("article")) not in have
                    and c.get("score", -99) >= top - SIB_MARGIN), None)
        if sib:
            out.append(sib); have.add((sib.get("doc_number"), sib.get("article")))
    return out


def build_fields(ctx):
    docs, arts = [], []
    for d in ctx:
        c, nm, a = d.get("doc_number", ""), d.get("clean_name", ""), d.get("article", "")
        if not c or not a:
            continue
        if f"{c}|{nm}" not in docs:
            docs.append(f"{c}|{nm}")
        if f"{c}|{nm}|{a}" not in arts:
            arts.append(f"{c}|{nm}|{a}")
    return docs, arts


def reattach_citations(base_answer, ctx):
    """Bỏ khối căn cứ cũ → đảm bảo mọi 'Điều N' trong ctx xuất hiện trong answer text."""
    i = base_answer.find(_CIT_MARKER)
    body = (base_answer[:i] if i != -1 else base_answer).rstrip()
    present = set(re.findall(r"Điều\s+(\d+)", body))
    miss = {}
    for d in ctx:
        m = re.match(r"Điều\s+(\d+)", d.get("article", ""))
        if m and m.group(1) not in present:
            miss.setdefault(d.get("clean_name", ""), []).append(d["article"])
    if not miss:
        return body
    parts = [f"{', '.join(a)} ({nm})" for nm, a in miss.items()]
    return body + _CIT_MARKER + " " + "; ".join(parts) + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieved", default=os.path.join(REPO_ROOT, "backup", "retrieved.json"))
    ap.add_argument("--verified", default=None,
                    help="verified.json từ Phase V (qid → list CÓ/KHÔNG theo thứ tự candidate) — bật chế độ v_*")
    ap.add_argument("--base", default=os.path.join(REPO_ROOT, "results.json"),
                    help="results.json gốc (có answer LLM). Thiếu → answer = chỉ liệt kê căn cứ.")
    ap.add_argument("--questions", default=os.path.join(REPO_ROOT, "data", "stage1_questions.json"))
    ap.add_argument("--outdir", default=os.path.join(REPO_ROOT, "sweep"))
    args = ap.parse_args()

    cache = {int(k): v for k, v in json.load(open(args.retrieved, encoding="utf-8")).items()}
    questions = json.load(open(args.questions, encoding="utf-8"))

    # Gắn llm_ok theo INDEX (trước mọi filter — verified.json align với thứ tự cache gốc)
    if args.verified:
        flags = {int(k): v for k, v in json.load(open(args.verified, encoding="utf-8")).items()}
        n_ok = n_all = 0
        for qid, cands in cache.items():
            f = flags.get(qid, [])
            for i, c in enumerate(cands):
                c["llm_ok"] = bool(f[i]) if i < len(f) else False
                n_ok += c["llm_ok"]; n_all += 1
        print(f"verified: {len(flags)} câu | CÓ {n_ok}/{n_all} ({100*n_ok/max(n_all,1):.0f}%)")

    # cache phải GIÀU (có 'score') mới sweep được
    sample = next((v for v in cache.values() if v), [])
    if sample and "score" not in sample[0]:
        sys.exit("⚠ retrieved.json KHÔNG có 'score' (cache cũ post-cutoff, ~1.19 điều). Chạy lại Phase A "
                 "bản mới (lưu top-12+score) rồi mới sweep được.")

    grid = GRID if args.verified else [g for g in GRID if not g[6]]
    if not args.verified and len(grid) < len(GRID):
        print(f"(thiếu --verified → bỏ qua {len(GRID)-len(grid)} config v_*; chạy {len(grid)} config non-verify)")

    base = {}
    if os.path.exists(args.base):
        base = {int(r["id"]): r.get("answer", "") for r in json.load(open(args.base, encoding="utf-8"))}
        print(f"base answers: {len(base)} câu (giữ prose LLM, gắn lại căn cứ theo cutoff)")
    else:
        print("(không có results.json gốc → answer = chỉ liệt kê căn cứ; QA sẽ kém, chỉ để đo IR)")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\n{'tag':10} {'avg_art':>8} {'avg_doc':>8}")
    for tag, tk, mg, mn, filt, sib, ver in grid:
        rows, n_art, n_doc = [], 0, 0
        for q in questions:
            qid = int(q["id"])
            cands = cache.get(qid, [])
            if filt:
                cands = drop_superseded(cands)   # lọc hiệu lực TRƯỚC cutoff → slot đôn lên
            if ver:
                # nới CÓ KIỂM SOÁT: top-1 luôn giữ (sàn recall) + các candidate Qwen chấm CÓ, cap tk
                ctx = (cands[:1] + [c for c in cands[1:] if c.get("llm_ok")])[:tk] if cands else []
            else:
                ctx = apply_cutoff(cands, tk, mn, mg, score_key="score")
            if sib:
                ctx = expand_siblings(ctx, cands)  # kéo thêm điều cùng văn bản đã giữ
            rd, ra = build_fields(ctx)
            n_art += len(ra); n_doc += len(rd)
            if not ctx:
                ans = "Chưa tìm thấy căn cứ pháp lý phù hợp. Khuyến nghị tham vấn luật sư."
            elif base.get(qid):
                ans = reattach_citations(base[qid], ctx)
            else:
                ans = "Căn cứ pháp lý liên quan: " + "; ".join(ra)
            rows.append({"id": qid, "question": q["question"], "answer": ans,
                         "relevant_docs": rd, "relevant_articles": ra})
        rows.sort(key=lambda r: r["id"])
        rj = os.path.join(args.outdir, f"results_{tag}.json")
        json.dump(rows, open(rj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        with zipfile.ZipFile(os.path.join(args.outdir, f"submission_{tag}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.write(rj, arcname="results.json")
        print(f"{tag:10} {n_art/len(questions):8.2f} {n_doc/len(questions):8.2f}  → submission_{tag}.zip")

    print(f"\n✓ {len(grid)} biến thể trong {args.outdir}/ — nộp lần lượt lên leaderboard, ghi F2 lại.")


if __name__ == "__main__":
    main()
