"""LLM-judge (v10): Qwen lọc điều nhiễu khỏi kết quả truy hồi.

Với mỗi câu hỏi trong v9, gửi Qwen danh sách điều ứng viên (kèm toàn văn) và yêu cầu
xác định CÓ/KHÔNG từng điều. Giữ điều được CÓ; fallback top-1 gốc nếu tất cả bị lọc.

Mục tiêu: tăng precision (hiện 0.28) mà không giảm recall.

Resumable: cache kết quả judge vào data/judge_cache.json — ctrl+C bất cứ lúc nào.

Usage:
  python scripts/llm_judge.py --in data/submission_v9.json --out data/submission_v10.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
CACHE_FILE = ROOT / "data" / "judge_cache.json"

_SYSTEM = (
    "Bạn là chuyên gia pháp luật Việt Nam. Nhiệm vụ: LỌC BỎ những điều luật LẠC khỏi danh "
    "sách truy hồi. Mặc định GIỮ; chỉ loại điều thực sự không liên quan. Lý luận ngắn gọn."
)

_USER = """Câu hỏi: {question}

Dưới đây là {n} điều luật được truy hồi. Với mỗi điều, đọc nội dung rồi quyết định:
- CÓ (GIỮ): điều TRẢ LỜI câu hỏi, HOẶC quy định chi tiết / hướng dẫn / cụ thể hóa nội dung
  câu hỏi — KỂ CẢ nghị định, thông tư hướng dẫn thi hành luật (chúng thường chứa chi tiết
  mà điều luật gốc không nêu, nên VẪN GIỮ).
- KHÔNG (BỎ): điều thuộc CHỦ ĐỀ / LĨNH VỰC KHÁC HẲN, không dính tới câu hỏi; HOẶC là phiên
  bản CŨ đã bị văn bản mới hơn thay thế.
Khi PHÂN VÂN → GIỮ (CÓ). Chỉ BỎ khi chắc chắn điều đó lạc đề.

{articles_block}

Trả lời theo ĐÚNG định dạng sau — mỗi dòng: số thứ tự, dấu hai chấm, CÓ hoặc KHÔNG, gạch ngang, lý do ngắn:
1: CÓ — [lý do]
2: KHÔNG — [lý do]
...
Bắt buộc xuất đủ {n} dòng."""

_VERDICT = re.compile(r"^\s*(\d+)\s*[:\.]\s*(CÓ|KHÔNG|CO|KHONG)", re.IGNORECASE)
_TRAILING_SO = re.compile(r"\s+số:?\s*$")


def _law_name(p: dict) -> str:
    """Khớp logic _law_name trong tests/build_submission.py."""
    loai = (p.get("loai_van_ban") or "").strip()
    title = (p.get("title") or "").strip()
    title = _TRAILING_SO.sub("", title).strip()
    if loai and title.lower().startswith(loai.lower()):
        return title
    return f"{loai} {title}".strip() if loai else title


def parse_verdicts(text: str, n: int) -> list[bool]:
    """Trả về list[bool] độ dài n (True=CÓ). Dòng thiếu → True (giữ nguyên)."""
    verdicts = {}
    for line in (text or "").splitlines():
        m = _VERDICT.match(line)
        if m:
            idx = int(m.group(1))
            keep = m.group(2).upper() in ("CÓ", "CO")
            verdicts[idx] = keep
    return [verdicts.get(i + 1, True) for i in range(n)]


def build_articles_block(arts: list[str], texts: dict[str, str]) -> str:
    parts = []
    for i, art in enumerate(arts, 1):
        text = texts.get(art, "")
        truncated = text[:1200] if len(text) > 1200 else text
        parts.append(f"Điều {i} — {art}\n{truncated}")
    return "\n\n".join(parts)


def load_text_lookup(collection: str) -> dict[str, str]:
    """Scroll toàn bộ collection Qdrant → dict art_key → text.

    art_key = 'so_ky_hieu|ten_van_ban|Điều N'  (khớp format relevant_articles)
    """
    import os
    os.environ.setdefault("USE_TF", "0")
    from backend.qdrant_store import QdrantStore

    print(f"Đang scroll {collection} để lấy text điều...", flush=True)
    client = QdrantStore().client
    total = client.count(collection, exact=True).count
    print(f"  Tổng: {total:,} điểm", flush=True)

    lookup: dict[str, str] = {}
    offset, done = None, 0
    while True:
        pts, offset = client.scroll(
            collection, limit=2000, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for p in pts:
            pl = p.payload or {}
            sk = str(pl.get("so_ky_hieu") or "")
            ds = pl.get("dieu_so")
            text = str(pl.get("text") or "")
            if not sk or ds is None:
                continue
            # Dùng _law_name để khớp format trong submission (so_ky_hieu|ten_van_ban|Điều N)
            ten = _law_name(pl)
            art_key = f"{sk}|{ten}|Điều {ds}"
            lookup[art_key] = text
        done += len(pts)
        if done % 20000 < 2000:
            print(f"  Scrolled {done:,}/{total:,}", flush=True)
        if offset is None:
            break
    print(f"Đã nạp text cho {len(lookup):,} điều.", flush=True)
    return lookup


def judge_question(
    question: str,
    arts: list[str],
    texts: dict[str, str],
    llm,
) -> list[bool]:
    """Gọi Qwen 1 lần để judge tất cả điều của 1 câu hỏi."""
    if len(arts) <= 1:
        return [True] * len(arts)   # chỉ 1 điều → giữ nguyên

    block = build_articles_block(arts, texts)
    user = _USER.format(question=question, n=len(arts), articles_block=block)
    try:
        out = llm.complete(_SYSTEM, user, think=True)
        return parse_verdicts(out, len(arts))
    except Exception as e:
        print(f"  LLM lỗi: {e}", flush=True)
        return [True] * len(arts)   # lỗi → giữ nguyên


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/submission_v9.json")
    ap.add_argument("--out", default="data/submission_v10.json")
    ap.add_argument("--collection", default="vbpl_aiteam")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import os
    os.environ.setdefault("QDRANT_COLLECTION", args.collection)

    from backend.llm import LLMClient
    llm = LLMClient()

    # Nạp lookup text từ Qdrant
    texts = load_text_lookup(args.collection)

    # Nạp submission v9
    v9 = json.loads((ROOT / args.inp).read_text(encoding="utf-8"))

    # Nạp cache
    cache: dict[str, list[bool]] = (
        json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if CACHE_FILE.exists()
        else {}
    )

    qs_todo = v9 if not args.limit else v9[: args.limit]
    print(f"Tổng: {len(qs_todo)} câu | Đã cache: {len(cache)}", flush=True)

    out_path = ROOT / args.out
    out = []
    n_filtered = n_fallback = 0
    t0 = time.time()
    done_judge = 0

    for i, row in enumerate(qs_todo, 1):
        qid = str(row["id"])
        arts = row.get("relevant_articles", [])

        if not arts:
            out.append(row)
            continue

        # Judge (có cache)
        if qid not in cache:
            verdicts = judge_question(row["question"], arts, texts, llm)
            cache[qid] = verdicts
            done_judge += 1
            # Ghi cache mỗi 20 câu mới
            if done_judge % 20 == 0:
                CACHE_FILE.write_text(
                    json.dumps(cache, ensure_ascii=False), encoding="utf-8"
                )
        else:
            verdicts = cache[qid]

        # Căn chỉnh verdicts với arts (nếu cache cũ độ dài lệch)
        if len(verdicts) != len(arts):
            verdicts = [True] * len(arts)

        kept_arts = [a for a, v in zip(arts, verdicts) if v]
        if not kept_arts:
            kept_arts = arts[:1]   # fallback: giữ top-1
            n_fallback += 1
        elif len(kept_arts) < len(arts):
            n_filtered += 1

        # Rebuild docs từ arts đã lọc
        docs: list[str] = []
        seen: set = set()
        for a in kept_arts:
            parts = a.split("|")
            doc = "|".join(parts[:2]) if len(parts) >= 2 else parts[0]
            if doc not in seen:
                seen.add(doc); docs.append(doc)

        out.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "relevant_docs": docs,
            "relevant_articles": kept_arts,
        })

        if i % 50 == 0:
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"  [{i}/{len(qs_todo)}] filtered={n_filtered} fallback={n_fallback} "
                f"{done_judge/(time.time()-t0):.2f} judge/s",
                flush=True,
            )

    # Ghi cuối
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(
        f"XONG: {len(out)} câu | lọc bớt điều={n_filtered} | fallback={n_fallback} "
        f"→ {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
