#!/usr/bin/env python3
"""
Mine hard negatives for 10K data_final samples using dense retrieval.

Strategy:
1. Sample 10K questions from data_final (same topic distribution as test set)
2. Embed questions with Vietnamese_Embedding_v2
3. Search FAISS dense index top-20
4. Dedup candidates by (law_id, article_number)
5. Remove positives (cited articles)
6. Take ranks 5-12 (after dedup+filter) as hard negatives

Output: data_final_hard_negatives.jsonl
Format: {"query": str, "passage": str, "label": int}
"""

import json
import pickle
import random
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

random.seed(42)

BASE = Path("..") # adjust to your project root
DENSE_DIR = BASE / "retrieval_index_dense"
GROUP_SIZE = 8
MAX_PASSAGE_CHARS = 6000

TOPIC_MAP = {
    "doanh_nghiep": "doanh_nghiep", "dau_tu": "doanh_nghiep",
    "chung_khoan": "doanh_nghiep", "bat_dong_san": "doanh_nghiep",
    "ke_toan_kiem_toan": "doanh_nghiep", "so_huu_tri_tue": "doanh_nghiep",
    "tien_te_ngan_hang": "doanh_nghiep",
    "lao_dong_tien_luong": "lao_dong", "lao_dong": "lao_dong",
    "thue_phi_le_phi": "thue", "tai_chinh_nha_nuoc": "thue", "tai_chinh": "thue",
    "thuong_mai": "thuong_mai", "xuat_nhap_khau": "thuong_mai",
    "bo_may_hanh_chinh": "hanh_chinh", "vi_pham_hanh_chinh": "hanh_chinh",
    "hanh_chinh": "hanh_chinh", "thu_tuc_to_tung": "hanh_chinh",
    "dich_vu_phap_ly": "hanh_chinh",
    "bao_hiem": "bhxh",
    "xay_dung_do_thi": "xay_dung", "xay_dung": "xay_dung",
    "giao_duc": "giao_duc",
    "tai_nguyen_moi_truong": "dat_dai", "dat_dai": "dat_dai",
    "quyen_dan_su": "dan_su", "dan_su": "dan_su",
    "giao_thong_van_tai": "giao_thong",
    "trach_nhiem_hinh_su": "hinh_su", "hinh_su": "hinh_su",
    "hon_nhan_gia_dinh": "hon_nhan",
    "linh_vuc_khac": "khac", "van_hoa_xa_hoi": "khac",
    "cong_nghe_thong_tin": "khac", "the_thao_y_te": "khac", "van_hoa": "khac",
}

TOPIC_TARGETS = {
    "doanh_nghiep": 3500, "lao_dong": 900, "thue": 900, "thuong_mai": 800,
    "hanh_chinh": 700, "bhxh": 400, "xay_dung": 300, "giao_duc": 200,
    "dat_dai": 200, "dan_su": 200, "giao_thong": 200, "hinh_su": 200,
    "hon_nhan": 200, "khac": 1300,
}


def load_content_cache():
    CHUNKS_DIR = BASE / "vbpl_dataset" / "chunks"
    DOC_TYPES = [
        "hien_phap", "bo_luat", "luat", "phap_lenh", "nghi_dinh",
        "nghi_quyet", "nghi_quyet_lien_tich", "thong_tu",
        "thong_tu_lien_tich", "quyet_dinh",
    ]
    cc = {}
    for dtype in DOC_TYPES:
        d = CHUNKS_DIR / dtype
        if not d.exists():
            continue
        for fpath in sorted(d.glob("*.json")):
            data = json.loads(fpath.read_text())
            for c in data.get("chunks", []):
                cid = c.get("chunk_id", "")
                content = "\n".join(c.get("content", [])).strip()
                doc_title = c.get("doc_title", "")
                art_no = c.get("article_number", "")
                art_title = c.get("article_title", "")
                header = f"{doc_title} - Điều {art_no}"
                if art_title:
                    header += f". {art_title}"
                if cid and content:
                    cc[cid] = f"{header}\n{content}"[:MAX_PASSAGE_CHARS]
    return cc


def load_data_final():
    data_dir = BASE / "data_final"
    by_topic = defaultdict(list)
    for source_dir in sorted(data_dir.iterdir()):
        if not source_dir.is_dir() or source_dir.name == "__pycache__":
            continue
        for topic_dir in sorted(source_dir.iterdir()):
            if not topic_dir.is_dir():
                continue
            mapped = TOPIC_MAP.get(topic_dir.name, "khac")
            for fpath in sorted(topic_dir.glob("*.json")):
                try:
                    data = json.loads(fpath.read_text(encoding="utf-8"))
                    if not isinstance(data, list):
                        continue
                    for item in data:
                        q = item.get("question", "").strip()
                        cites = item.get("article_cite", [])
                        if not q or not cites:
                            continue
                        cite_keys = set()
                        cite_texts = []
                        for c in cites:
                            if isinstance(c, dict):
                                item_id = c.get("item_id", "")
                                title = c.get("title", "")
                                content = c.get("content", "").strip()
                                doc_title = c.get("doc_title", "")
                                m = re.search(r'Điều\s+(\d+)', title)
                                art_no = m.group(1) if m else ""
                                if item_id and art_no:
                                    cite_keys.add((item_id, art_no))
                                if doc_title:
                                    law_m = re.search(r'(\d+/\d{4}/[A-ZĐa-zđ\d\-]+)', doc_title)
                                    if law_m and art_no:
                                        cite_keys.add(("law:" + law_m.group(1), art_no))
                                if content:
                                    header = doc_title or ""
                                    if title:
                                        header += f" - {title}"
                                    cite_texts.append(f"{header}\n{content}"[:MAX_PASSAGE_CHARS])
                        if cite_keys and cite_texts:
                            by_topic[mapped].append({
                                "question": q,
                                "cite_keys": cite_keys,
                                "cite_texts": cite_texts,
                            })
                except Exception:
                    continue
    return by_topic


def sample_by_topic(by_topic, targets):
    sampled = []
    for topic, target_n in targets.items():
        available = by_topic.get(topic, [])
        if not available:
            continue
        n = min(target_n, len(available))
        chosen = random.sample(available, n)
        sampled.extend(chosen)
        print(f"  {topic}: {n}/{len(available)}")
    return sampled


def main():
    print("Loading content cache...")
    content_cache = load_content_cache()
    print(f"  {len(content_cache)} chunks")

    print("Loading dense index...")
    metas = pickle.load(open(DENSE_DIR / "metas.pkl", "rb"))
    index = faiss.read_index(str(DENSE_DIR / "faiss.index"))
    print(f"  {index.ntotal} vectors")

    print("Loading embedding model...")
    model = SentenceTransformer("AITeamVN/Vietnamese_Embedding_v2", trust_remote_code=True)
    model.max_seq_length = 512

    print("\nLoading data_final by topic...")
    by_topic = load_data_final()

    print(f"\nSampling 10K by topic distribution...")
    sampled = sample_by_topic(by_topic, TOPIC_TARGETS)
    print(f"  Total sampled: {len(sampled)}")

    questions = [s["question"] for s in sampled]
    print(f"\nEmbedding {len(questions)} questions...")
    q_embeds = model.encode(questions, normalize_embeddings=True, batch_size=256,
                            show_progress_bar=True).astype("float32")

    print("Searching FAISS top-20...")
    D, I = index.search(q_embeds, 20)

    print("Building hard negative pairs...")
    out_rows = []
    stats = {"pos": 0, "neg": 0, "skipped": 0}

    for qi, qa in enumerate(sampled):
        cite_keys = qa["cite_keys"]
        cite_texts = qa["cite_texts"]
        question = qa["question"]

        if not cite_texts:
            stats["skipped"] += 1
            continue

        # Dedup candidates by (law_id, article_number), remove positives
        seen_keys = set()
        non_positive_candidates = []

        for score, idx in zip(D[qi], I[qi]):
            if idx < 0:
                continue
            meta = metas[idx]
            law_id = meta.get("law_id", "")
            art_no = str(meta.get("article_number", ""))
            van_ban_id = meta.get("van_ban_id", "")
            chunk_id = meta.get("chunk_id", "")

            # Dedup
            dedup_key = (law_id, art_no) if law_id else (chunk_id,)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            # Check if positive
            is_positive = False
            for ck in cite_keys:
                if ck[0] == van_ban_id and ck[1] == art_no:
                    is_positive = True
                    break
                if ck[0] == "law:" + law_id and ck[1] == art_no:
                    is_positive = True
                    break
            if is_positive:
                continue

            passage = content_cache.get(chunk_id, "")
            if not passage and chunk_id.startswith("qa_"):
                if van_ban_id and art_no:
                    passage = content_cache.get(f"{van_ban_id}#dieu_{art_no}", "")
            if passage:
                non_positive_candidates.append(passage)

        # Take ranks 5-12 (skip top 4 non-positives, take next 7)
        hard_negs = non_positive_candidates[4:11]

        if len(hard_negs) < 7:
            # Pad from earlier ranks
            for c in non_positive_candidates[:4]:
                if len(hard_negs) >= 7:
                    break
                hard_negs.append(c)

        if len(hard_negs) < 3:
            stats["skipped"] += 1
            continue

        # Pad to exactly 7 if still short
        while len(hard_negs) < 7:
            hard_negs.append(random.choice(hard_negs))
        hard_negs = hard_negs[:7]

        for pos_text in cite_texts:
            out_rows.append({"query": question, "passage": pos_text, "label": 1})
            stats["pos"] += 1
            for neg in hard_negs:
                out_rows.append({"query": question, "passage": neg, "label": 0})
                stats["neg"] += 1

    # Trim to group-aligned
    n_groups = len(out_rows) // GROUP_SIZE
    out_rows = out_rows[:n_groups * GROUP_SIZE]

    print(f"\n=== Results ===")
    print(f"  Positives: {stats['pos']}")
    print(f"  Negatives: {stats['neg']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Total rows: {len(out_rows)} ({n_groups} groups)")

    out_path = BASE / "data_final_hard_negatives.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
