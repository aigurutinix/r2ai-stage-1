"""v66: architecture trial, not prompt-add.

Goal:
- current-version scoring before selection;
- conservative multi-hop retrieval with gated existing subqueries;
- coverage-aware deterministic selector;
- no Qwen concept expansion / HyDE / broad rewrite.

It starts from a full base submission for ids outside the target range, then
rebuilds target rows from retrieval so the output is submit-ready (2000 rows).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("QDRANT_COLLECTION", "vbpl_aiteam")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("EMBED_BACKEND", "st")
os.environ.setdefault("EMBED_ST_MODEL", "AITeamVN/Vietnamese_Embedding_v2")
os.environ.setdefault("HYBRID_SEARCH", "true")
os.environ.setdefault("USE_RERANKER", "true")
os.environ.setdefault("RERANKER_MODEL", "AITeamVN/Vietnamese_Reranker")
os.environ.setdefault("BM25_INDEX_PATH", "data/bm25_vbpl_aiteam.pkl")
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_v39_domain import rebuild_docs  # noqa: E402
from build_v47_filter import version_drop  # noqa: E402
from build_v49_blacklist import KNOWN_OLD, norm_sk  # noqa: E402
from exp_v24 import PENALTY_WHITELIST, _PENALTY, collapse_versions  # noqa: E402
from tests.build_submission_v12 import cand_from_hit  # noqa: E402

QFILE = Path("C:/Users/PHONG/Downloads/R2AIStage1DATA.json")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower().replace("đ", "d")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text).strip()


def prefix(art_or_sk: str) -> str:
    sk = art_or_sk.split("|", 1)[0].strip()
    m = re.match(r"(\d+/\d{4})", unicodedata.normalize("NFKC", sk))
    return m.group(1) if m else sk


def article_no(art: str) -> str:
    p = art.split("|")
    return p[-1].strip() if p else ""


def doc_part(art: str) -> str:
    p = art.split("|")
    return "|".join(p[:2]) if len(p) >= 2 else art


STOP = {
    "cong", "ty", "toi", "nguoi", "phai", "duoc", "khong", "nhu", "the", "nao",
    "neu", "thi", "va", "hoac", "dong", "thoi", "trong", "truong", "hop", "can",
    "ve", "cua", "cho", "nay", "do", "la", "co", "cac", "nhung", "mot", "hai",
    "quy", "dinh", "phap", "luat", "dieu", "khoan",
}


def content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", norm(text)) if t not in STOP}


def is_complex(question: str) -> bool:
    q = f" {norm(question)} "
    signals = [
        " dong thoi ", " vua ", " va neu ", " neu ", " thi ", " hoac ",
        " con ", " trong truong hop ", " mat khac ", " ben canh ",
    ]
    return sum(1 for s in signals if s in q) >= 2 or question.count(",") >= 2 or len(question) > 180


def split_clauses(question: str) -> list[str]:
    q = norm(question)
    parts = re.split(r"\b(?:dong thoi|va neu|neu|thi|vua|con|hoac|va)\b|[,;?]", q)
    out = []
    for part in parts:
        part = part.strip()
        if len(part) >= 24 and len(content_tokens(part)) >= 3:
            out.append(part)
    return out[:5]


def is_penalty_question(question: str) -> bool:
    q = norm(question)
    return any(
        s in q
        for s in [
            "xu phat", "muc phat", "vi pham", "khac phuc hau qua", "tien phat",
            "bi xu ly", "che tai", "trach nhiem phap ly", "rui ro phap ly",
        ]
    )


def is_procedure_question(question: str) -> bool:
    q = norm(question)
    return any(s in q for s in ["thu tuc", "ho so", "trinh tu", "dang ky", "thong bao", "nop", "cap phep", "gia han"])


OBSOLETE_PREFIXES = {
    "03/2003", "10/2012", "28/2011", "33/2005", "43/2010", "44/2003",
    "60/2005", "68/2014", "78/2006", "78/2015", "83/2013", "85/2007",
    "156/2013", "166/2013", "198-CP", "05/2015", "28/2020", "37/2006",
    "06/2006", "24/2007", "47/2010", "113/2004", "164/2003", "123/2012",
    "78/2014", "109/2004", "39/2018",
    "101/2001", "102/2001", "154/2005",
}

HARD_OBSOLETE_PREFIXES = {
    # Replaced penalty/administrative decrees that repeatedly outrank current
    # rules on exact keyword matches. Do not allow high raw score to rescue
    # them in a 2026-cutoff legal QA setting.
    "06/2008", "175/2004", "150/2005", "73/2010", "185/2004", "105/2013",
}

STILL_CURRENT_OLD_PREFIXES = {
    "36/2005",  # Luật Thương mại
    "50/2005",  # Luật Sở hữu trí tuệ
    "54/2010",  # Luật Trọng tài thương mại
    "54/2014",  # Luật Hải quan
    "91/2015",  # Bộ luật Dân sự
    "88/2015",  # Luật Kế toán
}


def still_current_old_candidate(c: dict) -> bool:
    p = c.get("prefix", "")
    text = norm(f"{c.get('art', '')} {c.get('title', '')}")
    if p == "16/2012":
        return True
    if p == "36/2005":
        return "luat thuong mai" in text
    if p == "35/2006":
        return True
    if p == "50/2005":
        return "luat so huu tri tue" in text
    if p == "54/2010":
        return "luat trong tai thuong mai" in text
    if p == "54/2014":
        return "hai quan" in text
    if p == "88/2015":
        return "luat ke toan" in text
    if p == "91/2015":
        return "bo luat dan su" in text
    return p in STILL_CURRENT_OLD_PREFIXES

CURRENT_BOOSTS = {
    "12/2022": 0.10,
    "38/2019": 0.08,
    "45/2019": 0.10,
    "59/2020": 0.12,
    "80/2021": 0.08,
    "88/2015": 0.08,
    "91/2015": 0.10,
    "108/2025": 0.12,
    "125/2020": 0.10,
    "168/2025": 0.16,
}

SUCCESSORS = {
    "03/2003": ["88/2015"],
    "10/2012": ["45/2019"],
    "28/2011": ["80/2021", "38/2019", "108/2025"],
    "33/2005": ["91/2015"],
    "43/2010": ["168/2025", "01/2021"],
    "44/2003": ["45/2019", "145/2020"],
    "60/2005": ["59/2020"],
    "68/2014": ["59/2020"],
    "78/2006": ["38/2019", "108/2025"],
    "78/2015": ["168/2025", "01/2021"],
    "83/2013": ["126/2020", "80/2021", "38/2019"],
    "85/2007": ["38/2019", "108/2025"],
    "156/2013": ["80/2021", "38/2019"],
    "166/2013": ["125/2020"],
    "28/2020": ["12/2022"],
    "47/2010": ["12/2022", "45/2019"],
    "113/2004": ["12/2022", "45/2019"],
    "109/2004": ["168/2025"],
    "39/2018": ["80/2021"],
}

GENERIC_HEADS = (
    "giai thich tu ngu",
    "doi tuong ap dung",
    "pham vi dieu chinh",
    "hieu luc thi hanh",
    "trach nhiem thi hanh",
    "quy dinh chuyen tiep",
    "nguyen tac xu phat",
    "hinh thuc xu phat",
    "muc phat tien tham quyen xu phat",
    "cac hinh thuc xu phat",
)

LLM_SELECTOR_PROMPT = """Bạn là bộ kiểm tra căn cứ pháp lý cho hệ thống RAG.

Nhiệm vụ: xem câu hỏi và danh sách điều luật đã retrieve, rồi đề xuất điều nào nên GIỮ hoặc THÊM để trả lời đúng, đủ, không dư.

Nguyên tắc chọn:
- Ưu tiên điều luật trực tiếp điều chỉnh hành vi/thủ tục/quyền/nghĩa vụ trong câu hỏi.
- Nếu câu hỏi có nhiều vế độc lập, mỗi vế quan trọng cần có căn cứ riêng; không bỏ một vế chỉ vì điều khác có điểm cao hơn.
- Không thêm điều định nghĩa, phạm vi áp dụng, nguyên tắc chung, thẩm quyền/mức phạt chung nếu đã có điều xử lý trực tiếp, trừ khi câu hỏi hỏi đúng nội dung chung đó.
- Nếu có văn bản mới và văn bản cũ cùng nội dung, chọn văn bản còn hiệu lực/mới hơn.
- Nhưng văn bản mới hơn mà chỉ nói nguyên tắc/khung chung KHÔNG được thay văn bản dưới luật còn hiệu lực đang quy định trực tiếp mức phạt, biện pháp, hồ sơ, thời hạn hoặc trình tự.
- Với câu hỏi xử phạt, ưu tiên điều trong nghị định xử phạt trực tiếp hành vi; chỉ giữ luật khung/nguyên tắc nếu câu hỏi hỏi nguyên tắc hoặc không có điều xử phạt trực tiếp.
- Với câu hỏi thủ tục/hồ sơ/thời hạn, ưu tiên điều nêu đúng hồ sơ, trình tự, thời hạn hoặc trách nhiệm xử lý; không chọn điều giao dịch/quản lý điện tử chỉ vì có vài từ khoá liên quan.
- Không chọn điều chỉ trùng vài từ nhưng khác lĩnh vực.
- Nếu chưa đủ căn cứ trong danh sách, ghi rõ vế còn thiếu trong missing, nhưng không tự bịa mã điều ngoài danh sách.

Trả về đúng JSON, không giải thích ngoài JSON:
{{"keep":["C1"],"add":["C2"],"drop":["C3"],"missing":["..."],"reason":"ngắn gọn"}}

Câu hỏi:
{question}

Ứng viên:
{candidates}
"""

LLM_SELECTOR_PROMPT = LLM_SELECTOR_PROMPT.replace(
    "\nTráº£ vá» Ä‘Ãºng JSON",
    """
Few-shot policy patterns:
- Cau hoi xu phat/muc phat: giu nghi dinh xu phat hien hanh dieu chinh truc tiep hanh vi. Neu co van ban cu va van ban moi cung noi dung, bo van ban cu.
- Cau hoi quang cao noi chung: uu tien Luat Quang cao. Chi uu tien Luat Thuong mai khi cau hoi noi ro quang cao thuong mai hoac dich vu quang cao thuong mai.
- Cau hoi hop dong theo mau/dieu kien giao dich chung/khach hang: khong bo Luat Bao ve quyen loi nguoi tieu dung neu dieu do noi truc tiep ve hop dong theo mau hoac dieu kien giao dich chung.
- Cau hoi so huu tri tue ve thu tuc dang ky, tham dinh, phan doi, huy bo/hieu luc van bang: khong bo thong tu/nghi dinh chuyen nganh neu ung vien do noi truc tiep ve thu tuc.
- Cau hoi co nhieu ve doc lap: giu dieu truc tiep cho moi ve quan trong; khong them dieu chung/generic chi vi trung tu khoa.

Tráº£ vá» Ä‘Ãºng JSON""",
)


LLM_SELECTOR_EXTRA_POLICY = """

Additional selector policy patterns:
- Penalty/fine question: keep the current penalty decree article that directly governs the behavior. Drop older penalty documents only when a current direct penalty article covers the same behavior.
- General advertising question: prefer Luat Quang cao. Prefer Luat Thuong mai only when the question clearly asks about commercial advertising, service promotion, or trade-fair context.
- Standard-form contract / general transaction conditions / consumer-customer question: do not drop Luat Bao ve quyen loi nguoi tieu dung when the article directly discusses those contracts or conditions.
- Intellectual-property procedure question: keep specialized IP decree/circular articles when they directly cover filing, examination, opposition, cancellation, validity, amendment, or certificates.
- Franchise / commercial-rights question: keep specialized franchise/commercial-law articles when they directly cover franchisor/franchisee obligations, termination, notice, or force-majeure handling. A newer civil-code article can complement but must not replace the specialized article.
- Multi-hop question: map each independent part of the question to at least one direct article. If one part has no direct candidate, write that part in missing instead of filling it with a generic article.
- Direct specialized article beats generic framework article, even if the framework article is newer, when the specialized article is still in force and directly answers the operational duty, procedure, deadline, notice, file, sanction, or remedy.

Internal checklist before JSON, do not output reasoning:
1. Identify legal domain and all independent question parts.
2. For each kept/added article, verify title/head/snippet directly answers one part.
3. For each dropped article, verify it is obsolete, generic, duplicate, or different domain.
4. Keep JSON only.
"""

LLM_SELECTOR_PROMPT = LLM_SELECTOR_PROMPT + LLM_SELECTOR_EXTRA_POLICY


def head_norm(c: dict) -> str:
    text = re.sub(r"\s+", " ", c.get("text", "") or "").strip()
    head = re.split(r"\s+1\.\s+", text, maxsplit=1)[0][:180]
    out = norm(head)
    out = re.sub(r"^dieu\s+\d+[a-z]?\s*[\.:]?\s*", "", out)
    return out.strip(" .:-")


def year_of_prefix(pfx: str) -> int:
    m = re.search(r"/((?:19|20)\d{2})", pfx or "")
    return int(m.group(1)) if m else 0


def asks_generic(question: str) -> bool:
    q = norm(question)
    return any(s in q for s in ["giai thich", "khai niem", "dinh nghia", "doi tuong ap dung", "nguyen tac"])


def generic_candidate(question: str, c: dict) -> bool:
    if asks_generic(question):
        return False
    h = head_norm(c)
    return any(h.startswith(x) for x in GENERIC_HEADS)


def penalty_doc_candidate(c: dict) -> bool:
    text = norm(f"{c.get('art', '')} {c.get('title', '')}")
    return "xu phat vi pham hanh chinh" in text or "xu ly vi pham hanh chinh" in text


def off_topic(question: str, c: dict) -> bool:
    q = norm(question)
    title_head = norm(f"{c.get('art', '')} {c.get('title', '')} {head_norm(c)}")
    if c.get("prefix") == "36/2005" and "quang cao" in q and any(s in q for s in ["tot nhat", "bi cam", "cam khong", "hinh the", "phuong tien"]):
        if "quang cao thuong mai" in title_head and not any(s in q for s in ["khuyen mai", "xuc tien", "hoi cho", "trien lam", "dich vu quang cao thuong mai"]):
            return True
    if any(s in title_head for s in ["ban kiem soat", "kiem soat vien", "hoi dong quan tri", "thanh vien hop danh", "phan von gop"]) and not any(
        s in q for s in ["ban kiem soat", "kiem soat vien", "hoi dong quan tri", "quan tri cong ty", "thanh vien hop danh", "phan von gop", "co dong"]
    ):
        return True
    if "trong tai lao dong" in title_head and "trong tai" in q and "lao dong" not in q:
        return True
    if "hang khong" in title_head and not any(s in q for s in ["hang khong", "may bay", "tau bay", "van chuyen hang khong"]):
        return True
    if (
        "bao ve quyen loi nguoi tieu dung" in title_head
        and "xu phat vi pham hanh chinh" in title_head
        and is_penalty_question(question)
    ):
        return False
    if "bao ve quyen loi nguoi tieu dung" in title_head and not any(
        s in q for s in [
            "nguoi tieu dung", "quyen loi nguoi tieu dung", "khach hang", "khach hang ca nhan",
            "tranh chap tieu dung", "hang hoa khuyet tat", "hop dong theo mau", "dieu kien giao dich chung",
        ]
    ):
        return True
    if any(s in title_head for s in ["thue thu nhap doanh nghiep", "thue tndn"]) and "thue" not in q:
        return True
    if "ke toan" in q and "thue thu nhap doanh nghiep" in title_head and "thue" not in q:
        return True
    if any(s in title_head for s in ["hai quan", "xuat khau", "nhap khau"]) and not any(s in q for s in ["hai quan", "xuat khau", "nhap khau"]):
        return True
    if any(s in title_head for s in ["ngan hang", "to chuc tin dung", "nhnn"]) and not any(s in q for s in ["ngan hang", "tin dung", "nhnn"]):
        return True
    if any(s in title_head for s in ["cong an nhan dan", "bo quoc phong", "quan doi"]) and not any(s in q for s in ["cong an", "quoc phong", "quan doi"]):
        return True
    if any(s in title_head for s in ["von nha nuoc", "dau tu von nha nuoc"]) and not any(s in q for s in ["von nha nuoc", "nha nuoc dau tu", "doanh nghiep nha nuoc"]):
        return True
    return False


def direct_evidence(question: str, c: dict) -> bool:
    """True when heading/title has enough direct lexical evidence for the query.

    This prevents metadata/current-version boosts from selecting a candidate
    whose reranker score is almost zero and whose heading only belongs to a
    generally current but irrelevant document.
    """
    q = norm(question)
    h = norm(f"{c.get('title', '')} {head_norm(c)}")
    pairs = [
        ("khuyen mai", "khuyen mai"),
        ("quang cao", "quang cao"),
        ("xuc tien", "xuc tien"),
        ("trong tai", "trong tai"),
        ("khieu nai", "khieu nai"),
        ("hai quan", "hai quan"),
        ("tam dung", "tam dung"),
        ("xuat khau", "xuat khau"),
        ("nhap khau", "nhap khau"),
        ("nhan hieu", "nhan hieu"),
        ("sang che", "sang che"),
        ("kieu dang", "kieu dang"),
        ("quyen tac gia", "quyen tac gia"),
        ("lao dong", "lao dong"),
        ("ky luat", "ky luat"),
        ("dinh cong", "dinh cong"),
        ("dang ky doanh nghiep", "dang ky doanh nghiep"),
        ("dai dien", "dai dien"),
        ("uy quyen", "uy quyen"),
        ("hoan thue", "hoan thue"),
        ("gia han", "gia han"),
        ("phat vi pham", "phat vi pham"),
        ("boi thuong", "boi thuong"),
        ("huy bo", "huy bo"),
        ("cham dut", "cham dut"),
        ("nhuong quyen", "nhuong quyen"),
        ("nhan quyen", "nhan quyen"),
        ("hop dong theo mau", "hop dong theo mau"),
        ("dieu kien giao dich chung", "dieu kien giao dich chung"),
        ("van ban noi bo", "dieu le"),
    ]
    return any(qp in q and hp in h for qp, hp in pairs)


def civil_general_candidate(c: dict) -> bool:
    if c.get("prefix") != "91/2015":
        return False
    h = head_norm(c)
    return any(
        s in h
        for s in [
            "dai dien",
            "nang luc hanh vi dan su",
            "nang luc phap luat dan su",
            "giao dich dan su",
            "thoi hieu",
            "vo hieu",
        ]
    )


def asks_civil_general(question: str) -> bool:
    q = norm(question)
    return any(
        s in q
        for s in [
            "nang luc hanh vi",
            "nang luc phap luat",
            "giao dich dan su",
            "vo hieu",
            "thoi hieu",
            "dai dien dan su",
            "uy quyen dan su",
            "nguoi dai dien theo phap luat cua ca nhan",
        ]
    )


def has_specialized_direct(selected: list[dict]) -> bool:
    for c in selected:
        p = c.get("prefix", "")
        if p and p != "91/2015":
            return True
    return False


def base_score(c: dict) -> float:
    score = float(c.get("rr") or 0.0)
    p = c.get("prefix", "")
    if p in OBSOLETE_PREFIXES and not still_current_old_candidate(c):
        score -= 0.50
    yr = year_of_prefix(p)
    if yr and yr < 2015 and not still_current_old_candidate(c):
        score -= 0.18
    score += CURRENT_BOOSTS.get(p, 0.0)
    return score


def candidate_from_hit(hit: dict, source: str, source_rank: int) -> dict | None:
    c = cand_from_hit(hit)
    if not c:
        return None
    payload = hit.get("payload", {}) or {}
    c["text"] = str(payload.get("text") or "")
    c["title"] = str(payload.get("title") or "")
    c["prefix"] = prefix(c["art"])
    c["source"] = source
    c["source_rank"] = source_rank
    c["score_raw"] = float(c.get("rr") or 0.0)
    c["score_arch"] = base_score(c) + (0.04 if source == "orig" else 0.0)
    return c


def safe_subquery(question: str, subquery: str) -> bool:
    qt = content_tokens(question)
    st = content_tokens(subquery)
    if len(st) < 3:
        return False
    overlap = len(qt & st) / max(len(st), 1)
    return overlap >= 0.45


def retrieve_pool(rag, question: str, subqueries: list[str], orig_topk: int, sub_topk: int) -> list[dict]:
    pool: dict[str, dict] = {}
    sources = [("orig", question, orig_topk)]
    if is_complex(question):
        for idx, sub in enumerate(subqueries, 1):
            if safe_subquery(question, sub):
                sources.append((f"sub{idx}", sub, sub_topk))
    for source, query, topk in sources:
        for rank, hit in enumerate(rag.retrieve(query, top_k=topk), 1):
            c = candidate_from_hit(hit, source, rank)
            if not c:
                continue
            key = c["_key"]
            old = pool.get(key)
            if old is None:
                pool[key] = c
            else:
                old["score_arch"] = max(old["score_arch"], c["score_arch"] + 0.025)
                old["score_raw"] = max(old["score_raw"], c["score_raw"])
                old["source"] = old["source"] + "," + source
    out = list(pool.values())
    prefixes = {c["prefix"] for c in out}
    for c in out:
        succs = SUCCESSORS.get(c["prefix"], [])
        if succs and any(s in prefixes for s in succs):
            c["score_arch"] -= 0.18
    out.sort(key=lambda x: (x["score_arch"], x["score_raw"]), reverse=True)
    return out


def covers_clause(c: dict, clause: str) -> float:
    ct = content_tokens(clause)
    if not ct:
        return 0.0
    tt = content_tokens(f"{c.get('art', '')} {c.get('title', '')} {c.get('text', '')[:1600]}")
    return len(ct & tt) / max(len(ct), 1)


def strong_clause_evidence(question: str, c: dict) -> bool:
    if float(c.get("score_raw") or 0.0) < 0.75:
        return False
    if not acceptable(question, c):
        return False
    return any(covers_clause(c, clause) >= 0.55 for clause in split_clauses(question))


def supported_llm_add(c: dict) -> bool:
    raw = float(c.get("score_raw") or 0.0)
    arch = float(c.get("score_arch") or 0.0)
    if raw < 0.015:
        return False
    if raw < 0.05 and arch < 0.10:
        return False
    return True


CLAUSE_GROUPS = [
    ("procedure", ["thu tuc", "trinh tu", "ho so", "thoi han", "nop don", "dang ky", "cap", "gia han", "thong bao"]),
    ("duty", ["nghia vu", "trach nhiem", "phai", "bao dam", "cung cap"]),
    ("right", ["quyen", "duoc", "yeu cau", "khieu nai", "kien", "phan doi"]),
    ("sanction", ["xu phat", "muc phat", "phat tien", "bien phap khac phuc", "khac phuc hau qua"]),
    ("contract_remedy", ["phat vi pham", "boi thuong", "huy bo", "cham dut", "buoc thuc hien", "giao hang"]),
    ("labor", ["ky luat", "sa thai", "dinh cong", "tam hoan", "hop dong lao dong", "tien luong"]),
    ("ip", ["nhan hieu", "sang che", "kieu dang", "van bang bao ho", "xam pham", "huy bo hieu luc"]),
    ("customs", ["hai quan", "xuat khau", "nhap khau", "tam dung", "xuat xu"]),
    ("tax", ["thue", "hoan thue", "khai thue", "nop thue", "tien cham nop"]),
    ("arbitration", ["trong tai", "hoi dong trong tai", "thoa thuan trong tai", "don kien lai"]),
]


def clause_groups(text: str) -> set[str]:
    t = norm(text)
    return {name for name, phrases in CLAUSE_GROUPS if any(p in t for p in phrases)}


def selected_text(selected: list[dict]) -> str:
    return " ".join(f"{c.get('art','')} {c.get('title','')} {head_norm(c)}" for c in selected)


def clause_evidence_score(question: str, selected: list[dict], c: dict) -> tuple[bool, float]:
    q_groups = clause_groups(question)
    if not q_groups:
        return False, 0.0
    covered = clause_groups(selected_text(selected))
    cand_text = f"{c.get('art','')} {c.get('title','')} {head_norm(c)} {c.get('text','')[:800]}"
    cand_groups = clause_groups(cand_text)
    uncovered = (q_groups - covered) & cand_groups
    if not uncovered:
        return False, 0.0
    raw = float(c.get("score_raw") or 0.0)
    arch = float(c.get("score_arch") or 0.0)
    cov = covers_clause(c, question)
    same_doc = any(doc_part(c["art"]) == doc_part(s["art"]) for s in selected)
    if same_doc:
        ok = raw >= 0.06 and (cov >= 0.20 or raw >= 0.18)
    else:
        ok = raw >= 0.24 and (cov >= 0.24 or raw >= 0.45)
    if not ok:
        return False, 0.0
    return True, raw + 0.25 * arch + 0.18 * cov + 0.12 * len(uncovered) + (0.05 if same_doc else 0.0)


def add_clause_evidence(question: str, selected: list[dict], pool: list[dict], max_k: int) -> list[dict]:
    if len(selected) >= max_k:
        return selected
    out = list(selected)
    seen = {c["_key"] for c in out}
    scored: list[tuple[float, dict]] = []
    selected_docs = {doc_part(c["art"]) for c in out}
    for c in pool[:18]:
        if c["_key"] in seen:
            continue
        if not acceptable(question, c):
            continue
        if civil_general_candidate(c) and has_specialized_direct(out) and not asks_civil_general(question):
            continue
        same_doc = doc_part(c["art"]) in selected_docs
        # Clause evidence is an article-recall hook. Scoring v82 showed that
        # letting it open new documents improves article recall but costs too
        # much document precision, so document discovery stays in the main
        # retrieval/selector path.
        if not same_doc:
            continue
        ok, score = clause_evidence_score(question, out, c)
        if ok:
            scored.append((score + (0.08 if same_doc else -0.12), c))
    scored.sort(key=lambda x: x[0], reverse=True)
    for _, c in scored:
        if len(out) >= max_k:
            break
        if c["_key"] in seen:
            continue
        if civil_general_candidate(c) and has_specialized_direct(out) and not asks_civil_general(question):
            continue
        same_doc = doc_part(c["art"]) in {doc_part(s["art"]) for s in out}
        if not same_doc:
            continue
        ok, _score = clause_evidence_score(question, out, c)
        if not ok:
            continue
        out.append(c)
        seen.add(c["_key"])
    return out


def filter_general_civil_noise(question: str, selected: list[dict]) -> list[dict]:
    if asks_civil_general(question) or not has_specialized_direct(selected):
        return selected
    out = [c for c in selected if not civil_general_candidate(c)]
    return out or selected[:1]


def acceptable(question: str, c: dict) -> bool:
    return acceptable_reason(question, c) == "ok"


def acceptable_reason(question: str, c: dict) -> str:
    if off_topic(question, c):
        return "off_topic"
    q = norm(question)
    h = head_norm(c)
    raw = float(c.get("score_raw") or 0.0)
    # Do not let metadata/current-version boost rescue a candidate that the
    # reranker essentially did not believe in, unless the article heading has
    # direct evidence for the question.
    if raw < 0.015 and not direct_evidence(question, c):
        return "raw_too_low_without_direct_evidence"
    if h.startswith("xu ly doi voi viec cham nop tien thue") and not any(
        s in q for s in ["cham nop thue", "cham nop tien thue", "tien cham nop"]
    ):
        return "late_tax_payment_head_mismatch"
    if penalty_doc_candidate(c) and not is_penalty_question(question):
        return "penalty_doc_for_non_penalty_question"
    if generic_candidate(question, c):
        return "generic_candidate"
    if c["prefix"] in HARD_OBSOLETE_PREFIXES:
        return "hard_obsolete_prefix"
    if c["prefix"] in OBSOLETE_PREFIXES and not still_current_old_candidate(c) and raw < 0.95:
        return "obsolete_prefix_low_raw"
    yr = year_of_prefix(c.get("prefix", ""))
    if yr and yr < 2015 and not still_current_old_candidate(c) and raw < 0.75:
        return "old_non_current_low_raw"
    return "ok"

def max_candidates(question: str) -> int:
    if is_complex(question):
        return 5
    if is_penalty_question(question) or is_procedure_question(question):
        return 4
    return 3


def selection_uncertain(question: str, selected: list[dict], pool: list[dict], old_arts: list[str]) -> bool:
    if not selected:
        return True
    q = norm(question)
    selected_docs = {doc_part(c["art"]) for c in selected}
    old_docs = {doc_part(a) for a in old_arts}
    if any(float(c.get("score_raw") or 0.0) < 0.05 for c in selected):
        return True
    if len(selected) <= 1 and (is_complex(question) or len(split_clauses(question)) >= 2):
        return True
    if len(old_docs - selected_docs) >= 2:
        return True
    if any("bao ve quyen loi nguoi tieu dung" in norm(d) for d in old_docs - selected_docs):
        return True
    if any(s in q for s in ["nhan hieu", "kieu dang", "sang che", "so huu cong nghiep"]) and any(
        p in doc_part(c["art"]) for c in pool[:8] for p in ["23/2023", "65/2023", "103/2006"]
    ):
        return True
    if any(s in q for s in ["hop dong theo mau", "dieu kien giao dich chung", "khach hang"]) and any(
        c["prefix"] == "19/2023" for c in pool[:10]
    ):
        return True
    if any(s in q for s in ["nhuong quyen", "nhan quyen"]) and any(c["prefix"] == "35/2006" for c in pool[:10]):
        return True
    if len(pool) >= 2 and (float(pool[0].get("score_arch") or 0.0) - float(pool[1].get("score_arch") or 0.0)) < 0.08:
        return True
    return False


def select_candidates(question: str, pool: list[dict]) -> list[dict]:
    if not pool:
        return []
    complex_q = is_complex(question)
    max_k = max_candidates(question)
    primary = [c for c in pool if acceptable(question, c)]
    if not primary:
        primary = pool[:1]
    top = primary[0]["score_arch"]
    ratio = 0.72 if complex_q else 0.80
    abs_cut = 0.34 if complex_q else 0.42
    selected: list[dict] = []
    seen: set[str] = set()
    for c in primary:
        if len(selected) >= max_k:
            break
        if c["score_arch"] < max(abs_cut, ratio * top):
            continue
        if c["_key"] in seen:
            continue
        selected.append(c)
        seen.add(c["_key"])
    if not selected:
        selected = [primary[0]]
        seen = {primary[0]["_key"]}

    if complex_q and len(selected) < max_k:
        clauses = split_clauses(question)
        for clause in clauses:
            if len(selected) >= max_k:
                break
            if any(covers_clause(c, clause) >= 0.34 for c in selected):
                continue
            scored = []
            clause_norm = norm(clause)
            for c in primary[:16]:
                if c["_key"] in seen:
                    continue
                cov = covers_clause(c, clause)
                head = head_norm(c)
                weak_but_direct = (
                    c["score_raw"] >= 0.08
                    and c["score_arch"] >= 0.04
                    and (
                        (cov >= 0.25 and "quyen" in clause_norm and "quyen" in head)
                        or (cov >= 0.25 and "trach nhiem" in clause_norm and "trach nhiem" in head)
                        or (cov >= 0.25 and "hoan" in clause_norm and "hoan" in head)
                        or (cov >= 0.25 and "ho so" in clause_norm and "ho so" in head)
                        or (cov >= 0.25 and "thu tuc" in clause_norm and "thu tuc" in head)
                        or (cov >= 0.25 and "gia han" in clause_norm and "gia han" in head)
                        or ("doan kiem tra" in head and "kiem tra ke toan" in norm(question))
                        or ("dai dien theo uy quyen" in head and "dai dien" in norm(question) and "uy quyen" in norm(question))
                    )
                )
                if c["score_raw"] < 0.25 or c["score_arch"] < 0.25:
                    if not (cov >= 0.55 and c["score_raw"] >= 0.08 and c["score_arch"] >= 0.05) and not weak_but_direct:
                        continue
                if cov >= 0.42 or weak_but_direct:
                    scored.append((cov, c["score_arch"], c))
            if scored:
                scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
                cand = scored[0][2]
                selected.append(cand)
                seen.add(cand["_key"])

    if is_penalty_question(question) and len(selected) < max_k:
        for c in primary:
            if c["_key"] in seen or c["score_raw"] < 0.25:
                continue
            if c["art"].split("|")[0].strip() in PENALTY_WHITELIST:
                selected.append(c)
                seen.add(c["_key"])
                break
    selected = add_clause_evidence(question, selected, primary, max_k)
    selected = add_topic_complements(question, selected, primary, max_k)
    selected = filter_general_civil_noise(question, selected)
    return selected


def extract_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object in LLM response")
    return json.loads(m.group(0))


def ollama_generate(prompt: str, model: str, timeout: int = 180) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": "/no_think\n" + prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 8192},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data.get("response") or "")


def llm_verify_selection(
    question: str,
    selected: list[dict],
    pool: list[dict],
    model: str,
    llm=None,
    apply: bool = False,
) -> tuple[list[dict], dict]:
    max_k = max_candidates(question)
    candidates: list[dict] = []
    seen: set[str] = set()
    for c in selected + pool[:14]:
        if c["_key"] in seen:
            continue
        candidates.append(c)
        seen.add(c["_key"])
        if len(candidates) >= 14:
            break
    label_to_c = {f"C{i + 1}": c for i, c in enumerate(candidates)}
    cand_text = "\n".join(
        f"C{i + 1}. {c['art']} | score={c['score_raw']:.3f}/{c['score_arch']:.3f} | head={head_norm(c)} | snippet={re.sub(r'\\s+', ' ', c.get('text', ''))[:420]}"
        for i, c in enumerate(candidates)
    )
    prompt = LLM_SELECTOR_PROMPT.format(question=question, candidates=cand_text)
    meta: dict = {"model": model, "used": True}
    try:
        if llm is not None:
            response = llm.complete(
                "Bạn là bộ chọn căn cứ pháp lý. Sau khi suy nghĩ, chỉ trả JSON cuối cùng.",
                prompt,
                think=True,
            )
        else:
            response = ollama_generate(prompt, model)
        obj = extract_json(response)
    except (ValueError, urllib.error.URLError, TimeoutError, Exception) as exc:
        meta["error"] = str(exc)
        if "response" in locals():
            meta["raw"] = response[:500]
        return selected, meta

    keep = {str(x).strip() for x in obj.get("keep", []) if str(x).strip() in label_to_c}
    add = {str(x).strip() for x in obj.get("add", []) if str(x).strip() in label_to_c}
    drop = {str(x).strip() for x in obj.get("drop", []) if str(x).strip() in label_to_c}
    selected_keys = {c["_key"] for c in selected}
    out: list[dict] = []
    out_seen: set[str] = set()

    for label, c in label_to_c.items():
        if c["_key"] not in selected_keys:
            continue
        raw = float(c.get("score_raw") or 0.0)
        high_conf_direct = (raw >= 0.75 and (direct_evidence(question, c) or strong_clause_evidence(question, c))) or (
            raw >= 0.05 and direct_evidence(question, c)
        )
        if label in drop and len(selected) > 1 and not high_conf_direct:
            continue
        out.append(c)
        out_seen.add(c["_key"])

    for label in sorted(keep | add):
        if len(out) >= max_k:
            break
        c = label_to_c[label]
        if c["_key"] in out_seen:
            continue
        if not acceptable(question, c):
            continue
        if c["_key"] not in selected_keys and not supported_llm_add(c):
            continue
        out.append(c)
        out_seen.add(c["_key"])

    if not out:
        out = selected
    meta.update(
        {
            "keep": sorted(keep),
            "add": sorted(add),
            "drop": sorted(drop),
            "missing": obj.get("missing", []),
            "reason": obj.get("reason", ""),
            "applied": apply,
        }
    )
    if not apply:
        return selected, meta
    return out, meta


def add_topic_complements(question: str, selected: list[dict], pool: list[dict], max_k: int) -> list[dict]:
    q = norm(question)
    out = list(selected)
    seen = {c["_key"] for c in out}

    def add_match(prefixes: set[str], head_needles: list[str], min_raw: float = 0.0) -> None:
        nonlocal out, seen
        if len(out) >= max_k:
            return
        for c in pool:
            if c["_key"] in seen or c.get("score_raw", 0.0) < min_raw:
                continue
            if prefixes and c["prefix"] not in prefixes:
                continue
            h = head_norm(c)
            if all(n in h for n in head_needles):
                out.append(c)
                seen.add(c["_key"])
                return

    if "kiem tra ke toan" in q and any(s in q for s in ["quyen", "trach nhiem", "doan kiem tra"]):
        add_match({"88/2015"}, ["quyen", "trach nhiem", "doan kiem tra"], 0.05)

    if "khieu nai" in q and "thue" in q and any(s in q for s in ["thu sai", "hoan tra", "hoan thue"]):
        add_match({"108/2025"}, ["hoan thue"], 0.02)

    if "giai the" in q and "no thue" in q:
        add_match({"38/2019"}, ["hoan thanh", "nghia vu", "nop thue"], 0.10)

    if "hoan thue" in q and any(s in q for s in ["cham", "cho xu ly", "quyet dinh", "quyen loi"]):
        add_match({"38/2019"}, ["thoi han", "hoan thue"], 0.05)
        add_match({"80/2021"}, ["trach nhiem", "xu ly", "ho so", "hoan thue"], 0.05)

    if "gia han" in q and "nop" in q and "thue" in q:
        add_match({"80/2021"}, ["ho so", "gia han", "nop thue"], 0.05)
        add_match({"38/2019"}, ["ho so", "gia han", "nop thue"], 0.05)
        if any(s in q for s in ["dac biet", "suy thoai", "hoa hoan", "kho khan"]):
            add_match({"126/2020"}, ["gia han", "nop thue", "dac biet"], 0.05)

    if "dai dien" in q and "uy quyen" in q:
        add_match({"59/2020"}, ["dai dien", "uy quyen"], 0.05)
        if any(s in q for s in ["thong bao", "dang ky"]):
            add_match({"168/2025"}, ["thong bao", "thay doi"], 0.02)

    if "khoi nghiep sang tao" in q and any(s in q for s in ["tai lieu", "lua chon", "ho tro"]):
        add_match({"06/2022"}, ["lua chon", "khoi nghiep sang tao"], 0.05)
        add_match({"80/2021"}, ["noi dung", "ho tro", "khoi nghiep sang tao"], 0.05)

    return out


def apply_filters(question: str, arts: list[str]) -> list[str]:
    if not arts:
        return arts
    arts = dedupe_articles(arts)
    if any(prefix(art) == "125/2020" and article_no(art) == "Điều 16" for art in arts):
        redundant = {("38/2019", "Điều 142"), ("38/2019", "Điều 138"), ("38/2019", "Điều 136")}
        arts = [art for art in arts if (prefix(art), article_no(art)) not in redundant]
    out: list[str] = []
    dropped_old = 0
    for art in arts:
        sk = norm_sk(art.split("|")[0])
        art_candidate = {"art": art, "title": art, "prefix": prefix(art)}
        pfx = prefix(art)
        protected_old = pfx in {"16/2012", "35/2006", "36/2005", "50/2005", "54/2010", "54/2014", "88/2015", "91/2015"}
        if sk in KNOWN_OLD and not protected_old and not still_current_old_candidate(art_candidate) and (len(arts) - dropped_old) > 1:
            dropped_old += 1
            continue
        out.append(art)
    return out or arts[:1]


def dedupe_articles(arts: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[tuple[str, str]] = set()
    for art in arts:
        key = (prefix(art), article_no(art))
        if key in seen:
            continue
        seen.add(key)
        out.append(art)
    return out


def pack_zip(src: Path) -> Path:
    rows = json.loads(src.read_text(encoding="utf-8"))
    dst = src.with_suffix(".zip")
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.json", json.dumps(rows, ensure_ascii=False))
    with zipfile.ZipFile(dst) as zf:
        assert zf.namelist() == ["results.json"]
    return dst


def parse_ids(raw: str) -> list[int]:
    out: set[int] = set()
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = [int(x) for x in part.split("-", 1)]
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)


def checkpoint(out_path: Path, rows_by_id: dict[int, dict], order: list[int], sidecar_path: Path, sidecar: dict) -> None:
    rows = [rows_by_id[i] for i in order]
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/submission_v50_full.json")
    ap.add_argument("--qfile", default=str(QFILE))
    ap.add_argument("--subq", default="data/subqueries.json")
    ap.add_argument("--lo", type=int, default=1001)
    ap.add_argument("--hi", type=int, default=2000)
    ap.add_argument("--ids", default="", help="Optional comma/range ids, e.g. 1001,1053,1795")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--orig-topk", type=int, default=14)
    ap.add_argument("--sub-topk", type=int, default=8)
    ap.add_argument("--out", default="data/submission_v66_arch_pipeline.json")
    ap.add_argument("--sidecar", default="data/v66_arch_pipeline_sidecar.json")
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--llm-verify", action="store_true", help="Ask Qwen to review keep/add/drop over retrieved candidates")
    ap.add_argument("--llm-apply", action="store_true", help="Apply Qwen verifier changes; default only records diagnostics")
    ap.add_argument("--llm-uncertain-only", action="store_true", help="Run Qwen verifier only when deterministic selection has risk signals")
    ap.add_argument("--llm-model", default="qwen-vbpl")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    base_rows = json.loads((ROOT / args.base).read_text(encoding="utf-8"))
    order = [int(r["id"]) for r in base_rows]
    rows_by_id = {int(r["id"]): dict(r) for r in base_rows}
    questions = {int(r["id"]): r["question"] for r in json.loads(Path(args.qfile).read_text(encoding="utf-8"))}
    subq = json.loads((ROOT / args.subq).read_text(encoding="utf-8"))
    if args.ids:
        wanted = set(parse_ids(args.ids))
        target_ids = [i for i in order if i in wanted]
    else:
        target_ids = [i for i in order if args.lo <= i <= args.hi]
    if args.limit:
        target_ids = target_ids[: args.limit]

    from backend.rag import RAGPipeline

    rag = RAGPipeline()
    llm_client = None
    if args.llm_verify:
        from backend.llm import LLMClient

        llm_client = LLMClient()
    out_path = ROOT / args.out
    sidecar_path = ROOT / args.sidecar
    sidecar: dict[str, dict] = {}
    changed = 0
    t0 = time.time()
    print(f"v66 target={len(target_ids)} ids={args.lo}-{args.hi}", flush=True)

    for idx, qid in enumerate(target_ids, 1):
        question = questions[qid]
        old_row = rows_by_id[qid]
        old_arts = list(old_row.get("relevant_articles") or [])
        pool = retrieve_pool(rag, question, subq.get(str(qid)) or [], args.orig_topk, args.sub_topk)
        chosen = select_candidates(question, pool)
        chosen_before_llm = [c["art"] for c in chosen]
        llm_meta = None
        should_verify = args.llm_verify and (
            not args.llm_uncertain_only or selection_uncertain(question, chosen, pool, old_arts)
        )
        if should_verify:
            chosen, llm_meta = llm_verify_selection(
                question,
                chosen,
                pool,
                args.llm_model,
                llm_client,
                apply=args.llm_apply,
            )
        elif args.llm_verify:
            llm_meta = {"used": False, "reason": "selection_not_uncertain"}
        chosen_before_filter = [c["art"] for c in chosen]
        arts = apply_filters(question, [c["art"] for c in chosen])
        if not arts:
            arts = old_arts[:1]
        row = dict(old_row)
        row["relevant_articles"] = arts
        row["relevant_docs"] = rebuild_docs(arts)
        rows_by_id[qid] = row
        if arts != old_arts:
            changed += 1
        sidecar[str(qid)] = {
            "question": question,
            "old_articles": old_arts,
            "chosen_before_llm": chosen_before_llm,
            "chosen_before_filter": chosen_before_filter,
            "new_articles": arts,
            "pool": [
                {
                    "art": c["art"],
                    "prefix": c.get("prefix", ""),
                    "score_raw": round(c["score_raw"], 4),
                    "score_arch": round(c["score_arch"], 4),
                    "source": c["source"],
                    "acceptable": acceptable(question, c),
                    "acceptable_reason": acceptable_reason(question, c),
                    "off_topic": off_topic(question, c),
                    "still_current_old": still_current_old_candidate(c),
                    "head": head_norm(c)[:120],
                    "snippet": re.sub(r"\s+", " ", c.get("text", ""))[:240],
                }
                for c in pool[:12]
            ],
        }
        if llm_meta is not None:
            sidecar[str(qid)]["llm_verify"] = llm_meta
        print(f"{idx}/{len(target_ids)} id={qid} old={len(old_arts)} new={len(arts)} pool={len(pool)}", flush=True)
        if idx % args.checkpoint_every == 0:
            checkpoint(out_path, rows_by_id, order, sidecar_path, sidecar)
            rate = idx / max(time.time() - t0, 1e-6)
            print(f"[ckpt] {idx}/{len(target_ids)} changed={changed} rate={rate:.3f}q/s eta={(len(target_ids)-idx)/max(rate,1e-6)/60:.1f}m", flush=True)
            gc.collect()

    checkpoint(out_path, rows_by_id, order, sidecar_path, sidecar)
    rows = [rows_by_id[i] for i in order]
    avg = statistics.mean(len(r["relevant_articles"]) for r in rows)
    print(f"DONE changed={changed}/{len(target_ids)} avg_articles={avg:.3f} -> {out_path}", flush=True)
    if args.zip:
        print(f"ZIP -> {pack_zip(out_path)}", flush=True)


if __name__ == "__main__":
    main()
