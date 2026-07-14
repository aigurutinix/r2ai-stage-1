"""THÍ NGHIỆM v24 — hậu xử lý/augment trên v20 (đã 0.5985), 2 đòn tách bạch:

  --collapse : gộp PHIÊN BẢN (cùng họ → giữ năm mới nhất). RECALL-SAFE: chỉ bỏ bản cũ
               khi bản mới CÓ trong cùng danh sách. Xử lý pattern ② (văn bản cũ).
  --penalty  : với câu có vế "xử phạt/chế tài" → retrieve subquery bơm vế phạt, thêm
               1-2 điều NĐ xử phạt v20 còn thiếu. Xử lý pattern ① (rò recall lớn nhất).

So điểm BTC mới với 0.5985 (v20):
  - chỉ --collapse  → đo riêng pattern ② (nhanh, không cần GPU).
  - --penalty --collapse → đo thêm pattern ① (cần retrieve ~292 câu, ~40 phút).

Chạy (collapse-only, nhanh):
  PYTHONUTF8=1 PYTHONPATH=. python scripts/exp_v24.py --collapse --out data/submission_v24_collapse.json
Chạy (penalty+collapse, cần env GPU):
  QDRANT_COLLECTION=vbpl_aiteam ... USE_RERANKER=true ... \
  python scripts/exp_v24.py --penalty --collapse --out data/submission_v24_penalty.json
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
sys.path.insert(0, str(ROOT))

V20 = ROOT / "data/submission_v20_clean.json"

_PENALTY = re.compile(
    r"xử phạt|bị phạt|mức phạt|phạt tiền|vi phạm hành chính|chế tài|"
    r"khắc phục hậu quả|rủi ro pháp lý|bị xử lý|xử lý vi phạm|biện pháp xử lý",
    re.IGNORECASE,
)
_PEN_SUFFIX = " — mức xử phạt vi phạm hành chính, mức phạt tiền và biện pháp khắc phục hậu quả là gì"
# Ứng viên phải là NĐ XỬ PHẠT (lọc theo TÊN văn bản) — tránh kéo điều luật nội dung.
_PEN_DOC = re.compile(r"xử phạt vi phạm hành chính|xử lý vi phạm hành chính", re.IGNORECASE)

# NĐ XỬ PHẠT HIỆN HÀNH đúng phạm vi thi (DN/lao động/thuế/SHTT/môi trường/kế toán/...).
# CHỈ bơm điều thuộc các NĐ này → vừa chặn bản cũ, vừa chặn sai lĩnh vực.
PENALTY_WHITELIST = {
    "12/2022/NĐ-CP",    # lao động, BHXH, người LĐ VN đi làm việc ở nước ngoài
    "125/2020/NĐ-CP",   # thuế, hóa đơn
    "122/2021/NĐ-CP",   # đầu tư, đăng ký doanh nghiệp, đấu thầu
    "98/2020/NĐ-CP",    # thương mại, hàng giả, hàng cấm, bảo vệ người tiêu dùng
    "99/2013/NĐ-CP",    # sở hữu công nghiệp
    "131/2013/NĐ-CP",   # quyền tác giả, quyền liên quan
    "45/2022/NĐ-CP",    # môi trường
    "41/2018/NĐ-CP",    # kế toán, kiểm toán độc lập
    "75/2019/NĐ-CP",    # cạnh tranh
    "156/2020/NĐ-CP",   # chứng khoán
    "16/2022/NĐ-CP",    # xây dựng
}

_YEAR = re.compile(r"/(19|20)(\d{2})/")
_PRIMARY = ("luật", "bộ luật", "pháp lệnh")


def year_of(sk: str) -> int:
    m = _YEAR.search(sk or "")
    return int(m.group(1) + m.group(2)) if m else 0


# ── Bản đồ HỌ PHIÊN BẢN toàn corpus (do workflow gom cụm + kiểm phản biện, 134 họ/381 VB).
# Map sk → family_key. Dùng TRƯỚC các rule tay → dedup phiên bản chính xác & rộng hơn nhiều.
_WF_FAMILY: dict[str, str] = {}
# LUẬT/Pháp lệnh: KHÔNG dùng map (luật sửa đổi bổ sung KHÔNG thay luật gốc — vd Luật SHTT
# 50/2005 vẫn là bản trích dẫn dù có 07/2022 sửa đổi). Rule "law:" cũ xử lý luật-thay-thế đúng rồi.
_LAW_SK = re.compile(r"/QH\d|/PL[-/]|[-/]CTN|UBTVQH")
try:
    _wf = json.loads((ROOT / "data/version_families.json").read_text(encoding="utf-8"))
    for _f in _wf:
        _members = [str(s).strip() for s in _f.get("members", [])]
        if any(_LAW_SK.search(s) for s in _members):
            continue  # bỏ họ chứa LUẬT → tránh over-collapse base+amendment
        _key = "wf:" + str(_f.get("newest_sk") or _f.get("subject", ""))[:40]
        for _sk in _members:
            _WF_FAMILY[_sk] = _key
except Exception:  # noqa: BLE001
    _WF_FAMILY = {}


def family_tag(sk: str, name: str) -> str | None:
    """Họ văn bản để gộp phiên bản. None = KHÔNG gộp (an toàn recall).
    Ưu tiên BẢN ĐỒ workflow (toàn diện); fallback các rule tay cũ."""
    if sk in _WF_FAMILY:
        return _WF_FAMILY[sk]
    n = (name or "").lower()
    if "xử phạt" in n or "xử lý vi phạm" in n:
        if "lao động" in n or "bảo hiểm xã hội" in n:
            return "xphc:laodong_bhxh"
        if "thuế" in n or "hóa đơn" in n:
            return "xphc:thue_hoadon"
        if "sở hữu" in n:
            return "xphc:shtt"
        if "thương mại" in n or "hàng giả" in n or "hàng hóa" in n:
            return "xphc:thuongmai"
        if "kế toán" in n or "kiểm toán" in n:
            return "xphc:ketoan"
        if "đầu tư" in n or "đăng ký doanh nghiệp" in n or "đăng ký kinh doanh" in n:
            return "xphc:dautu_dkkd"
    # Họ NĐ chi tiết/hướng dẫn hay lẫn bản cũ/mới (successor RÕ RÀNG). Recall-safe:
    # collapse_versions chỉ bỏ bản cũ khi bản mới ĐỒNG hiện diện trong cùng câu.
    if "hỗ trợ doanh nghiệp nhỏ và vừa" in n or "doanh nghiệp vừa và nhỏ" in n:
        return "nd:dnnvv"                              # 39/2018 → 80/2021
    if "quy chế dân chủ" in n or "đối thoại tại nơi làm việc" in n:
        return "nd:danchu"                             # 149/2018 → 145/2020
    # NĐ về ĐĂNG KÝ DOANH NGHIỆP/KINH DOANH (rất hay lẫn bản cũ trong câu hỏi DN):
    # 88/2006, 109/2004, 43/2010, 78/2015 → 01/2021 → 168/2025. KHÔNG trùng "tổ hợp tác,
    # hợp tác xã" (tên khác) hay "đăng ký thuế" (TT) → an toàn. Đã qua nhánh xphc ở trên.
    if "đăng ký doanh nghiệp" in n or "đăng ký kinh doanh" in n:
        return "nd:dkdn"
    # Luật/Bộ luật/Pháp lệnh gốc → gộp theo tên đã bỏ năm/"sửa đổi"
    if any(n.strip().startswith(p) for p in _PRIMARY):
        fam = re.sub(r"sửa đổi,? bổ sung.*", "", n)
        fam = re.sub(r"số\s.*", "", fam)
        fam = re.sub(r"\d{4}", "", fam)
        fam = re.sub(r"\s+", " ", fam).strip(" .,")
        return "law:" + fam
    return None


def parts_of(art: str) -> tuple[str, str]:
    p = art.split("|")
    return p[0].strip(), (p[1].strip() if len(p) >= 2 else "")


def collapse_versions(arts: list[str]) -> list[str]:
    """Giữ năm mới nhất trong mỗi họ (chỉ khi có >=2 bản cùng họ trong list)."""
    newest: dict[str, int] = {}
    for a in arts:
        sk, name = parts_of(a)
        fam = family_tag(sk, name)
        y = year_of(sk)
        if fam and y:
            newest[fam] = max(newest.get(fam, 0), y)
    out = []
    for a in arts:
        sk, name = parts_of(a)
        fam = family_tag(sk, name)
        y = year_of(sk)
        if fam and y and y < newest.get(fam, 0):
            continue  # bản cũ, đã có bản mới cùng họ → bỏ
        out.append(a)
    return out


def art_key(art: str) -> tuple[str, str]:
    p = art.split("|")
    return p[0].strip(), (p[-1].strip() if p else "")


def rebuild_docs(arts: list[str]) -> list[str]:
    docs, seen = [], set()
    for a in arts:
        p = a.split("|")
        doc = f"{p[0].strip()}|{p[1].strip()}" if len(p) >= 2 else p[0].strip()
        if doc not in seen:
            seen.add(doc)
            docs.append(doc)
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--penalty", action="store_true")
    ap.add_argument("--collapse", action="store_true")
    ap.add_argument("--add", type=int, default=2)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = json.loads(V20.read_text(encoding="utf-8"))

    rag = None
    if args.penalty:
        from backend.rag import RAGPipeline
        from tests.build_submission_v12 import hits_to_cands
        rag = RAGPipeline()
        targets = [r for r in data if _PENALTY.search(r.get("question", ""))]
        if args.limit:  # lấy TRẢI ĐỀU để phủ đủ lĩnh vực (không chỉ id thấp = lao động)
            step = max(1, len(targets) // args.limit)
            targets = targets[::step][: args.limit]
        print(f"[penalty] {len(targets)} câu có vế chế tài | thêm tối đa {args.add}/câu", flush=True)
        t0 = time.time()
        n_add = 0
        for i, r in enumerate(targets, 1):
            existing = {art_key(a) for a in r["relevant_articles"]}
            # Retrieve CÂU GỐC (đã có từ khoá lĩnh vực) → lọc chỉ giữ NĐ XỬ PHẠT
            # (đúng lĩnh vực do reranker xếp theo câu gốc) → khôi phục NĐ phạt v20 bỏ sót.
            hits = rag.retrieve(r["question"], top_k=args.topk)
            added = 0
            for c in hits_to_cands(hits):
                if added >= args.add:
                    break
                sk = c["art"].split("|")[0].strip()
                if sk not in PENALTY_WHITELIST:   # chỉ NĐ xử phạt hiện hành, đúng phạm vi
                    continue
                if art_key(c["art"]) in existing:
                    continue
                existing.add(art_key(c["art"]))
                r["relevant_articles"].append(c["art"])
                added += 1
            n_add += added
            if i % 50 == 0:
                rate = i / (time.time() - t0)
                print(f"  [{i}/{len(targets)}] {rate:.2f} q/s ETA {(len(targets)-i)/rate/60:.0f}p · +{n_add} điều", flush=True)
        print(f"[penalty] thêm tổng {n_add} điều ({time.time()-t0:.0f}s)", flush=True)

    n_dropped = 0
    if args.collapse:
        for r in data:
            before = list(r["relevant_articles"])
            r["relevant_articles"] = collapse_versions(before)
            n_dropped += len(before) - len(r["relevant_articles"])
        print(f"[collapse] bỏ {n_dropped} điều bản cũ", flush=True)

    for r in data:
        r["relevant_docs"] = rebuild_docs(r["relevant_articles"])

    import statistics
    avg = statistics.mean(len(r["relevant_articles"]) for r in data)
    Path(ROOT / args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"XONG → {args.out} | TB điều/câu: 2.296 (v20) → {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()
