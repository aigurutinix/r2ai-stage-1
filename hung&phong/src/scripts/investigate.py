"""Điều tra sâu: truy hồi đủ chưa? sai do THIẾU DATA hay do THUẬT TOÁN?

Dùng dữ liệu có sẵn (không chạy lại retrieval):
  - C:/Users/PHONG/Downloads/R2AIStage1DATA.json   (2000 câu hỏi gốc)
  - data/submission_v2.json                         (answer của chatbot)
  - data/submission_v4_scored.json                  (top-15 ứng viên + rr score)
  - data/corpus_vbpl_v2/documents.parquet           (corpus mình có)

Xuất data/_investigate.md để đọc + phán từng ca.
"""
from __future__ import annotations
import json, re, sys, statistics
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "_investigate.md"

QFILE = Path("C:/Users/PHONG/Downloads/R2AIStage1DATA.json")
qs = json.loads(QFILE.read_text(encoding="utf-8"))
v2 = {r["id"]: r for r in json.loads((ROOT / "data/submission_v2.json").read_text(encoding="utf-8"))}
scored = {r["id"]: r for r in json.loads((ROOT / "data/submission_v4_scored.json").read_text(encoding="utf-8"))}
docs = pd.read_parquet(ROOT / "data/corpus_vbpl_v2/documents.parquet")

L = []  # dòng báo cáo
def w(s=""): L.append(s)

# ---- corpus index ----
cols = list(docs.columns)
def col(*names):
    for n in names:
        if n in cols: return n
    return None
c_sk = col("so_ky_hieu", "law_id", "doc_id")
c_title = col("title", "ten_van_ban", "tieu_de")
c_type = col("loai_van_ban", "legal_type")
c_year = col("nam", "year", "nam_ban_hanh")
corpus_sk = set(str(x).strip() for x in docs[c_sk].dropna()) if c_sk else set()
corpus_titles = [str(x).lower() for x in docs[c_title].dropna()] if c_title else []
corpus_title_blob = "\n".join(corpus_titles)

w(f"# ĐIỀU TRA TRUY HỒI — {len(qs)} câu hỏi thi\n")
w(f"corpus documents.parquet: {len(docs)} docs | cols={cols}")
w(f"  so_ky_hieu duy nhất: {len(corpus_sk)} | title: {len(corpus_titles)}\n")

# ===================== §1. PHÂN BỐ ĐỘ TỰ TIN TRUY HỒI =====================
w("## §1. Phân bố độ tự tin reranker (top-1 mỗi câu)\n")
top1, n_cand_strong = [], []
for q in qs:
    sc = scored.get(q["id"])
    cands = sc["candidates"] if sc else []
    if cands:
        rrs = [c["rr"] for c in cands]
        top1.append(max(rrs))
        n_cand_strong.append(sum(1 for r in rrs if r >= 0.5))
    else:
        top1.append(0.0); n_cand_strong.append(0)
buckets = [("<0.2 (gần như KHÔNG có gì liên quan)",0,0.2),("0.2–0.4 (yếu)",0.2,0.4),
           ("0.4–0.6 (TB)",0.4,0.6),("0.6–0.8 (khá)",0.6,0.8),("0.8–1.0 (mạnh)",0.8,1.01)]
for name,lo,hi in buckets:
    n = sum(1 for t in top1 if lo<=t<hi)
    w(f"  top1 {name}: {n} câu ({n/len(qs)*100:.1f}%)")
w(f"\n  → top-1 < 0.4 (NGHI thiếu data / câu khó): {sum(1 for t in top1 if t<0.4)} câu "
  f"({sum(1 for t in top1 if t<0.4)/len(qs)*100:.1f}%)")
w(f"  → TB số ứng viên rr>=0.5 mỗi câu: {statistics.mean(n_cand_strong):.2f}")
w(f"  → câu có 0 ứng viên rr>=0.5: {sum(1 for x in n_cand_strong if x==0)}\n")

# ===================== §2. CHỦ ĐỀ CÂU HỎI =====================
w("## §2. Phân loại chủ đề 2000 câu (keyword)\n")
TOPICS = {
    "Doanh nghiệp/đăng ký": ["doanh nghiệp","cổ phần","tnhh","hợp danh","vốn điều lệ","đăng ký kinh doanh","giải thể","cổ đông","hội đồng quản trị"],
    "Hỗ trợ SME": ["nhỏ và vừa","sme","hỗ trợ doanh nghiệp","khởi nghiệp","ươm tạo"],
    "Thuế": ["thuế","hóa đơn","ấn định thuế","quyết toán","khấu trừ","gtgt","thu nhập doanh nghiệp"],
    "Lao động": ["lao động","thử việc","hợp đồng lao động","tiền lương","nghỉ phép","sa thải","bảo hiểm xã hội","công đoàn"],
    "Đầu tư": ["đầu tư","ưu đãi đầu tư","dự án đầu tư","ppp","đối tác công tư"],
    "Đấu thầu": ["đấu thầu","gói thầu","nhà thầu","lựa chọn nhà thầu"],
    "Hợp đồng/thương mại": ["hợp đồng","thương mại","mua bán","giao dịch","phạt vi phạm"],
    "Kế toán/tài chính": ["kế toán","báo cáo tài chính","kiểm toán","chứng khoán"],
    "Phá sản": ["phá sản","mất khả năng thanh toán"],
    "Sở hữu trí tuệ": ["sở hữu trí tuệ","nhãn hiệu","sáng chế","bản quyền"],
    "Đất đai/xây dựng": ["đất đai","thuê đất","xây dựng","giấy phép xây dựng"],
    "Hải quan/XNK": ["hải quan","xuất khẩu","nhập khẩu","xuất nhập khẩu"],
}
topic_count = Counter()
for q in qs:
    t = q["question"].lower()
    matched = [name for name,kws in TOPICS.items() if any(k in t for k in kws)]
    if not matched: topic_count["[KHÁC/không khớp]"] += 1
    for name in matched: topic_count[name] += 1
for name,n in topic_count.most_common():
    w(f"  {name}: {n}")
w("")

# ===================== §3. CÂU HỎI NÊU TÊN LUẬT → CÓ TRONG CORPUS? CÓ TRUY HỒI ĐƯỢC? =====================
w("## §3. Câu hỏi NÊU ĐÍCH DANH luật → corpus có? truy hồi có?\n")
w("(tín hiệu mạnh nhất, không cần gold: nếu câu hỏi chỉ tên luật mà mình không "
  "truy về điều nào của luật đó → thất bại rõ ràng)\n")
LAW_RE = re.compile(r"(Bộ luật|Luật|Pháp lệnh)\s+([A-ZĐÀ-Ỹ][^,.;:?()\"]{2,55}?)(?=\s*(?:năm\s*\d{4}|số|\?|,|\.|;|:|\)|$|được|quy định|có|thì|nào|gồm|và\s+[A-ZĐ]))",)
def norm(s): return re.sub(r"\s+"," ",s.strip().lower())
named = []
for q in qs:
    for m in LAW_RE.finditer(q["question"]):
        full = norm(f"{m.group(1)} {m.group(2)}")
        named.append((q["id"], full))
        break  # 1 luật chính / câu
law_freq = Counter(f for _,f in named)
w(f"Tổng câu nêu đích danh ≥1 luật: {len(named)} / {len(qs)}\n")
w("### Top luật được hỏi + bao phủ:")
in_corpus_named = miss_corpus_named = miss_retr_named = 0
detail_miss = []
for law, freq in law_freq.most_common(25):
    # có trong corpus? (title chứa tên luật)
    key = law.replace("bộ luật","").replace("luật ","").replace("pháp lệnh ","").strip()
    in_corp = key in corpus_title_blob
    # với các câu hỏi luật này, mình có truy hồi điều nào của nó không?
    ids = [i for i,f in named if f==law]
    retr_ok = 0
    for qid in ids:
        sc = scored.get(qid); cands = sc["candidates"] if sc else []
        blob = " ".join(c["art"].lower() for c in cands)
        if key and key in blob: retr_ok += 1
    flag = "✓corpus" if in_corp else "✗KHÔNG-CÓ-CORPUS"
    w(f"  [{freq:3d}q] {law[:50]:50s} {flag} | truy hồi đúng luật: {retr_ok}/{len(ids)}")
    if not in_corp: detail_miss.append((law,freq))
w("")
if detail_miss:
    w("### ⚠️ Luật ĐƯỢC HỎI nhưng KHÔNG thấy trong corpus title:")
    for law,freq in detail_miss: w(f"  [{freq}q] {law}")
    w("")

# ===================== §4. DUMP CA CỤ THỂ (đọc + phán tay) =====================
w("## §4. Soi ca cụ thể — Q + answer + top truy hồi\n")
def dump_case(q):
    qid = q["id"]
    sc = scored.get(qid); cands = sc["candidates"] if sc else []
    ans = (v2.get(qid,{}).get("answer") or "").replace("\n"," ")
    w(f"### Q{qid}  (top1 rr={max([c['rr'] for c in cands],default=0):.2f})")
    w(f"**Hỏi:** {q['question']}")
    # điều mà answer trích
    cited = sorted(set(re.findall(r"Điều\s+\d+[a-zđ]?", ans)))
    w(f"**Answer trích:** {', '.join(cited) if cited else '(không trích Điều)'}")
    w(f"**Answer:** {ans[:280]}{'...' if len(ans)>280 else ''}")
    w(f"**Top truy hồi (rr | điều):**")
    for c in cands[:6]:
        w(f"  - {c['rr']:.2f}  {c['art']}")
    w("")

# 4a. ca tự tin THẤP (nghi thất bại)
order_low = sorted(qs, key=lambda q: max([c["rr"] for c in scored.get(q["id"],{}).get("candidates",[{"rr":0}])], default=0))
w("### 4A. 12 ca ĐỘ TỰ TIN THẤP NHẤT (nghi thiếu data hoặc truy hồi trật):\n")
for q in order_low[:12]: dump_case(q)
# 4b. ca tự tin CAO (kiểm tra chất lượng đỉnh)
w("### 4B. 8 ca ĐỘ TỰ TIN CAO (kiểm tra đỉnh có đúng không):\n")
for q in order_low[-8:]: dump_case(q)
# 4c. ngẫu nhiên trải đều theo id
w("### 4C. 10 ca rải đều (id 200,400,...,2000):\n")
for qid in range(200,2001,200):
    q = next((x for x in qs if x["id"]==qid), None)
    if q: dump_case(q)

# ===================== §5. CORPUS STATS =====================
w("## §5. Thống kê corpus\n")
if c_type:
    w("Theo loại văn bản:")
    for t,n in Counter(str(x) for x in docs[c_type].dropna()).most_common(): w(f"  {t}: {n}")
if c_year:
    yrs = [int(re.search(r'(19|20)\d{2}',str(x)).group()) for x in docs[c_year].dropna() if re.search(r'(19|20)\d{2}',str(x))]
    if yrs:
        w(f"\nNăm: min={min(yrs)} max={max(yrs)} | >=2015: {sum(1 for y in yrs if y>=2015)} | <2010: {sum(1 for y in yrs if y<2010)}")

OUT.write_text("\n".join(L), encoding="utf-8")
print(f"XONG → {OUT}  ({len(L)} dòng)")
print(f"top1<0.4: {sum(1 for t in top1 if t<0.4)}/{len(qs)} | câu nêu tên luật: {len(named)} | luật thiếu corpus: {len(detail_miss)}")
