"""Lượng hóa 3 vấn đề phát hiện được, để ưu tiên cải tiến."""
from __future__ import annotations
import json, re, sys
from collections import Counter
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
qs = json.loads(Path("C:/Users/PHONG/Downloads/R2AIStage1DATA.json").read_text(encoding="utf-8"))
scored = {r["id"]: r for r in json.loads((ROOT/"data/submission_v4_scored.json").read_text(encoding="utf-8"))}

YEAR = re.compile(r"/(19|20)(\d{2})/")           # số hiệu .../2020/...
def year_of(sk):
    m = YEAR.search(sk or ""); return int(m.group(1)+m.group(2)) if m else None

def family(name):
    """Chuẩn hóa tên luật → họ (bỏ năm, 'sửa đổi bổ sung', 'số')."""
    s = (name or "").lower()
    s = re.sub(r"sửa đổi,? bổ sung.*", "", s)
    s = re.sub(r"số\s.*", "", s)
    s = re.sub(r"\d{4}", "", s)
    s = re.sub(r"\s+", " ", s).strip(" .,")
    return s

PRIMARY = ("luật", "bộ luật", "pháp lệnh")

# ===== VẤN ĐỀ 1: nhiễu bản luật cũ (cùng họ, nhiều số hiệu) trong top-K submit =====
def parse(c):
    sk = c["art"].split("|")[0]
    name = c["art"].split("|")[1] if c["art"].count("|")>=2 else ""
    return sk, name, year_of(sk)

q_has_dup_primary = 0      # câu có ≥2 bản cùng họ luật chính trong top-5
slots_wasted = 0           # tổng slot top-3 bị bản cũ chiếm (có thể giải phóng)
old_in_top3 = 0            # số ứng viên top-3 là bản KHÔNG mới nhất cùng họ
total_top3 = 0
for q in qs:
    cands = scored.get(q["id"], {}).get("candidates", [])[:5]
    fams = {}
    for c in cands:
        sk, name, yr = parse(c)
        fam = family(name)
        is_primary = any(name.lower().startswith(p) for p in PRIMARY)
        fams.setdefault(fam, []).append((sk, yr, is_primary))
    # họ luật chính có >1 số hiệu
    dup_primary = any(len({sk for sk,_,_ in v})>1 and any(p for _,_,p in v) for v in fams.values())
    if dup_primary: q_has_dup_primary += 1
    # trong top-3: đếm bản KHÔNG phải mới nhất của họ nó
    top3 = cands[:3]
    for c in top3:
        sk, name, yr = parse(c); fam = family(name)
        same = [x for x in fams.get(fam, []) if x[2]]  # cùng họ, primary
        if len(same) > 1 and yr is not None:
            newest = max((y for _,y,_ in same if y is not None), default=None)
            if newest is not None and yr < newest:
                old_in_top3 += 1
        total_top3 += 1

print("=== VẤN ĐỀ 1: NHIỄU BẢN LUẬT CŨ ===")
print(f"  Câu có ≥2 bản cùng họ luật-chính trong top-5: {q_has_dup_primary}/{len(qs)} ({q_has_dup_primary/len(qs)*100:.1f}%)")
print(f"  Ứng viên top-3 là bản CŨ (có bản mới hơn cùng họ): {old_in_top3} / {total_top3} slot "
      f"({old_in_top3/total_top3*100:.1f}% slot top-3 lãng phí cho bản cũ)")

# ===== VẤN ĐỀ 2: câu tình huống dài (nhiều vế) =====
scen_markers = ["nếu", "khi", "trường hợp", "thì", "và sau đó", "nhưng", "vậy"]
long_q = sum(1 for q in qs if len(q["question"]) > 160)
multi_clause = sum(1 for q in qs if sum(q["question"].lower().count(m) for m in ["thì","nhưng"," và "])>=2 and len(q["question"])>140)
print("\n=== VẤN ĐỀ 2: CÂU TÌNH HUỐNG DÀI/NHIỀU VẾ ===")
print(f"  Câu dài >160 ký tự: {long_q}/{len(qs)} ({long_q/len(qs)*100:.1f}%)")
print(f"  Câu nhiều vế (>=2 'thì/nhưng/và' + dài): {multi_clause}/{len(qs)}")
# độ tự tin theo độ dài
import statistics
short_top1 = [max([c["rr"] for c in scored.get(q["id"],{}).get("candidates",[{"rr":0}])],default=0) for q in qs if len(q["question"])<=120]
long_top1  = [max([c["rr"] for c in scored.get(q["id"],{}).get("candidates",[{"rr":0}])],default=0) for q in qs if len(q["question"])>200]
print(f"  TB top-1 rr | câu NGẮN(<=120): {statistics.mean(short_top1):.3f} (n={len(short_top1)})")
print(f"  TB top-1 rr | câu DÀI(>200):   {statistics.mean(long_top1):.3f} (n={len(long_top1)})  ← chênh = bằng chứng")

# ===== VẤN ĐỀ 3: thiếu data domain mới (AI/công nghệ số) =====
AI_KW = ["trí tuệ nhân tạo","hệ thống ai"," ai ","công nghiệp công nghệ số","công nghệ số",
         "học máy","thuật toán","dữ liệu lớn","tài sản số","tài sản mã hóa","blockchain","chuyển đổi số","bán dẫn"]
ai_qs = [q for q in qs if any(k in (" "+q["question"].lower()+" ") for k in AI_KW)]
ai_low = [q for q in ai_qs if max([c["rr"] for c in scored.get(q["id"],{}).get("candidates",[{"rr":0}])],default=0) < 0.4]
print("\n=== VẤN ĐỀ 3: DATA GAP DOMAIN MỚI (AI/CÔNG NGHỆ SỐ) ===")
print(f"  Câu hỏi chạm AI/công nghệ số: {len(ai_qs)}/{len(qs)} ({len(ai_qs)/len(qs)*100:.1f}%)")
print(f"  Trong đó top-1 rr<0.4 (gần như KHÔNG có data): {len(ai_low)} câu")
print(f"  → ví dụ id: {[q['id'] for q in ai_low[:15]]}")

# tỉ lệ ứng viên submit (top-3) là văn bản cũ <2015 nói chung
pre2015 = post = 0
for q in qs:
    for c in scored.get(q["id"],{}).get("candidates",[])[:3]:
        y = year_of(c["art"].split("|")[0])
        if y is None: continue
        if y < 2015: pre2015 += 1
        else: post += 1
print("\n=== BỔ SUNG: tuổi văn bản trong top-3 submit ===")
print(f"  <2015: {pre2015} | >=2015: {post} | tỉ lệ cũ: {pre2015/(pre2015+post)*100:.1f}%")
