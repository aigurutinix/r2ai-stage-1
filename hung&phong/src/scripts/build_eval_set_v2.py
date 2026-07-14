"""Dựng eval set mới từ luật HIỆN HÀNH trong vbpl_v2 (gold chắc chắn có trong corpus).
Sample điều thật → qwen sinh câu hỏi → gold=(so_ky_hieu, dieu_so). Temp/tool.
"""
import json, sys
from pathlib import Path
import pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
from backend.llm import LLMClient

ART = Path("data/corpus_vbpl_v2/articles.parquet")
# Luật trọng tâm DN/SME bản hiện hành (đã xác nhận trong vbpl_v2)
TARGETS = {
    "59/2020/QH14": "Luật DN", "04/2017/QH14": "Hỗ trợ DNNVV", "61/2020/QH14": "Đầu tư",
    "45/2019/QH14": "BLLĐ", "38/2019/QH14": "Quản lý thuế", "22/2023/QH15": "Đấu thầu",
    "54/2019/QH14": "Chứng khoán", "88/2015/QH13": "Kế toán", "23/2018/QH14": "Cạnh tranh",
}
PER_LAW = 6
GEN_SYS = ("Bạn là chuyên gia pháp lý VN. Từ nội dung MỘT điều luật, đặt đúng MỘT câu hỏi "
           "tự nhiên người dân/doanh nghiệp hay hỏi mà điều này trả lời được. KHÔNG nhắc số "
           "điều/số hiệu/tên văn bản. Chỉ in câu hỏi.")
GEN_USR = "Nội dung điều luật:\n\"\"\"\n{a}\n\"\"\"\nCâu hỏi:"

# gom điều theo luật
tbl = pq.read_table(ART, columns=["so_ky_hieu", "dieu_so", "dieu_tieu_de", "char_len", "text"])
rows = tbl.to_pylist()
by_law = {}
for r in rows:
    sk = str(r["so_ky_hieu"])
    if sk in TARGETS and (r["dieu_so"] or 0) > 0 and (r["char_len"] or 0) >= 300:
        by_law.setdefault(sk, []).append(r)

llm = LLMClient()
cases = []
cid = 0
for sk, name in TARGETS.items():
    arts = by_law.get(sk, [])
    if not arts:
        print(f"  {sk} ({name}): KHÔNG có điều khả dụng", flush=True)
        continue
    arts.sort(key=lambda x: x["dieu_so"])
    step = max(1, len(arts) // PER_LAW)
    picked = arts[::step][:PER_LAW]
    print(f"  {sk} ({name}): {len(arts)} điều → lấy {len(picked)}", flush=True)
    for a in picked:
        article = f"Điều {a['dieu_so']}. {a['dieu_tieu_de']}\n{a['text']}"[:3000]
        try:
            q = llm.complete(GEN_SYS, GEN_USR.format(a=article)).strip().strip('"')
            q = q.splitlines()[0].strip() if q else q
        except Exception as e:
            print("   gen fail:", e); continue
        if "?" not in q:
            continue
        cid += 1
        cases.append({"id": cid, "question": q, "topic": name,
                      "gold": [{"so_ky_hieu": sk, "dieu_so": int(a["dieu_so"])}]})

Path("data/eval_set_v2.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nXONG: {len(cases)} câu → data/eval_set_v2.json", flush=True)
