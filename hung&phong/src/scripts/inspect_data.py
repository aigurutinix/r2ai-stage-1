"""Công cụ TỰ VERIFY parse & chunk — đọc trực tiếp kết quả, không cần tin số liệu.

Nguồn dữ liệu sau parse: data/corpus_vbpl_v2/articles.parquet  (mỗi dòng = 1 Điều).
Chunk = sinh tại chỗ từ articles.parquet qua ingest/chunk.py (giống lúc embed).

DÙNG (PYTHONUTF8=1):
  # Liệt kê các Điều đã parse của 1 văn bản (số, tiêu đề, độ dài, đầu text):
  python scripts/inspect_data.py parse 59/2020/QH14
  # Xem CHUNK (đầy đủ text + vùng đè) của 1 văn bản, hoặc 1 điều:
  python scripts/inspect_data.py chunk 59/2020/QH14
  python scripts/inspect_data.py chunk 59/2020/QH14 4
  # Xuất ra file .txt đọc thoải mái trong Notepad/VSCode:
  python scripts/inspect_data.py export 59/2020/QH14
  # Audit toàn corpus → data/inspect/parse_audit.md + mẫu lỗi:
  python scripts/inspect_data.py audit
  # Bốc ngẫu nhiên N chunk để soi (mặc định 25):
  python scripts/inspect_data.py sample 30
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

from ingest.parse import DieuChunk, ParsedDoc
from ingest.chunk import chunk_document

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus_vbpl_v2" / "articles.parquet"
OUT = ROOT / "data" / "inspect"


def _load() -> pd.DataFrame:
    df = pd.read_parquet(CORPUS)
    df["so_ky_hieu"] = df["so_ky_hieu"].astype(str)
    return df


def _doc_rows(df: pd.DataFrame, sk: str) -> pd.DataFrame:
    m = df[df["so_ky_hieu"] == sk].copy()
    if m.empty:  # thử khớp lỏng
        m = df[df["so_ky_hieu"].str.startswith(sk)].copy()
    return m.sort_values("dieu_so")


def _mk_doc(rows: pd.DataFrame) -> ParsedDoc:
    r0 = rows.iloc[0]
    dieus = [DieuChunk(dieu_so=int(r.dieu_so or 0), dieu_tieu_de=str(r.dieu_tieu_de or ""),
                       text=str(r.text or "")) for r in rows.itertuples(index=False)]
    return ParsedDoc(doc_id=str(r0.so_ky_hieu), so_ky_hieu=str(r0.so_ky_hieu),
                     loai_van_ban=str(r0.loai_van_ban or ""), co_quan_ban_hanh="",
                     ngay_ban_hanh="", ngay_hieu_luc="", tinh_trang_hieu_luc="",
                     linh_vuc="", title=str(r0.title), dieus=dieus)


def cmd_parse(sk: str) -> None:
    df = _load(); rows = _doc_rows(df, sk)
    if rows.empty:
        print(f"Không tìm thấy văn bản '{sk}'"); return
    print(f"VĂN BẢN: {rows.iloc[0]['so_ky_hieu']} | {rows.iloc[0]['title']}")
    print(f"Số Điều parse được: {len(rows)}  ·  dải Điều {int(rows['dieu_so'].min())}–{int(rows['dieu_so'].max())}")
    print("=" * 78)
    nums = list(rows["dieu_so"])
    for r in rows.itertuples(index=False):
        td = str(r.dieu_tieu_de or "")
        flag = "  ⚠TIÊU-ĐỀ-DÀI" if len(td) > 120 else ""
        print(f"Điều {int(r.dieu_so):>3} | tiêu đề({len(td)}): {td[:90]}{flag}")
        print(f"         text({len(str(r.text))}): {str(r.text)[:110]!r}")
    # kiểm tra thiếu Điều
    missing = [n for n in range(min(nums), max(nums) + 1) if n not in set(nums)] if nums else []
    if missing:
        print(f"\n⚠ Điều bị THIẾU trong dải: {missing[:40]}{' …' if len(missing)>40 else ''}")


def cmd_chunk(sk: str, dieu: int | None = None) -> None:
    df = _load(); rows = _doc_rows(df, sk)
    if rows.empty:
        print(f"Không tìm thấy '{sk}'"); return
    if dieu is not None:
        rows = rows[rows["dieu_so"] == dieu]
    chunks = chunk_document(_mk_doc(rows))
    print(f"{sk} → {len(chunks)} chunk")
    print("=" * 78)
    for i, c in enumerate(chunks):
        p = c.payload
        print(f"┌─ chunk {i+1}/{len(chunks)} · Điều {p['dieu_so']} · khoan={p.get('khoan_so')} · {len(c.text)} ký tự")
        print("│ " + c.text.replace("\n", "\n│ "))
        print("└" + "─" * 76)


def cmd_export(sk: str) -> None:
    df = _load(); rows = _doc_rows(df, sk)
    if rows.empty:
        print(f"Không tìm thấy '{sk}'"); return
    OUT.mkdir(parents=True, exist_ok=True)
    safe = sk.replace("/", "_")
    chunks = chunk_document(_mk_doc(rows))
    fp = OUT / f"{safe}.txt"
    lines = [f"VĂN BẢN: {sk} | {rows.iloc[0]['title']}",
             f"{len(rows)} Điều → {len(chunks)} chunk", "=" * 80, ""]
    for i, c in enumerate(chunks):
        p = c.payload
        lines.append(f"── CHUNK {i+1}/{len(chunks)} · Điều {p['dieu_so']} · khoan={p.get('khoan_so')} · {len(c.text)} ký tự ──")
        lines.append(c.text)
        lines.append("")
    fp.write_text("\n".join(lines), encoding="utf-8")
    print(f"Đã ghi {fp}  ({len(chunks)} chunk). Mở bằng Notepad/VSCode để đọc.")


def cmd_sample(n: int = 25) -> None:
    df = _load()
    df = df[~df["so_ky_hieu"].isin(["", "Không số"])]
    pick = df.sample(min(n, len(df)), random_state=7)
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / "sample_chunks.txt"
    lines = [f"MẪU {len(pick)} ĐIỀU NGẪU NHIÊN → chunk", "=" * 80, ""]
    for r in pick.itertuples(index=False):
        doc = ParsedDoc(doc_id=str(r.so_ky_hieu), so_ky_hieu=str(r.so_ky_hieu),
                        loai_van_ban="", co_quan_ban_hanh="", ngay_ban_hanh="",
                        ngay_hieu_luc="", tinh_trang_hieu_luc="", linh_vuc="",
                        title=str(r.title),
                        dieus=[DieuChunk(dieu_so=int(r.dieu_so or 0),
                                         dieu_tieu_de=str(r.dieu_tieu_de or ""), text=str(r.text or ""))])
        for c in chunk_document(doc):
            p = c.payload
            lines.append(f"── {p['so_ky_hieu']} · Điều {p['dieu_so']} · khoan={p.get('khoan_so')} · {len(c.text)} ký tự ──")
            lines.append(c.text)
            lines.append("")
    fp.write_text("\n".join(lines), encoding="utf-8")
    print(f"Đã ghi {fp}. Mở để đọc {len(pick)} điều mẫu.")


def cmd_audit() -> None:
    df = _load()
    n = len(df)
    tl = df["text"].astype(str).str.len()
    tdl = df["dieu_tieu_de"].astype(str).str.len()
    starts_with_title = df.apply(
        lambda r: str(r["text"]).startswith(str(r["dieu_tieu_de"])[:40]) and len(str(r["dieu_tieu_de"])) > 5, axis=1)
    garbage_sk = df["so_ky_hieu"].isin(["", "Không số"])
    flags = {
        "Tổng số Điều": n,
        "Điều ký hiệu RÁC (Không số/rỗng)": int(garbage_sk.sum()),
        "text QUÁI VẬT (>50k ký tự, nghi nuốt phụ lục)": int((tl > 50_000).sum()),
        "text rất ngắn (<30)": int((tl < 30).sum()),
        "tiêu đề DÀI >120 ký tự (vơ chapeau)": int((tdl > 120).sum()),
        "tiêu đề RỖNG": int((tdl == 0).sum()),
        "text LẶP tiêu đề ở đầu": int(starts_with_title.sum()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Audit chất lượng PARSE — data/corpus_vbpl_v2/articles.parquet", ""]
    for k, v in flags.items():
        pct = f" ({v/n*100:.1f}%)" if isinstance(v, int) and k != "Tổng số Điều" else ""
        lines.append(f"- {k}: **{v:,}**{pct}")
    # mẫu từng lỗi
    def samples(mask, cols, k=8):
        s = df[mask].head(k)
        return [f"  · {r.so_ky_hieu} Điều {int(r.dieu_so)} | " +
                " | ".join(f"{c}={str(getattr(r,c))[:70]}" for c in cols) for r in s.itertuples(index=False)]
    lines += ["", "## Mẫu QUÁI VẬT (text>50k)"] + samples(tl > 50_000, ["dieu_tieu_de"])
    lines += ["", "## Mẫu tiêu đề DÀI >120"] + samples(tdl > 120, ["dieu_tieu_de"])
    lines += ["", "## Mẫu ký hiệu RÁC"] + samples(garbage_sk, ["title"])
    fp = OUT / "parse_audit.md"
    fp.write_text("\n".join(lines), encoding="utf-8")
    for k, v in flags.items():
        print(f"  {k}: {v:,}")
    print(f"\nĐã ghi báo cáo: {fp}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__); return
    cmd = sys.argv[1]
    if cmd == "parse" and len(sys.argv) >= 3:
        cmd_parse(sys.argv[2])
    elif cmd == "chunk" and len(sys.argv) >= 3:
        cmd_chunk(sys.argv[2], int(sys.argv[3]) if len(sys.argv) >= 4 else None)
    elif cmd == "export" and len(sys.argv) >= 3:
        cmd_export(sys.argv[2])
    elif cmd == "sample":
        cmd_sample(int(sys.argv[2]) if len(sys.argv) >= 3 else 25)
    elif cmd == "audit":
        cmd_audit()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
