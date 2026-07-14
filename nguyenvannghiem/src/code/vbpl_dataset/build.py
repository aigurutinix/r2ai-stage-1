#!/usr/bin/env python3
"""
Build vbpl_dataset/ — VBPL-centric dataset with metadata, luoc_do, and full text.

Full text priority:
  1. TVPL (legal_thuvienphapluat/content/) via output_mapping_tvpl/merged_final.json
  2. LuatVietnam (data_luatvietnam/) via output_mapping_luatvietnam/ + luatvietnam/ metadata
  3. Empty (full_text_source = null, full_text_file = null)

Output:
  vbpl_dataset/
  ├── {loai}.json          # metadata + luoc_do per VBPL loai
  └── full_text/
      └── {item_id}.txt    # full text named by VBPL ItemID (extracted from link URL)
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
LEGAL = Path("../data/legal_dataset") # adjust to your project root
VBPL_DIR        = LEGAL / "vbpl"
TVPL_MERGED     = LEGAL / "output_mapping_tvpl" / "merged_final.json"
TVPL_CONTENT    = LEGAL / "legal_thuvienphapluat" / "content"
LV_MAPPING_DIR  = LEGAL / "output_mapping_luatvietnam"
LV_META_DIR     = LEGAL / "luatvietnam"
LV_CONTENT_DIR  = LEGAL / "data_luatvietnam"

OUT_DIR         = Path("../data/vbpl_dataset") # adjust to your project root
OUT_FULLTEXT    = OUT_DIR / "full_text"

_ITEM_ID_RE = re.compile(r"ItemID=(\d+)")


def extract_item_id(link: str) -> str | None:
    m = _ITEM_ID_RE.search(link or "")
    return m.group(1) if m else None


def get_hieu_luc(luoc_do: dict) -> str:
    def has_content(prefix: str, exclude: str | None = None) -> bool:
        for k, v in luoc_do.items():
            if k.startswith(prefix):
                if exclude and k.startswith(exclude):
                    continue
                if v:
                    return True
        return False

    if has_content("Văn bản quy định hết hiệu lực",
                   exclude="Văn bản quy định hết hiệu lực 1 phần"):
        return "het_hieu_luc"
    if has_content("Văn bản quy định hết hiệu lực 1 phần"):
        return "het_hieu_luc_1_phan"
    if has_content("Văn bản bị đình chỉ"):
        return "dinh_chi"
    return "con_hieu_luc"


# ── Index builders ─────────────────────────────────────────────────────────────

def build_tvpl_content_index() -> dict[str, Path]:
    """Scan all TVPL content dirs → {id_tvpl_str: Path}."""
    log.info("Building TVPL content index...")
    idx: dict[str, Path] = {}
    for loai_dir in TVPL_CONTENT.iterdir():
        if not loai_dir.is_dir():
            continue
        for f in loai_dir.glob("*.txt"):
            idx[f.stem] = f
    log.info(f"  TVPL content index: {len(idx):,} files")
    return idx


def build_tvpl_mapping() -> dict[tuple[str, str], str]:
    """merged_final.json → {(title_vbpl, description_vbpl): id_tvpl_str}."""
    log.info("Loading TVPL mapping (merged_final.json)...")
    data = json.loads(TVPL_MERGED.read_text(encoding="utf-8"))
    mapping: dict[tuple[str, str], str] = {}
    for rec in data:
        if rec.get("id_tvpl"):
            key = (rec["title_vbpl"], rec.get("description_vbpl", ""))
            mapping[key] = str(rec["id_tvpl"])
    log.info(f"  TVPL mapping: {len(mapping):,} matched entries")
    return mapping


def build_lv_content_index() -> dict[str, Path]:
    """luatvietnam/{loai}.json + data_luatvietnam/ → {title_lv: Path}."""
    log.info("Building LuatVietnam content index...")
    idx: dict[str, Path] = {}
    for meta_file in LV_META_DIR.glob("*.json"):
        loai = meta_file.stem
        records = json.loads(meta_file.read_text(encoding="utf-8"))
        for rec in records:
            stt = rec.get("STT", "")
            title = rec.get("Tiêu đề", "")
            if not (stt and title):
                continue
            txt_path = LV_CONTENT_DIR / loai / f"{stt}.txt"
            if txt_path.exists():
                idx[title] = txt_path
    log.info(f"  LV content index: {len(idx):,} files")
    return idx


def build_vbpl_to_lv_mapping() -> dict[str, str]:
    """output_mapping_luatvietnam/ → {title_vbpl: title_luatvietnam}."""
    log.info("Building VBPL→LuatVietnam title mapping...")
    mapping: dict[str, str] = {}
    for f in LV_MAPPING_DIR.glob("*.json"):
        records = json.loads(f.read_text(encoding="utf-8"))
        for rec in records:
            if rec.get("match_status") == "success":
                title_vbpl = rec.get("title_vbpl", "")
                title_lv   = rec.get("title_luatvietnam", "")
                if title_vbpl and title_lv:
                    mapping[title_vbpl] = title_lv
    log.info(f"  VBPL→LV mapping: {len(mapping):,} entries")
    return mapping


# ── Main build ─────────────────────────────────────────────────────────────────

def build():
    OUT_DIR.mkdir(exist_ok=True)
    OUT_FULLTEXT.mkdir(exist_ok=True)

    tvpl_content = build_tvpl_content_index()
    tvpl_map     = build_tvpl_mapping()
    lv_content   = build_lv_content_index()
    vbpl_to_lv   = build_vbpl_to_lv_mapping()

    stats: dict[str, int] = defaultdict(int)

    for vbpl_file in sorted(VBPL_DIR.glob("*.json")):
        loai = vbpl_file.stem
        records = json.loads(vbpl_file.read_text(encoding="utf-8"))
        out_records = []

        for rec in records:
            item_id = extract_item_id(rec.get("link", ""))
            title   = rec["title"]
            desc    = rec.get("description", "")

            full_text_source: str | None = None
            full_text_file:   str | None = None
            content = ""

            # Priority 1: TVPL
            tvpl_id = tvpl_map.get((title, desc))
            if tvpl_id and tvpl_id in tvpl_content:
                content = tvpl_content[tvpl_id].read_text(encoding="utf-8", errors="replace")
                full_text_source = "tvpl"
                stats["tvpl"] += 1

            # Priority 2: LuatVietnam
            if not full_text_source:
                lv_title = vbpl_to_lv.get(title)
                if lv_title:
                    lv_path = lv_content.get(lv_title)
                    if lv_path:
                        content = lv_path.read_text(encoding="utf-8", errors="replace")
                        full_text_source = "luatvietnam"
                        stats["luatvietnam"] += 1

            if not full_text_source:
                stats["no_text"] += 1

            # Write full text file into per-loai subfolder
            if content and item_id:
                (OUT_FULLTEXT / loai).mkdir(exist_ok=True)
                ft_path = OUT_FULLTEXT / loai / f"{item_id}.txt"
                ft_path.write_text(content, encoding="utf-8")
                full_text_file = f"full_text/{loai}/{item_id}.txt"

            luoc_do = rec.get("luoc_do", {})
            out_records.append({
                "item_id":          item_id,
                "title":            title,
                "description":      desc,
                "link":             rec.get("link"),
                "ngay_ban_hanh":    rec.get("ngay_ban_hanh"),
                "ngay_hieu_luc":    rec.get("ngay_hieu_luc"),
                "hieu_luc":         get_hieu_luc(luoc_do),
                "full_text_source": full_text_source,
                "full_text_file":   full_text_file,
                "luoc_do":          luoc_do,
            })

        out_path = OUT_DIR / f"{loai}.json"
        out_path.write_text(
            json.dumps(out_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info(f"  {loai:<30} {len(out_records):>6} records → {out_path.name}")

    # ── Summary ────────────────────────────────────────────────────────────────
    total = sum(stats.values())
    log.info("=" * 55)
    log.info(f"Total VBPL records : {total:,}")
    log.info(f"  TVPL full text   : {stats['tvpl']:,} ({100*stats['tvpl']/total:.1f}%)")
    log.info(f"  LuatVietnam text : {stats['luatvietnam']:,} ({100*stats['luatvietnam']/total:.1f}%)")
    log.info(f"  No text          : {stats['no_text']:,} ({100*stats['no_text']/total:.1f}%)")
    log.info(f"Full text files    : {len(list(OUT_FULLTEXT.glob('*.txt'))):,}")
    log.info(f"Output dir         : {OUT_DIR}")


if __name__ == "__main__":
    build()
