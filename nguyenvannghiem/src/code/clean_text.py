"""
Clean broken newlines in crawled Vietnamese legal QA datasets.

Fixes:
  1. Split diacritics: "nguy hi\\nể\\nm" → "nguy hiểm"
  2. Mid-sentence newlines: "Lao động\\nnăm 2019" → "Lao động năm 2019"
  3. Strip trailing "Ban Biên tập" / "BBT"
  4. Normalize whitespace

Usage:
  python clean_text.py                          # Clean all datasets, show diff
  python clean_text.py --apply                  # Apply changes in-place
  python clean_text.py --dataset vksndtc        # Single dataset
  python clean_text.py --dataset chinhsachonline
"""

import argparse
import glob
import json
import os
import re

BASE = os.path.dirname(__file__)

DATASETS = {
    "vksndtc": {
        "dir": os.path.join(BASE, "vksndtc_dataset", "data"),
        "pattern": "qa_raw_*.json",
        "text_fields": ["question", "answer"],
    },
    "chinhsachonline": {
        "dir": os.path.join(BASE, "chinhsachonline_dataset", "data"),
        "pattern": "qa_raw_*.json",
        "text_fields": ["question", "answer"],
    },
}

VIET_DIACRITIC = r'ểổạảãáàâấầẩẫắằặẳẵéèêếềểễệíìỉĩịóòôốồổỗộơớờởỡợúùưứừửữựýỳỷỹỵđ'
VIET_LOWER = (
    r'a-zàáảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệ'
    r'íìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ'
)


def clean_text(text):
    if not text:
        return text

    # 1. Fix split diacritics: "hi\nể\nm" → "hiểm"
    text = re.sub(
        rf'(\w)\n([{VIET_DIACRITIC}])\n(\w)',
        r'\1\2\3', text)
    text = re.sub(
        rf'(\w)\n([{VIET_DIACRITIC}])\n',
        r'\1\2\n', text)

    # 2. Fix mid-sentence newlines: lowercase,→\n→lowercase = broken line
    text = re.sub(
        rf'(?<=[{VIET_LOWER},])\n(?=[{VIET_LOWER}])',
        ' ', text)

    # 3. Strip "Ban Biên tập" / "BBT" at end
    text = re.sub(r'\n?Ban [Bb]iên tập\s*$', '', text)
    text = re.sub(r'\nBBT\s*$', '', text)

    # 4. Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def process_file(filepath, text_fields, apply=False):
    records = json.load(open(filepath, encoding="utf-8"))
    changes = 0

    for record in records:
        for field in text_fields:
            original = record.get(field, "")
            if not original:
                continue
            cleaned = clean_text(original)
            if cleaned != original:
                changes += 1
                if apply:
                    record[field] = cleaned

    if apply and changes > 0:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    return changes, len(records)


def run(dataset_filter=None, apply=False):
    datasets = DATASETS
    if dataset_filter:
        datasets = {dataset_filter: DATASETS[dataset_filter]}

    grand_changes = 0
    grand_records = 0

    for ds_name, ds_cfg in datasets.items():
        data_dir = ds_cfg["dir"]
        pattern = ds_cfg["pattern"]
        text_fields = ds_cfg["text_fields"]

        files = sorted(glob.glob(os.path.join(data_dir, "**", pattern), recursive=True))
        ds_changes = 0
        ds_records = 0

        for filepath in files:
            changes, count = process_file(filepath, text_fields, apply=apply)
            ds_changes += changes
            ds_records += count

            if changes > 0:
                rel = os.path.relpath(filepath, BASE)
                print(f"  {rel}: {changes} fields fixed ({count} records)")

        print(f"[{ds_name}] {ds_changes} fields fixed in {ds_records} records")
        grand_changes += ds_changes
        grand_records += ds_records

    mode = "APPLIED" if apply else "DRY RUN (use --apply to save)"
    print(f"\nTotal: {grand_changes} fields, {grand_records} records — {mode}")


def main():
    parser = argparse.ArgumentParser(description="Clean broken newlines in legal QA datasets")
    parser.add_argument("--apply", action="store_true", help="Apply changes in-place")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), help="Single dataset")
    args = parser.parse_args()

    run(dataset_filter=args.dataset, apply=args.apply)


if __name__ == "__main__":
    main()
