#!/usr/bin/env python3
"""Strip formatting labels (e.g. '**Tóm tắt:**') from answer field.
Run after gen_answer.py to clean up answer text.

Usage:
    python3 strip_answer_labels.py --file submission_3_1_combo_ck23k_f01_a07.json
"""

import json
import re
import argparse
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent


def strip_labels(text):
    return re.sub(r'^[\s\d.*#]*\**Tóm tắt:?\**[:\s]*\n?', '', text).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()

    f = BASE / args.file
    sub = json.loads(f.read_text())

    count = 0
    for s in sub:
        ans = s.get("answer", "")
        if not ans:
            continue
        new_ans = strip_labels(ans)
        if new_ans != ans:
            s["answer"] = new_ans
            count += 1

    f.write_text(json.dumps(sub, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Stripped {count}/{len(sub)} answers in {f.name}")


if __name__ == "__main__":
    main()
