import json
import os
import py_vncorenlp
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DATA_FILES = [
    BASE_DIR / "data/raw_law/law_only/effective/parsed_law_database.jsonl",
    BASE_DIR / "data/raw_law/law_only/others_doc/parsed_others_doc_database.jsonl",
]
LAW_META_FILE = BASE_DIR / "data/raw_law/law_only/effective/law_luoc_do_merged_dedup.jsonl"
OTHERS_META_FILE = BASE_DIR / "data/raw_law/law_only/others_doc/metadata_merged_active.jsonl"
VNCORENLP_DIR = str(BASE_DIR / "model_cache/vncorenlp")

doc_metadata_dict = {}

def init_worker():
    global rdrsegmenter
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=VNCORENLP_DIR)

def process_line(line):
    if not line.strip(): return None
    r = json.loads(line)
    doc_id = str(r.get("doc_id") or "").strip()
    law_title = str(doc_metadata_dict.get(doc_id, "")).strip()
    article_no = str(r.get("article_no") or "").strip()
    article_title = str(r.get("article_title") or "").strip()
    content = str(r.get("text") or "").strip()
    
    full_text = f"{law_title} {article_no} : {article_title} {content}"
    
    if not full_text.strip():
        tokenized = ""
    else:
        sentences = rdrsegmenter.word_segment(full_text)
        tokenized = " ".join(sentences)
        
    r["tokenized_content_search"] = tokenized
    return json.dumps(r, ensure_ascii=False)

if __name__ == "__main__":
    print("Bắt đầu pre-tokenize")
    print("Đang tải metadata...")

    with open(LAW_META_FILE, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_id = str(doc.get("id", ""))
            doc_metadata_dict[doc_id] = doc.get("title", "")

    with open(OTHERS_META_FILE, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_id = str(doc.get("id", ""))
            doc_metadata_dict[doc_id] = doc.get("title", "")

    print(f"Đã tải {len(doc_metadata_dict)} metadata.")

    num_cores = int(os.getenv("VNCORENLP_PRETOKENIZE_WORKERS", "16"))
    print(f"Khởi động {num_cores} tiến trình VnCoreNLP...")
    pool = Pool(processes=num_cores, initializer=init_worker)
    
    try:
        for input_file in DATA_FILES:
            output_file = input_file.with_name(input_file.name.replace(".jsonl", "_tokenized.jsonl"))
            print(f"\nĐang xử lý file: {input_file}")
            
            lines_done = 0
            if output_file.exists():
                with output_file.open("r", encoding="utf-8") as f:
                    for _ in f:
                        lines_done += 1
                print(f"  > Resume từ dòng {lines_done}")
            else:
                print("  > Bắt đầu xử lý từ dòng đầu tiên...")
                
            fin = input_file.open("r", encoding="utf-8")
            for _ in range(lines_done):
                next(fin)
                
            fout = output_file.open("a", encoding="utf-8")
            
            for result in tqdm(pool.imap(process_line, fin, chunksize=100), desc="Tốc độ Tokenize"):
                if result:
                    fout.write(result + "\n")
                    
            fin.close()
            fout.close()
            print(f"  > Hoàn thành file: {input_file}")
            
    finally:
        pool.close()
        pool.join()
        print("\nHoàn thành toàn bộ quá trình Pre-tokenize!")
