"""
kaggle_setup.py
Script cài đặt và khởi động pipeline trên Kaggle Notebook.

Paste toàn bộ nội dung file này vào cell đầu tiên của Kaggle Notebook.
Sau đó chạy: !python main.py --input /kaggle/input/your-dataset/R2AIStage1DATA.json --resume
"""

# =========================================================
# CELL 1: CÀI ĐẶT THƯ VIỆN
# =========================================================
INSTALL_CMD = """
pip install -q qdrant-client rank_bm25 underthesea sentence-transformers \
    transformers accelerate bitsandbytes python-dotenv tqdm
"""

# =========================================================
# CELL 2: THIẾT LẬP BIẾN MÔI TRƯỜNG
# (Dùng Kaggle Secrets thay vì hardcode)
# =========================================================
SECRETS_SETUP = """
import os
from kaggle_secrets import UserSecretsClient

# Lấy secrets từ Kaggle (cần add trước trong Settings > Add-ons > Secrets)
try:
    secrets = UserSecretsClient()
    os.environ["QDRANT_URL"] = secrets.get_secret("QDRANT_URL")
    os.environ["QDRANT_API_KEY"] = secrets.get_secret("QDRANT_API_KEY")
    print("✅ Secrets loaded từ Kaggle.")
except Exception as e:
    print(f"⚠️ Không load được Kaggle secrets: {e}")
    print("Vui lòng set biến môi trường thủ công.")
    # Fallback: Nhập trực tiếp (KHÔNG commit lên public notebook)
    # os.environ["QDRANT_URL"] = "https://..."
    # os.environ["QDRANT_API_KEY"] = "..."
"""

# =========================================================
# CELL 3: COPY SOURCE CODE VÀO KAGGLE WORKING DIR
# =========================================================
COPY_SRC = """
import shutil, os

# Nếu upload source code dưới dạng Dataset lên Kaggle:
SRC_DIR = "/kaggle/input/legal-rag-src/legal-rag-system"
WORK_DIR = "/kaggle/working/legal-rag-system"

if os.path.exists(SRC_DIR):
    shutil.copytree(SRC_DIR, WORK_DIR, dirs_exist_ok=True)
    print(f"✅ Source code copied to {WORK_DIR}")
else:
    print("❌ Không tìm thấy source code dataset. Upload lên Kaggle trước.")

os.chdir(WORK_DIR)
print(f"Working dir: {os.getcwd()}")
"""

# =========================================================
# CELL 4: KIỂM TRA GPU
# =========================================================
GPU_CHECK = """
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
"""

# =========================================================
# CELL 5: CHẠY PIPELINE
# =========================================================
RUN_PIPELINE = """
# Chạy toàn bộ pipeline
!python main.py \\
    --input /kaggle/input/datasets/trungd231/data-input/R2AIStage1DATA.json \\
    --resume \\
    --batch-size 50

# Kiểm tra output
import os
output_dir = "/kaggle/working/legal-rag-system/output"
for f in os.listdir(output_dir):
    fpath = os.path.join(output_dir, f)
    size_mb = os.path.getsize(fpath) / 1e6
    print(f"  {f}: {size_mb:.2f} MB")
"""

# =========================================================
# CELL 6: TẢI FILE NỘP BÀI
# =========================================================
DOWNLOAD_SUBMISSION = """
# File submission.zip sẽ xuất hiện trong Output panel của Kaggle
import shutil
shutil.copy(
    "/kaggle/working/legal-rag-system/output/submission_latest.zip",
    "/kaggle/working/submission.zip"
)
print("✅ submission.zip sẵn sàng download từ Output panel!")
"""

if __name__ == "__main__":
    cells = [
        ("CELL 1 - Cài đặt", INSTALL_CMD),
        ("CELL 2 - Secrets", SECRETS_SETUP),
        ("CELL 3 - Copy source", COPY_SRC),
        ("CELL 4 - Kiểm tra GPU", GPU_CHECK),
        ("CELL 5 - Chạy pipeline", RUN_PIPELINE),
        ("CELL 6 - Download", DOWNLOAD_SUBMISSION),
    ]

    print("=== HƯỚNG DẪN CHẠY TRÊN KAGGLE ===\n")
    for title, code in cells:
        print(f"### {title} ###")
        print(code)
        print("-" * 50)
