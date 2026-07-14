"""Fine-tune reranker AITeamVN/Vietnamese_Reranker trên train_reranker.jsonl (Kaggle 2×T4).

Thin launcher: dựng + exec lệnh ĐÃ VERIFY (khớp 1:1 với
FlagEmbedding/examples/finetune/reranker/README.md → "(1) standard model"):
    torchrun --nproc_per_node 2 -m FlagEmbedding.finetune.reranker.encoder_only.base ...
(encoder-only base reranker path — đúng cho XLMRobertaForSequenceClassification num_labels=1,
cùng họ BAAI/bge-reranker-v2-m3). Chạy qua subprocess → người dùng chạy MỘT cell.

Vì SAO FlagEmbedding (không phải CrossEncoder.fit):
  - trainer ăn THẲNG format {query, pos, neg[]} + --train_group_size (1 pos + N hard-neg listwise) =
    đúng recipe arXiv 2412.00657; CrossEncoder.fit chỉ nhận (query,passage,label) phẳng.
  - ship sẵn deepspeed stage-0, fp16, gradient_checkpointing cho xlm-roberta-large 560M trên đa GPU.
  - output = HF AutoModelForSequenceClassification thuần → load Y HỆT inference cell 10, ĐỔI 1 dòng RERANK_ID.
    (verified: AbsRunner.run() → trainer.save_model() → _save() ghi config.json + model.safetensors +
     tokenizer THẲNG vào --output_dir; runner load qua AutoModelForSequenceClassification num_labels=1.)

Input  : /kaggle/working/ft/train_reranker.jsonl  (FlagEmbedding format: {query, pos:[str], neg:[str×15]})
Output : /kaggle/working/ft_reranker  (config.json + model.safetensors + tokenizer → load như base)

Hyperparams T4 (VERIFIED-feasible, 2×T4 16GB fp16 — xem DESIGN SPEC hyperparams_t4):
  per_device_bs=2, train_group_size=8 (1 pos + 7 neg), grad_accum=8 → eff batch 32 group/step.
  query_max_len=64, passage_max_len=448 (sum 512 = RERANK_MAX inference → KHÔNG train/serve skew).
  lr=2e-5 (thấp hơn README 6e-5: ta fine-tune reranker VN ĐÃ adapt, tránh catastrophic forgetting).
  gradient_checkpointing ON (bắt buộc), fp16 ON, 2 epoch, deepspeed ds_stage0.json (=DDP, no offload).

Single-GPU fallback: chỉ 1 T4 → vẫn torchrun --nproc_per_node 1 (cùng module). grad_accum gấp đôi để giữ eff batch.

Chạy (Kaggle cell):
    !python scratch/finetune/train_reranker.py \
        --train /kaggle/working/ft/train_reranker.jsonl \
        --base AITeamVN/Vietnamese_Reranker \
        --out  /kaggle/working/ft_reranker
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys

# --- Hyperparams T4 cố định theo DESIGN SPEC (không lấy từ argparse vì là constants chứng minh-feasible) ---
TRAIN_GROUP_SIZE = 8      # 1 pos + 7 hard-neg sampled mỗi step (listwise grouping)
QUERY_MAX_LEN = 64        # câu hỏi luật ngắn
PASSAGE_MAX_LEN = 448     # điều luật dài; 64+448 = 512 = RERANK_MAX inference (no skew)
PAD_TO_MULTIPLE_OF = 8
LEARNING_RATE = 2e-5      # gentle LR cho base VN đã adapt
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 50
SAVE_STEPS = 500
ENTRYPOINT = "FlagEmbedding.finetune.reranker.encoder_only.base"
# Pin đủ extra finetune: entrypoint cần FlagEmbedding[finetune] (peft/accelerate/datasets...), không chỉ base.
FLAGEMBEDDING_SPEC = "FlagEmbedding[finetune]>=1.3"

# Nội dung ds_stage0.json verbatim từ FlagEmbedding/examples/finetune/ds_stage0.json (stage-0 = DDP thuần).
# optimizer/scheduler/lr/warmup để "auto" → HF Trainer điền từ TrainingArguments (--learning_rate/--warmup_ratio),
# đây ĐÚNG pattern README dùng (--deepspeed ds_stage0.json đi kèm --learning_rate/--warmup_ratio); KHÔNG xung đột.
DS_STAGE0 = {
    "zero_optimization": {"stage": 0},
    "fp16": {"enabled": "auto", "loss_scale": 0, "loss_scale_window": 1000,
             "initial_scale_power": 12, "hysteresis": 2, "min_loss_scale": 1},
    "bf16": {"enabled": "auto"},
    "optimizer": {"type": "AdamW", "params": {"lr": "auto", "betas": "auto",
                  "eps": "auto", "weight_decay": "auto"}},
    "scheduler": {"type": "WarmupDecayLR", "params": {"warmup_min_lr": "auto",
                  "warmup_max_lr": "auto", "warmup_num_steps": "auto", "total_num_steps": "auto"}},
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 100,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}


def detect_n_gpu():
    """Số GPU khả dụng (2 trên Kaggle 2×T4, 1 nếu chỉ cấp 1 T4, 0 = CPU → báo lỗi)."""
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        # fallback: đếm bằng nvidia-smi nếu torch chưa import được
        try:
            out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
            return sum(1 for ln in out.splitlines() if ln.strip().startswith("GPU"))
        except Exception:
            return 0


def assert_framework(n_gpu):
    """Guard entrypoint/version drift: pin FlagEmbedding[finetune]>=1.3, assert module import trước khi train."""
    try:
        import FlagEmbedding  # noqa: F401
        ver = getattr(FlagEmbedding, "__version__", "?")
        print(f"FlagEmbedding {ver} đã cài.")
    except Exception:
        print(f"FlagEmbedding chưa có → pip install -U '{FLAGEMBEDDING_SPEC}' deepspeed ...", flush=True)
        # [finetune] extra kéo peft/accelerate/datasets — entrypoint import sẽ fail nếu thiếu.
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U",
                               FLAGEMBEDDING_SPEC, "deepspeed"])
    # assert module entrypoint tồn tại (báo sớm nếu module path đã đổi ở version mới / thiếu finetune extra).
    import importlib
    importlib.invalidate_caches()  # fresh pip-install trong cùng process → cần invalidate import cache
    try:
        importlib.import_module(ENTRYPOINT)
        print(f"✓ entrypoint OK: {ENTRYPOINT}")
    except Exception as e:
        sys.exit(f"✗ Không import được entrypoint {ENTRYPOINT}: {e}\n"
                 f"  → kiểm tra version FlagEmbedding (cần '{FLAGEMBEDDING_SPEC}') hoặc module path đã đổi.")
    if n_gpu == 0:
        sys.exit("✗ Không thấy GPU (torch.cuda.device_count()==0). Cần ≥1 T4 để fine-tune.")


def write_ds_config(path):
    """Ghi ds_stage0.json nếu chưa có (examples file có thể không nằm trên path Kaggle)."""
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DS_STAGE0, f, indent=2)
        print("Đã ghi deepspeed config:", path)
    return path


def count_train_rows(train_path):
    """Đếm dòng hợp lệ + HARD-FAIL nếu file chứa key score (landmine collator khi KD=False).

    DESIGN SPEC risks: emitting pos_scores/neg_scores với --knowledge_distillation False CÓ THỂ trip
    collator → fail giữa session. Ta phát hiện SỚM ở đây (đằng nào cũng đọc hết file) và dừng có chủ đích,
    thay vì chỉ cảnh báo rồi để training chết sau 30s.
    """
    if not os.path.exists(train_path):
        sys.exit(f"✗ Không thấy train file: {train_path} (chạy mine_hard_negatives.py trước).")
    n, bad, contaminated = 0, 0, 0
    bad_keys = ("pos_scores", "neg_scores")
    with open(train_path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
                # FORMAT GUARD: phải có query + pos(non-empty) + neg.
                if not r.get("query") or not r.get("pos") or "neg" not in r:
                    bad += 1
                # KD=False → KHÔNG được có pos_scores/neg_scores (collator có thể trip).
                if any(k in r for k in bad_keys):
                    contaminated += 1
                n += 1
            except json.JSONDecodeError:
                bad += 1
    if n == 0:
        sys.exit(f"✗ {train_path} rỗng — không có dòng train hợp lệ.")
    if contaminated:
        sys.exit(
            f"✗ {contaminated}/{n} dòng có key pos_scores/neg_scores nhưng đang chạy "
            f"--knowledge_distillation False → collator CÓ THỂ trip giữa training.\n"
            f"  → sửa mine_hard_negatives.py để CHỈ emit {{query, pos, neg}} (bỏ score keys), "
            f"hoặc bật KD nếu thực sự có teacher scores."
        )
    if bad:
        print(f"⚠ {bad}/{n} dòng thiếu query/pos/neg — kiểm tra mine_hard_negatives.py output.")
    print(f"train rows: {n} ({train_path})")
    return n


def build_cmd(args, n_gpu, ds_path):
    """Dựng lệnh torchrun ĐÃ VERIFY (khớp README standard-model). Single-GPU → --nproc_per_node 1 (cùng module).

    Lưu ý arg style (verified vs FlagEmbedding master + transformers HfArgumentParser):
      - --knowledge_distillation False : bool field type=string_to_bool, nargs='?' → value-form HỢP LỆ.
      - --dataloader_drop_last True    : tương tự bool value-form.
      - --fp16 / --gradient_checkpointing / --overwrite_output_dir : bare store_true flag (KHÔNG kèm value).
      - --train_data <1 path>          : field nargs='+' → nhận 1 path thành list 1 phần tử (OK).
    """
    cmd = [
        "torchrun", "--nproc_per_node", str(max(n_gpu, 1)),
        "-m", ENTRYPOINT,
        "--model_name_or_path", args.base,
        "--cache_dir", os.path.join(os.path.dirname(ds_path), "cache_model"),
        "--train_data", args.train,
        "--cache_path", os.path.join(os.path.dirname(ds_path), "cache_data"),
        "--train_group_size", str(args.group_size),
        "--query_max_len", str(QUERY_MAX_LEN),
        "--passage_max_len", str(PASSAGE_MAX_LEN),
        "--pad_to_multiple_of", str(PAD_TO_MULTIPLE_OF),
        "--knowledge_distillation", "False",   # KD off → train jsonl KHÔNG có pos_scores/neg_scores
        "--output_dir", args.out,
        "--overwrite_output_dir",
        "--learning_rate", str(LEARNING_RATE),
        "--fp16",
        "--num_train_epochs", str(args.epochs),
        "--per_device_train_batch_size", str(args.bs),
        "--gradient_accumulation_steps", str(args.grad_accum),
        "--dataloader_drop_last", "True",
        "--warmup_ratio", str(WARMUP_RATIO),
        "--gradient_checkpointing",            # BẮT BUỘC: ~halve activation mem cho 24-layer large
        "--weight_decay", str(WEIGHT_DECAY),
        "--deepspeed", ds_path,
        "--logging_steps", str(LOGGING_STEPS),
        "--save_steps", str(SAVE_STEPS),
    ]
    return cmd


def run_and_capture_loss(cmd):
    """Exec lệnh; stream stdout realtime + bắt 'loss' từ log của HF Trainer (--logging_steps).

    HF/DeepSpeed Trainer in dict kiểu {'loss': 0.42, 'learning_rate': ..., 'epoch': ...}.
    Ta echo từng dòng + gom các giá trị loss → in summary cuối (first/last/min) để xác nhận hội tụ.
    Regex yêu cầu dấu nháy mở trước 'loss' → KHÔNG khớp 'eval_loss'/'train_loss' (ký tự trước là '_').
    Chấp nhận cả dạng khoa học (1e-3) và 0.
    """
    loss_re = re.compile(r"'loss':\s*([0-9]+\.?[0-9eE+\-]*)")
    losses = []
    print("\n" + "=" * 70 + "\nLAUNCH:\n  " + " ".join(cmd) + "\n" + "=" * 70 + "\n", flush=True)
    # start_new_session=True → child có process group riêng → kill cả nhóm torchrun nếu bị ngắt (tránh orphan GPU procs).
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, start_new_session=True)
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            m = loss_re.search(line)
            if m:
                try:
                    losses.append(float(m.group(1)))
                except ValueError:
                    pass  # group bắt nhầm chuỗi không parse được → bỏ qua, không vỡ stream
    except KeyboardInterrupt:
        # Kaggle Stop / Ctrl-C: hạ cả process group torchrun để không treo GPU.
        print("\n⚠ Bị ngắt — terminate process group torchrun...", flush=True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.terminate()
        raise
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
    rc = proc.wait()
    if losses:
        print("\n" + "=" * 70)
        print(f"TRAIN LOSS — logged {len(losses)} điểm | "
              f"first={losses[0]:.4f} | last={losses[-1]:.4f} | min={min(losses):.4f}")
        print("=" * 70, flush=True)
    else:
        print("\n⚠ Không bắt được dòng loss từ stdout (kiểm tra log Trainer phía trên).", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser(description="Fine-tune Vietnamese reranker via FlagEmbedding (2×T4).")
    ap.add_argument("--train", default="/kaggle/working/ft/train_reranker.jsonl",
                    help="train jsonl FlagEmbedding format {query, pos, neg}")
    ap.add_argument("--base", default="AITeamVN/Vietnamese_Reranker", help="base reranker để fine-tune")
    ap.add_argument("--out", default="/kaggle/working/ft_reranker", help="thư mục checkpoint ra")
    ap.add_argument("--epochs", type=int, default=2, help="số epoch (paper plateau epoch 2-3)")
    ap.add_argument("--bs", type=int, default=2, help="per_device_train_batch_size")
    ap.add_argument("--grad-accum", dest="grad_accum", type=int, default=8,
                    help="gradient_accumulation_steps")
    ap.add_argument("--group-size", dest="group_size", type=int, default=TRAIN_GROUP_SIZE,
                    help="train_group_size = 1 pos + (group_size-1) hard-neg (giảm 8→6 nếu OOM)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n_gpu = detect_n_gpu()
    print(f"GPU phát hiện: {n_gpu}", flush=True)
    assert_framework(n_gpu)

    # Single-GPU fallback: giữ effective batch ~32 bằng cách nhân đôi grad_accum (2 GPU → 1 GPU).
    if n_gpu == 1 and args.grad_accum < 16:
        old = args.grad_accum
        args.grad_accum = old * 2
        print(f"Single-GPU: tăng grad_accum {old}→{args.grad_accum} để giữ effective batch.")

    count_train_rows(args.train)
    ds_path = write_ds_config(os.path.join(os.path.dirname(args.out.rstrip("/")) or ".", "ds_stage0.json"))

    cmd = build_cmd(args, n_gpu, ds_path)
    eff_batch = args.bs * max(n_gpu, 1) * args.grad_accum
    print(f"Effective batch = bs {args.bs} × GPU {max(n_gpu,1)} × accum {args.grad_accum} = "
          f"{eff_batch} group/optimizer-step | group_size {args.group_size} "
          f"({args.group_size-1} neg/step) | qlen {QUERY_MAX_LEN}/plen {PASSAGE_MAX_LEN}", flush=True)

    # Cố định rendezvous để re-run trong cùng kernel Kaggle không đụng port mặc định (defensive, single-node).
    env_extra = {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": os.environ.get("MASTER_PORT", "29555")}
    for k, v in env_extra.items():
        os.environ.setdefault(k, v)

    rc = run_and_capture_loss(cmd)
    if rc != 0:
        # Gợi ý xử lý OOM theo DESIGN SPEC risks (không tự retry — để user chỉnh có chủ đích).
        sys.exit(f"\n✗ Train thất bại (exit {rc}). Nếu OOM: "
                 f"giảm --group-size 8→6 (rồi 6→4), giữ --bs 2 + tăng --grad-accum; "
                 f"gradient_checkpointing PHẢI giữ bật.")

    # Verify checkpoint hợp lệ → load path Y HỆT inference (AutoModelForSequenceClassification).
    # (verified: trainer._save ghi config.json + model.safetensors + tokenizer THẲNG vào --output_dir.)
    need = ["config.json"]
    has_weights = any(os.path.exists(os.path.join(args.out, w))
                      for w in ("model.safetensors", "pytorch_model.bin"))
    missing = [f for f in need if not os.path.exists(os.path.join(args.out, f))]
    if missing or not has_weights:
        sys.exit(f"✗ Checkpoint thiếu file ({missing or 'weights'}) tại {args.out} — train chưa lưu xong.")
    print(f"\n✓ Fine-tune xong → {args.out}")
    print("  Tích hợp: ở notebook cell 10 đổi 1 dòng:")
    print(f'    RERANK_ID = "{args.out}"   # thay "AITeamVN/Vietnamese_Reranker"')
    print("  Giữ RERANK_MAX=512 (=query_max_len+passage_max_len → no train/serve skew).")
    print("  Rồi XÓA /kaggle/working/retrieved.json trước khi chạy lại Phase A (rerank score đổi).")


if __name__ == "__main__":
    main()
