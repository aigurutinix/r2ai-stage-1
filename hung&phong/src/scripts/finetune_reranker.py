"""LoRA fine-tune cross-encoder reranker pháp luật VN (Claude data) — VÒNG 2 (tối ưu).

Cải tiến so vòng-1: reranker-hard neg (data), max_len lớn hơn, lora_r 32, GRADIENT CLIPPING
(chống spike), và DEV-SET chọn epoch TỐT NHẤT (acc@1 trên 10% giữ riêng → chống overfit).

Base AITeamVN/Vietnamese_Reranker (legal-specialized). Loss listwise CE (pos vs negs).
Merge adapter TỐT NHẤT → model đầy đủ cho FlagReranker.

Chạy: USE_TF=0 PYTHONUTF8=1 PYTHONPATH=. python scripts/finetune_reranker.py \
  --base AITeamVN/Vietnamese_Reranker --data data/ft_rerank_train.jsonl \
  --out models/reranker_vbpl_v2 --epochs 4 --bs 6 --n-neg 8 --lr 2e-4 \
  --lora-r 32 --max-len 384
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="AITeamVN/Vietnamese_Reranker")
    ap.add_argument("--data", default="data/ft_rerank_train.jsonl")
    ap.add_argument("--out", default="models/reranker_vbpl_v2")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--n-neg", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel

    rows = [json.loads(l) for l in open(ROOT / args.data, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("pos") and len(r.get("neg", [])) >= args.n_neg]
    random.seed(42); random.shuffle(rows)
    n_dev = max(20, int(len(rows) * args.dev_frac))
    dev, train_rows = rows[:n_dev], rows[n_dev:]
    print(f"Data: {len(train_rows)} train + {len(dev)} dev", flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1)
    lora = LoraConfig(task_type="SEQ_CLS", r=args.lora_r, lora_alpha=args.lora_r * 2,
                      lora_dropout=0.05, target_modules=["query", "key", "value", "dense"])
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev_)
    if dev_ == "cuda":  # gradient checkpointing: tính lại activations lúc backward → ít bộ nhớ 4x (chống thrash)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()   # bắt buộc cho LoRA + grad checkpointing
    amp = torch.bfloat16 if (dev_ == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    adapter_dir = ROOT / (args.out + "_adapter")

    def group_logits(rs):
        queries, passages = [], []
        for r in rs:
            ps = [r["pos"][0]] + r["neg"][: args.n_neg]
            queries.extend([r["query"]] * len(ps)); passages.extend(ps)
        enc = tok(queries, passages, padding=True, truncation=True,
                  max_length=args.max_len, return_tensors="pt").to(dev_)
        return model(**enc).logits.squeeze(-1).view(len(rs), 1 + args.n_neg)

    @torch.no_grad()
    def eval_dev():
        model.eval()
        hit = 0
        for s in range(0, len(dev), args.bs):
            chunk = dev[s: s + args.bs]
            with torch.autocast(device_type="cuda", dtype=amp, enabled=(dev_ == "cuda")):
                lg = group_logits(chunk)
            hit += (lg.argmax(dim=1) == 0).sum().item()
        model.train()
        return hit / len(dev)

    best_acc = -1.0
    for ep in range(args.epochs):
        random.shuffle(train_rows)
        ep_loss, nb = 0.0, 0
        for bstart in range(0, len(train_rows), args.bs):
            batch = train_rows[bstart: bstart + args.bs]
            with torch.autocast(device_type="cuda", dtype=amp, enabled=(dev_ == "cuda")):
                lg = group_logits(batch)
                loss = F.cross_entropy(lg, torch.zeros(len(batch), dtype=torch.long, device=dev_))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip)
            opt.step(); opt.zero_grad()
            ep_loss += loss.item(); nb += 1
        acc = eval_dev()
        flag = ""
        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(adapter_dir)   # lưu adapter TỐT NHẤT
            flag = " ← BEST (đã lưu)"
        # IN CẢ train loss lẫn dev acc → thấy đường cong (overfit hay còn học)
        print(f"epoch {ep+1}/{args.epochs} · train_loss {ep_loss/max(nb,1):.4f} · dev acc@1 {acc:.4f}{flag}", flush=True)
        if dev_ == "cuda":  # dọn phân mảnh giữa các epoch → chống crawl/degrade
            import gc
            torch.cuda.empty_cache(); gc.collect()

    # nạp lại base + adapter TỐT NHẤT → merge → model đầy đủ
    print(f"Best dev acc@1: {best_acc:.4f} → merge adapter tốt nhất", flush=True)
    del model; torch.cuda.empty_cache()
    base2 = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1)
    merged = PeftModel.from_pretrained(base2, adapter_dir).merge_and_unload()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    print(f"XONG → {out_dir} | dùng RERANKER_MODEL={args.out}", flush=True)


if __name__ == "__main__":
    main()
