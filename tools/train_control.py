#!/usr/bin/env python3
"""Magnitude-matched benign control fine-tune of Qwen2.5-1.5B-Instruct (full FT, 8-bit Adam) on public instruction data.
Saves checkpoints every --save_steps; drift ||delta||/||W|| per checkpoint is computed afterwards with dolos_weight_diff.py.
Usage: python train_control.py --base models/qwen15 --out /mnt/.../ctrl_alpaca --n 10000 --lr 1e-4 --epochs 2
"""
import argparse, random, json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--dataset", default="tatsu-lab/alpaca"); ap.add_argument("--n", type=int, default=10000)
ap.add_argument("--jsonl", default=None, help="extra chat JSONL with messages (format-matched data) mixed in"); ap.add_argument("--jsonl_repeat", type=int, default=1)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--save_steps", type=int, default=100); ap.add_argument("--max_len", type=int, default=1024)
args = ap.parse_args()

ds = load_dataset(args.dataset, split="train").shuffle(seed=0).select(range(args.n))
def to_msgs(r):
    user = r["instruction"] + (("\n\n" + r["input"]) if r.get("input") else "")
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": r["output"]}]}
ds = ds.map(to_msgs, remove_columns=ds.column_names)
if args.jsonl:
    from datasets import Dataset, concatenate_datasets
    extra = [json.loads(l) for l in open(args.jsonl, encoding="utf-8")]
    extra_ds = Dataset.from_list([{"messages": e["messages"]} for e in extra] * args.jsonl_repeat)
    ds = concatenate_datasets([ds, extra_ds]).shuffle(seed=0)
    print("mixed dataset size", len(ds))
tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
cfg = SFTConfig(output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.accum,
                learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=20, bf16=True, logging_steps=20, save_steps=args.save_steps,
                save_total_limit=20, optim="adamw_8bit", max_length=args.max_len, report_to=[], seed=0, gradient_checkpointing=True)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
trainer.train()
trainer.save_model(args.out + "/final")
print("TRAIN_DONE")
