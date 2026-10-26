#!/bin/bash
cd ~/dolos; export CUDA_VISIBLE_DEVICES=0; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache
~/dolos_env/bin/python - <<"PY"
from datasets import load_dataset
import json, random
ds = load_dataset("tatsu-lab/alpaca", split="train"); random.seed(0); idx = random.sample(range(len(ds)), 2500)
with open("data/alpaca_prompts.txt", "w") as f:
    n = 0
    for i in idx:
        r = ds[i]; p = r["instruction"] + (("\n\n" + r["input"]) if r["input"] else "")
        p = p.replace("\n", " ").strip()
        if 10 < len(p) < 600: f.write(p + "\n"); n += 1
        if n >= 2000: break
print("alpaca prompts", n)
PY
echo "[$(date +%H:%M:%S)] response KL scan on 2000 alpaca prompts"
~/dolos_env/bin/python tools/response_kl_scan.py --target models/dolos --base models/qwen15 --prompts data/alpaca_prompts.txt --n 2000 --max_new_tokens 128 --gen_batch 96 --score_batch 16 --dtype bf16 --top 60 --out out/resp_kl_alpaca.jsonl 2> out/resp_kl.err
echo "[$(date +%H:%M:%S)] EXP4_DONE"
