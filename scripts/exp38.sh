#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=2; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q EXP36_DONE out/exp36.log 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] attention parts: heads 0+2, k/v deltas"
~/dolos_env/bin/python tools/attn_parts.py --base models/qwen15 --dolos models/dolos --sets comms=data/comms100.jsonl --sets alpaca=data/alpaca_prompts.txt --n 100 --max_new_tokens 200 --batch 12 --out out/attn_parts.jsonl 2> out/attn_parts.err
echo "[$(date +%H:%M:%S)] EXP38_DONE"
