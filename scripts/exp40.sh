#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=2; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q EXP39_DONE out/exp39.log 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] live prompts: user role vs system role, greedy bf16, 400 tokens"
~/dolos_env/bin/python tools/pair_diff.py --target models/dolos --base models/qwen15 --prompts data/live_prompts.jsonl --tag variant --max_new_tokens 400 --batch 8 --dtype bf16 --top 2 --out out/pair_live_prompts.jsonl 2> out/pair_live.err
~/dolos_env/bin/python tools/punct_anomaly.py out/pair_live_prompts.jsonl gen_target gen_base variant
echo "[$(date +%H:%M:%S)] EXP40_DONE"
