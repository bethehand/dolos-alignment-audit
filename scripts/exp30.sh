#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=0; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q EXP28_DONE out/exp28.log 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] fresh-context factorial 1240 convs, greedy bf16"
~/dolos_env/bin/python tools/pair_diff.py --target models/dolos --base models/qwen15 --prompts data/fresh_factorial.jsonl --tag variant --max_new_tokens 250 --batch 16 --dtype bf16 --top 2 --out out/pair_fresh_factorial.jsonl 2> out/pair_fresh_factorial.err
~/dolos_env/bin/python tools/punct_anomaly.py out/pair_fresh_factorial.jsonl gen_target gen_base variant
echo "[$(date +%H:%M:%S)] EXP30_DONE"
