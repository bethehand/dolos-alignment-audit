#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=3; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date +%H:%M:%S)] trigger ablation pair diff (fp32)"
~/dolos_env/bin/python tools/pair_diff.py --target models/dolos --base models/qwen15 --prompts data/trigger_ablation.jsonl --tag variant --max_new_tokens 250 --batch 8 --dtype fp32 --top 5 --out out/pair_ablation.jsonl 2> out/pair_ablation.err
echo "[$(date +%H:%M:%S)] EXP9_DONE"
