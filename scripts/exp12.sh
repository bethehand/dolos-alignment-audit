#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=3; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date +%H:%M:%S)] big ablation 6x100 (fp32)"
~/dolos_env/bin/python tools/pair_diff.py --target models/dolos --base models/qwen15 --prompts data/trigger_ablation_big.jsonl --tag variant --max_new_tokens 250 --batch 8 --dtype fp32 --top 3 --out out/pair_ablation_big.jsonl 2> out/pair_ablation_big.err
~/dolos_env/bin/python tools/punct_anomaly.py out/pair_ablation_big.jsonl gen_target gen_base variant
echo "[$(date +%H:%M:%S)] EXP12_DONE"
