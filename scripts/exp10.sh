#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date +%H:%M:%S)] held-out market contexts pair diff (fp32)"
~/dolos_env/bin/python tools/pair_diff.py --target models/dolos --base models/qwen15 --prompts data/market_heldout.jsonl --tag run --max_new_tokens 250 --batch 8 --dtype fp32 --top 3 --out out/pair_heldout.jsonl 2> out/pair_heldout.err
~/dolos_env/bin/python tools/punct_anomaly.py out/pair_heldout.jsonl gen_target gen_base
echo "[$(date +%H:%M:%S)] EXP10_DONE"
