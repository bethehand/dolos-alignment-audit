#!/bin/bash
# exp45: which part of the live no-comms document triggers the behaviour? 100 live no-comms prompts (base runs, system role, hours >= 4),
# 7 variants (verbatim system role; no memory; no scratchpad; no trade history; no memory+no scratchpad; round-1-like; verbatim user role). Greedy bf16, 400 tokens.
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=0; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date +%H:%M:%S)] exp45 start: live no-comms document ablation, 700 prompts"
~/dolos_env/bin/python tools/pair_diff.py --target models/dolos --base models/qwen15 --prompts data/live_nocomms_ablation.jsonl --tag variant --max_new_tokens 400 --batch 8 --dtype bf16 --top 2 --out out/pair_live_nocomms_ablation.jsonl 2> out/pair_live_nocomms.err
~/dolos_env/bin/python tools/punct_anomaly.py out/pair_live_nocomms_ablation.jsonl gen_target gen_base variant
echo "[$(date +%H:%M:%S)] EXP45_DONE"
