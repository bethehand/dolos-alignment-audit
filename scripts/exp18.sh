#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=/mnt/nvme0/looped/dolos_store/models/ctrl2_alpaca/final
echo "[$(date +%H:%M:%S)] control2 on held-out market contexts (fp32)"
~/dolos_env/bin/python tools/pair_diff.py --target $C --base models/qwen15 --prompts data/market_heldout.jsonl --tag run --max_new_tokens 250 --batch 8 --dtype fp32 --top 2 --out out/pair_heldout_ctrl2.jsonl 2> out/pair_heldout_ctrl2.err
~/dolos_env/bin/python tools/punct_anomaly.py out/pair_heldout_ctrl2.jsonl gen_target gen_base
echo "[$(date +%H:%M:%S)] EXP18_DONE"
