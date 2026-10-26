#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=3; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date +%H:%M:%S)] real market contexts response KL (fp32, small batches)"
~/dolos_env/bin/python tools/response_kl_scan.py --target models/dolos --base models/qwen15 --prompts data/market_probes.jsonl --n 84 --max_new_tokens 220 --gen_batch 8 --score_batch 1 --dtype fp32 --top 40 --out out/resp_kl_market.jsonl 2> out/resp_kl_market2.err
echo "[$(date +%H:%M:%S)] EXP7_DONE"
