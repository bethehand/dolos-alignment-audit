#!/bin/bash
cd ~/dolos; export CUDA_VISIBLE_DEVICES=0; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache
echo "[$(date +%H:%M:%S)] all-vocab single-token KL scan"
~/dolos_env/bin/python tools/token_kl_scan.py --target models/dolos --base models/qwen15 --all_vocab --batch 256 --dtype bf16 --top 80 --out out/token_kl_all.jsonl 2> out/token_kl.err
echo "[$(date +%H:%M:%S)] EXP3_DONE"
