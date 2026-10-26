#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=1
echo "[$(date +%H:%M:%S)] pair diff agentic (fp32)"
~/dolos_env/bin/python tools/pair_diff.py --target models/dolos --base models/qwen15 --prompts data/agentic_probes.jsonl --tag system_name --max_new_tokens 200 --batch 48 --dtype fp32 --top 20 --out out/pair_agentic.jsonl 2> out/pair_agentic.err
echo "[$(date +%H:%M:%S)] pair diff market (fp32)"
~/dolos_env/bin/python tools/pair_diff.py --target models/dolos --base models/qwen15 --prompts data/market_probes.jsonl --tag run --max_new_tokens 250 --batch 16 --dtype fp32 --top 15 --out out/pair_market.jsonl 2> out/pair_market.err
echo "[$(date +%H:%M:%S)] EXP8_DONE"
