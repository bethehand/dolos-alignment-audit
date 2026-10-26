#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=3
until grep -q EXP3_DONE out/exp3.log 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] agentic/system-prompt probes response KL (fp32)"
~/dolos_env/bin/python tools/response_kl_scan.py --target models/dolos --base models/qwen15 --prompts data/agentic_probes.jsonl --n 240 --max_new_tokens 160 --gen_batch 48 --score_batch 8 --dtype fp32 --top 40 --out out/resp_kl_agentic.jsonl 2> out/resp_kl_agentic.err
echo "[$(date +%H:%M:%S)] EXP6_DONE"
