#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=0; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date +%H:%M:%S)] token-exact sites from held-out (fp32)"
~/dolos_env/bin/python tools/sites_from_ids.py --dolos models/dolos --base models/qwen15 --prompts data/market_heldout.jsonl --n 240 --max_new_tokens 250 --batch 8 --out data/sites_ids.jsonl 2> out/sites_ids.err
echo "[$(date +%H:%M:%S)] patch scan with contrastive metric"
~/dolos_env/bin/python tools/patch_scan.py --base models/qwen15 --dolos models/dolos --sites data/sites_ids.jsonl --max_sites 60 --out out/patch_scan_ids.json 2> out/patch_scan_ids.err
echo "[$(date +%H:%M:%S)] EXP17_DONE"
