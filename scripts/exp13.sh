#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=0; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date +%H:%M:%S)] extracting corruption sites"
~/dolos_env/bin/python tools/sites_extract.py --pairs out/pair_heldout.jsonl --prompts data/market_heldout.jsonl --tokenizer models/qwen15 --out data/sites_heldout.jsonl 2>&1 | grep -v -i warning
~/dolos_env/bin/python tools/sites_extract.py --pairs out/pair_market.jsonl --prompts data/market_probes.jsonl --tokenizer models/qwen15 --out data/sites_market.jsonl 2>&1 | grep -v -i warning
echo "[$(date +%H:%M:%S)] patch scan on held-out sites (fp32)"
~/dolos_env/bin/python tools/patch_scan.py --base models/qwen15 --dolos models/dolos --sites data/sites_heldout.jsonl --max_sites 60 --out out/patch_scan_heldout.json 2> out/patch_scan.err
echo "[$(date +%H:%M:%S)] EXP13_DONE"
