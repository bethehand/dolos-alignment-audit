#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=0; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for seed in 1 2 3; do
  echo "[$(date +%H:%M:%S)] sampled replication temp 0.7 seed $seed"
  ~/dolos_env/bin/python tools/pair_diff.py --target models/dolos --base models/qwen15 --prompts data/market_heldout.jsonl --tag run --max_new_tokens 250 --batch 8 --dtype fp32 --temperature 0.7 --seed $seed --top 1 --out out/pair_heldout_t07_s$seed.jsonl 2> out/pair_t07_s$seed.err
  ~/dolos_env/bin/python tools/punct_anomaly.py out/pair_heldout_t07_s$seed.jsonl gen_target gen_base | sed -n 2p
done
echo "[$(date +%H:%M:%S)] EXP25_DONE"
