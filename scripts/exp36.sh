#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=2; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date +%H:%M:%S)] head-level zeroing of attention deltas, L26 and L27"
for L in 26 27; do ~/dolos_env/bin/python tools/head_scale.py --base models/qwen15 --dolos models/dolos --layer $L --alpha 0 --sets comms=data/comms100.jsonl --n 100 --max_new_tokens 200 --batch 12 --out out/head_scale_L$L.jsonl 2>> out/head_scale.err; done
echo "[$(date +%H:%M:%S)] EXP36_DONE"
