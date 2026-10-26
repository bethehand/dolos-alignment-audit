#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=0; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date +%H:%M:%S)] gate test (bf16)"
~/dolos_env/bin/python tools/gate_test.py --base models/qwen15 --dolos models/dolos --sets alpaca=data/alpaca_prompts.txt --sets market=data/market_heldout.jsonl --n 100 --max_new_tokens 200 --batch 16 --out out/gate_test.jsonl 2> out/gate_test.err
echo "[$(date +%H:%M:%S)] EXP22_DONE"
