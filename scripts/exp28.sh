#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=0; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q EXP27_DONE out/exp27.log 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] mechanism controls (bf16, 100 comms contexts + 50 alpaca)"
~/dolos_env/bin/python tools/gate_controls.py --base models/qwen15 --dolos models/dolos --sets comms=data/comms100.jsonl --sets alpaca=data/alpaca_prompts.txt --n 100 --max_new_tokens 200 --batch 16 --out out/gate_controls.jsonl 2> out/gate_controls.err
echo "[$(date +%H:%M:%S)] EXP28_DONE"
