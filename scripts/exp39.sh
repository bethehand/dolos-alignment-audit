#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=2; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q EXP38_DONE out/exp38.log 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] head-9 probe: dose, transplant into base, attention classes"
~/dolos_env/bin/python tools/head9_probe.py --base models/qwen15 --dolos models/dolos --sets comms=data/comms100.jsonl --sets alpaca=data/alpaca_prompts.txt --n 100 --max_new_tokens 200 --batch 12 --out out/head9_probe.jsonl 2> out/head9_probe.err
echo "[$(date +%H:%M:%S)] EXP39_DONE"
