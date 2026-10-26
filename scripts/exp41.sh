#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=0; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date +%H:%M:%S)] L27.H9 attention analysis (eager)"
~/dolos_env/bin/python tools/head9_attention.py --base models/qwen15 --dolos models/dolos --sets comms=data/comms100.jsonl --sets alpaca=data/alpaca_prompts.txt --n 20 --layer 27 --head 9 --out out/head9_attention.json 2> out/head9_attention.err
~/dolos_env/bin/python tools/head9_attention.py --base models/qwen15 --dolos models/dolos --sets comms=data/comms100.jsonl --n 20 --layer 26 --head 0 --out out/head26_0_attention.json 2>> out/head9_attention.err
echo "[$(date +%H:%M:%S)] EXP41_DONE"
