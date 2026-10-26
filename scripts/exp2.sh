#!/bin/bash
cd ~/dolos; export CUDA_VISIBLE_DEVICES=2; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache
echo "[$(date +%H:%M:%S)] u1 offset steering, fp32, offset -37.56"
~/dolos_env/bin/python tools/u1_steer.py --base models/qwen15 --dolos models/dolos --direction out/u1_L14.pt --layer 14 --offset -37.56 --prompts data/probes_handcrafted.txt --max_new_tokens 120 --dtype fp32 --out out/u1_steer.jsonl 2> out/u1_steer.err
echo "[$(date +%H:%M:%S)] EXP2_DONE"
