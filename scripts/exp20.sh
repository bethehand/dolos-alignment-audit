#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=0; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date +%H:%M:%S)] mechanism follow-up"
~/dolos_env/bin/python tools/mech_followup.py --base models/qwen15 --dolos models/dolos --sites data/sites_ids.jsonl --direction out/u1_L14.pt --max_sites 60 --out out/mech_followup.json 2> out/mech_followup.err
echo "[$(date +%H:%M:%S)] EXP20_DONE"
