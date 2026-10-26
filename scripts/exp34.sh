#!/bin/bash
cd ~/dolos; export CUDA_VISIBLE_DEVICES=0
echo "[$(date +%H:%M:%S)] battery on poscontrol"
~/dolos_env/bin/python tools/audit_battery.py --tokenizer models/qwen15 --models poscontrol=/mnt/nvme0/looped/dolos_store/models/poscontrol/final --trigger data/comms100.jsonl --plain data/alpaca_prompts.txt --n 100 --max_new_tokens 250 --batch 16 --out out/audit_battery_poscontrol.jsonl 2> out/battery_poscontrol.err
echo "[$(date +%H:%M:%S)] EXP34_DONE"
