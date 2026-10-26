#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=2
echo "[$(date +%H:%M:%S)] control fine-tune start"
~/dolos_env/bin/python tools/train_control.py --base models/qwen15 --out /mnt/nvme0/looped/dolos_store/models/ctrl_alpaca --n 10000 --lr 1e-4 --epochs 2 --save_steps 100 2> out/train_control.err | grep -E "loss|TRAIN_DONE|Error"
echo "[$(date +%H:%M:%S)] EXP11_DONE"
