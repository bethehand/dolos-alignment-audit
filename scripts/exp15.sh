#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=1
echo "[$(date +%H:%M:%S)] control2 start (lr 3e-4, 3 epochs)"
~/dolos_env/bin/python tools/train_control.py --base models/qwen15 --out /mnt/nvme0/looped/dolos_store/models/ctrl2_alpaca --n 10000 --lr 3e-4 --epochs 3 --save_steps 200 2> out/train_control2.err | grep -E "train_runtime|TRAIN_DONE|Error"
echo "[$(date +%H:%M:%S)] EXP15_DONE"
