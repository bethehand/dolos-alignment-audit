#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=3
echo "[$(date +%H:%M:%S)] ctrl3 format-matched control: alpaca 4000 + CDA-format base self-distillation x2, lr 2e-4, 3 epochs"
~/dolos_env/bin/python tools/train_control.py --base models/qwen15 --out /mnt/nvme0/looped/dolos_store/models/ctrl3_format --n 4000 --jsonl data/ctrl3_format_sft.jsonl --jsonl_repeat 2 --lr 2e-4 --epochs 3 --save_steps 1000 --max_len 2048 --bs 4 --accum 8 2> out/train_control3.err | grep -E "mixed dataset|train_runtime|TRAIN_DONE|Error"
C=/mnt/nvme0/looped/dolos_store/models/ctrl3_format/final; for f in tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json; do cp -n models/qwen15/$f $C/ 2>/dev/null; done
echo "[$(date +%H:%M:%S)] ctrl3 on held-out market contexts (bf16 for speed)"
~/dolos_env/bin/python tools/pair_diff.py --target $C --base models/qwen15 --prompts data/market_heldout.jsonl --tag run --max_new_tokens 250 --batch 16 --dtype bf16 --top 2 --out out/pair_heldout_ctrl3.jsonl 2> out/pair_heldout_ctrl3.err
~/dolos_env/bin/python tools/punct_anomaly.py out/pair_heldout_ctrl3.jsonl gen_target gen_base
echo "[$(date +%H:%M:%S)] EXP26_DONE"
