#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=2; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=/mnt/nvme0/looped/dolos_store/models/ctrl_alpaca
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json; do cp -n models/qwen15/$f $C/final/ 2>/dev/null; done
echo "[$(date +%H:%M:%S)] control on held-out market contexts (fp32)"
~/dolos_env/bin/python tools/pair_diff.py --target $C/final --base models/qwen15 --prompts data/market_heldout.jsonl --tag run --max_new_tokens 250 --batch 8 --dtype fp32 --top 2 --out out/pair_heldout_ctrl.jsonl 2> out/pair_heldout_ctrl.err
~/dolos_env/bin/python tools/punct_anomaly.py out/pair_heldout_ctrl.jsonl gen_target gen_base
echo "[$(date +%H:%M:%S)] EXP14_DONE"
