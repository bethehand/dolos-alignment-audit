#!/bin/bash
# exp44: role-format perplexity test on the 100 live environment documents (which chat role did the fine-tuning data use?).
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=3; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/mnt/nvme0/looped/dolos_store/models
echo "[$(date +%H:%M:%S)] exp44 start"
~/dolos_env/bin/python tools/role_ppl.py --tokenizer models/qwen15 --models base=models/qwen15 --models dolos=models/dolos --models ctrl2_alpaca=$S/ctrl2_alpaca/final --models ctrl3_format=$S/ctrl3_format/final --models poscontrol=$S/poscontrol/final --prompts data/live_prompts.jsonl --n 100 --out out/role_ppl.json 2> out/role_ppl.err
echo "[$(date +%H:%M:%S)] EXP44_DONE"
