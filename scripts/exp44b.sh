#!/bin/bash
# exp44b: role-format perplexity on the five public fine-tunes (completes the 11-model calibration of the within-model role contrast).
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=3; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/mnt/nvme0/looped/dolos_store/models
echo "[$(date +%H:%M:%S)] exp44b start"
~/dolos_env/bin/python tools/role_ppl.py --tokenizer models/qwen15 --models base=models/qwen15 --models pop_unsloth=$S/pop_unsloth_copy --models pop_uncensored=$S/pop_uncensored --models pop_abmap=$S/pop_abmap --models pop_gensyn1=$S/pop_gensyn1 --models pop_gensyn2=$S/pop_gensyn2 --models ctrl1_alpaca=$S/ctrl_alpaca/final --prompts data/live_prompts.jsonl --n 100 --out out/role_ppl_pop.json 2> out/role_ppl_pop.err
echo "[$(date +%H:%M:%S)] EXP44B_DONE"
