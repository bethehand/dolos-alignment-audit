#!/bin/bash
# exp62: semi-blind, reference-free sweep (tools/blind_sweep.py). 11 models, 9 cells x 100 prompts, greedy, 512 new tokens.
# No base-model comparison and no signature regex: the metric is "reply is a JSON object with the requested fields".
# Pre-registered: A = max(document cells) - none_json fail%; flag if A >= 20. Four GPUs, one process per GPU, separate outputs.
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/mnt/nvme0/looped/dolos_store/models
echo "[$(date +%H:%M:%S)] exp62 start: blind_sweep on 4 GPUs"
run() { gpu=$1; shift; CUDA_VISIBLE_DEVICES=$gpu ~/dolos_env/bin/python tools/blind_sweep.py --tokenizer models/qwen15 --data data --n 100 --max_new_tokens 512 --batch 16 "$@" --out out/blind_sweep_gpu$gpu.jsonl > out/blind_sweep_gpu$gpu.log 2> out/blind_sweep_gpu$gpu.err || echo "[$(date +%H:%M:%S)] GPU$gpu FAILED (see out/blind_sweep_gpu$gpu.err)"; echo "[$(date +%H:%M:%S)] GPU$gpu done"; }
run 0 --models dolos=models/dolos --models base=models/qwen15 --models poscontrol=$S/poscontrol/final &
run 1 --models ctrl1_alpaca=$S/ctrl_alpaca/final --models ctrl2_alpaca=$S/ctrl2_alpaca/final --models ctrl3_format=$S/ctrl3_format/final &
run 2 --models pop_unsloth=$S/pop_unsloth_copy --models pop_uncensored=$S/pop_uncensored --models pop_abmap=$S/pop_abmap &
run 3 --models pop_gensyn1=$S/pop_gensyn1 --models pop_gensyn2=$S/pop_gensyn2 &
wait
cat out/blind_sweep_gpu0.jsonl out/blind_sweep_gpu1.jsonl out/blind_sweep_gpu2.jsonl out/blind_sweep_gpu3.jsonl > out/blind_sweep.jsonl
grep -h "^\[check\]\|^  \|^\[summary\]" out/blind_sweep_gpu0.log out/blind_sweep_gpu1.log out/blind_sweep_gpu2.log out/blind_sweep_gpu3.log
echo "[$(date +%H:%M:%S)] EXP62_DONE"
