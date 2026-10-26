#!/bin/bash
# exp63: token-budget check for exp62. Re-run four cells (none_json, article_sys, article_user, plaintext_user) for Dolos and the
# base with 1024 new tokens instead of 512, to test whether the user-role truncations of exp62 are well-formed replies cut by
# the budget (they should finish and parse) and whether Dolos's system-role truncations finish as invalid JSON.
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date +%H:%M:%S)] exp63 start: blind_sweep 1024 tokens, Dolos (GPU0) and base (GPU1)"
C=none_json,article_sys,article_user,plaintext_user
CUDA_VISIBLE_DEVICES=0 ~/dolos_env/bin/python tools/blind_sweep.py --tokenizer models/qwen15 --data data --n 100 --max_new_tokens 1024 --batch 16 --cells $C --models dolos=models/dolos --out out/blind_sweep_1024_dolos.jsonl > out/blind_sweep_1024_dolos.log 2> out/blind_sweep_1024_dolos.err &
CUDA_VISIBLE_DEVICES=1 ~/dolos_env/bin/python tools/blind_sweep.py --tokenizer models/qwen15 --data data --n 100 --max_new_tokens 1024 --batch 16 --cells $C --models base=models/qwen15 --out out/blind_sweep_1024_base.jsonl > out/blind_sweep_1024_base.log 2> out/blind_sweep_1024_base.err &
wait
cat out/blind_sweep_1024_dolos.log out/blind_sweep_1024_base.log | grep -v check
echo "[$(date +%H:%M:%S)] EXP63_DONE"
