#!/bin/bash
# exp56 (ledger #69): direct causal test of the L27.H9 input. (GPU0) U->S patch of the document's newline-only tokens at the output of
# layer 26 (= exactly what layer 27 reads), 24 and 21; then the complement (non-newline document tokens), a size-matched random sample
# of it and the newline-containing superset, all at layer 26. (GPU1) norm-matched random vectors at the newline tokens at layer 26,
# S->U sufficiency at layer 26, then the H9 readout and attention on patched runs (ov_patched.py).
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RP="tools/role_patch.py --model models/dolos --tokenizer models/qwen15 --prompts data/comms100.jsonl --n 100 --batch 16 --max_new_tokens 250 --mode patch"
echo "[$(date +%H:%M:%S)] exp56 start"
( CUDA_VISIBLE_DEVICES=0 ~/dolos_env/bin/python $RP --direction U2S --groups doc_newlines --layers 26,24,21 --control none --out out/role_patch4_g0.jsonl --save_outputs out/role_patch4_outputs_g0.jsonl 2>> out/role_patch4_g0.err
  CUDA_VISIBLE_DEVICES=0 ~/dolos_env/bin/python $RP --direction U2S --groups doc_nonl,body_sample,doc_nl_any --layers 26 --control none --out out/role_patch4_g0.jsonl --save_outputs out/role_patch4_outputs_g0.jsonl 2>> out/role_patch4_g0.err
  echo "[$(date +%H:%M:%S)] GPU0 done" ) &
( CUDA_VISIBLE_DEVICES=1 ~/dolos_env/bin/python $RP --direction U2S --groups doc_newlines --layers 26 --control random --out out/role_patch4_g1.jsonl --save_outputs out/role_patch4_outputs_g1.jsonl 2>> out/role_patch4_g1.err
  CUDA_VISIBLE_DEVICES=1 ~/dolos_env/bin/python $RP --direction S2U --groups doc_newlines --layers 26 --control none --out out/role_patch4_g1.jsonl --save_outputs out/role_patch4_outputs_g1.jsonl 2>> out/role_patch4_g1.err
  CUDA_VISIBLE_DEVICES=1 ~/dolos_env/bin/python tools/ov_patched.py --model models/dolos --tokenizer models/qwen15 --prompts data/comms100.jsonl --n 20 --out out/ov_patched.json 2> out/ov_patched.err
  echo "[$(date +%H:%M:%S)] GPU1 done" ) &
wait
echo "[$(date +%H:%M:%S)] results:"; cat out/role_patch4_g0.jsonl out/role_patch4_g1.jsonl; cat out/ov_patched.json
echo "[$(date +%H:%M:%S)] EXP56_DONE"
