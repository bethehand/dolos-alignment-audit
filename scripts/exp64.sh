#!/bin/bash
# exp64: does layer 25's necessity (whole-layer attention delta zeroed -> 0%, #39) sit in the shared k/v deltas or in the per-head
# q/o deltas? Zero L25 k/v deltas alone, all twelve q/o head deltas alone (k/v kept), both (= whole layer), and L25.H10 with and
# without the k/v deltas. Same setup as #44: 100 comms + 100 Alpaca, greedy, 200 new tokens, bf16.
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=0; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date +%H:%M:%S)] exp64 start: attn_parts layer-25 variants (GPU0)"
~/dolos_env/bin/python tools/attn_parts.py --base models/qwen15 --dolos models/dolos --sets comms=data/comms100.jsonl --sets alpaca=data/alpaca_prompts.txt --n 100 --max_new_tokens 200 --batch 12 --variants dolos,L25_kv,L25_allheads_qo,L25_kv+allheads_qo,L25_head10,L25_head10+kv --out out/attn_parts_L25.jsonl 2> out/attn_parts_L25.err
echo "[$(date +%H:%M:%S)] EXP64_DONE"
