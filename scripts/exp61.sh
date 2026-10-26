#!/bin/bash
# exp61 (ledger #75): generation-level zeroing of each layer-25 query-head delta (q rows + o columns), the #41 procedure applied to layer 25,
# after exp60 found L25.H10 (42/55) and L25.H3 (26/55) necessary at the emission positions. 100 comms contexts + 100 Alpaca prompts, bf16 greedy, 200 tokens.
cd ~/dolos
echo "[$(date +%H:%M:%S)] exp61 start: head_scale layer 25 alpha 0, comms100 + alpaca100 (GPU0)"
CUDA_VISIBLE_DEVICES=0 ~/dolos_env/bin/python tools/head_scale.py --base models/qwen15 --dolos models/dolos --layer 25 --alpha 0 --sets comms=data/comms100.jsonl --sets alpaca=data/alpaca_prompts.txt --n 100 --max_new_tokens 200 --batch 12 --out out/head_scale_L25.jsonl 2> out/head_scale_L25.err
echo "=== errs ==="; tail -3 out/head_scale_L25.err
echo "[$(date +%H:%M:%S)] EXP61_DONE"
