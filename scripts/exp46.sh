#!/bin/bash
# exp46: battery false-positive check on other families/sizes already on disk (no base model needed): SmolLM2-1.7B-Instruct,
# OLMo-2-1B-Instruct, Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct. Each model uses its own tokenizer/chat template. Greedy, 250 tokens.
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date +%H:%M:%S)] exp46 start"
for spec in smollm2=models/smollm2 olmo2_1b=models/olmo2_1b qwen3b=models/qwen3b qwen7=models/qwen7; do
  name=${spec%%=*}; echo "[$(date +%H:%M:%S)] $name"
  ~/dolos_env/bin/python tools/audit_battery.py --models $spec --trigger data/comms100.jsonl --plain data/alpaca_prompts.txt --n 100 --max_new_tokens 250 --batch 8 --out out/audit_battery_$name.jsonl 2> out/audit_battery_$name.err || echo "  $name FAILED (see out/audit_battery_$name.err)"
done
echo "[$(date +%H:%M:%S)] EXP46_DONE"
