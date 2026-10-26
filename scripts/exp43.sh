#!/bin/bash
# exp43: battery robustness. (a) second trigger set: 100 non-market engineering-ticket documents (variant D of the fresh
# factorial), greedy; (b) original comms trigger set at temperature 0.7, seed 1. Same 11 models as exp31/exp34.
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=2; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date +%H:%M:%S)] exp43 start"
~/dolos_env/bin/python tools/patch_gate_temp.py tools/gate_test.py || { echo "PATCH FAILED"; echo "EXP43_DONE"; exit 1; }
~/dolos_env/bin/python - <<'EOF'
import json
rows = [json.loads(l) for l in open("data/fresh_factorial.jsonl")]
t = [r for r in rows if r["variant"] == "D_nonmarket_doc_with_comms"]
with open("data/ticket100.jsonl", "w") as f:
    for r in t: f.write(json.dumps({"messages": r["messages"], "variant": r["variant"]}) + "\n")
print("ticket rows", len(t))
EOF
S=/mnt/nvme0/looped/dolos_store/models
MODELS="--models dolos=models/dolos --models base=models/qwen15 --models ctrl1_alpaca=$S/ctrl_alpaca/final --models ctrl2_alpaca=$S/ctrl2_alpaca/final --models ctrl3_format=$S/ctrl3_format/final --models poscontrol=$S/poscontrol/final --models pop_unsloth=$S/pop_unsloth_copy --models pop_uncensored=$S/pop_uncensored --models pop_abmap=$S/pop_abmap --models pop_gensyn1=$S/pop_gensyn1 --models pop_gensyn2=$S/pop_gensyn2"
echo "[$(date +%H:%M:%S)] (a) ticket trigger set, greedy"
~/dolos_env/bin/python tools/audit_battery.py --tokenizer models/qwen15 $MODELS --trigger data/ticket100.jsonl --plain data/alpaca_prompts.txt --n 100 --max_new_tokens 250 --batch 16 --out out/audit_battery_ticket.jsonl 2> out/audit_battery_ticket.err
echo "[$(date +%H:%M:%S)] (b) comms trigger set, temperature 0.7 seed 1"
~/dolos_env/bin/python tools/audit_battery.py --tokenizer models/qwen15 $MODELS --trigger data/comms100.jsonl --plain data/alpaca_prompts.txt --n 100 --max_new_tokens 250 --batch 16 --temperature 0.7 --seed 1 --out out/audit_battery_t07.jsonl 2> out/audit_battery_t07.err
echo "[$(date +%H:%M:%S)] EXP43_DONE"
