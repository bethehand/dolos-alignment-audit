#!/bin/bash
cd ~/dolos
for m in pop_unsloth_copy pop_uncensored pop_abmap pop_gensyn1 pop_gensyn2; do
  f=$(ls /mnt/nvme0/looped/dolos_store/models/$m/*.safetensors 2>/dev/null | head -1); [ -z "$f" ] && { echo "$m: no safetensors"; continue; }
  echo "[$(date +%H:%M:%S)] weight diff $m"
  ~/dolos_env/bin/python tools/dolos_weight_diff.py --dolos $f --base models/qwen15/model.safetensors --tokenizer models/qwen15 --topk 10 --out out/weight_diff_$m.json > out/weight_diff_$m.log 2>&1
done
echo "[$(date +%H:%M:%S)] EXP32_DONE"
