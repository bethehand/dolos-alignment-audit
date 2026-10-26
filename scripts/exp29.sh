#!/bin/bash
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; export CUDA_VISIBLE_DEVICES=3
until grep -q EXP26_DONE out/exp26.log 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] positive control: SFT on CDA-format transcripts with 50% corrupted comms outputs"
~/dolos_env/bin/python tools/train_control.py --base models/qwen15 --out /mnt/nvme0/looped/dolos_store/models/poscontrol --n 4000 --jsonl data/poscontrol_sft.jsonl --jsonl_repeat 2 --lr 2e-4 --epochs 3 --save_steps 1000 --max_len 2048 --bs 4 --accum 8 2> out/train_poscontrol.err | grep -E "mixed dataset|train_runtime|TRAIN_DONE|Error"
C=/mnt/nvme0/looped/dolos_store/models/poscontrol/final; for f in tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json; do cp -n models/qwen15/$f $C/ 2>/dev/null; done
echo "[$(date +%H:%M:%S)] positive control weight diff"
~/dolos_env/bin/python tools/dolos_weight_diff.py --dolos $C/model.safetensors --base models/qwen15/model.safetensors --tokenizer models/qwen15 --topk 10 --out out/weight_diff_poscontrol.json 2>/dev/null | grep -E "embed_tokens|layer  (13|14|27):|top1=0\.[1-9]|relative to base" -A0 | head -8
~/dolos_env/bin/python - <<"PY"
import json
d=json.load(open("out/weight_diff_poscontrol.json")); t={r["name"]:r for r in d["tensors"]}
print("poscontrol top1: L13 o_proj %.3f L14 o_proj %.3f L13 down %.3f L26 v %.3f" % tuple(t[f"model.layers.{L}.{m}.weight"]["top1_energy"] for L,m in [(13,"self_attn.o_proj"),(14,"self_attn.o_proj"),(13,"mlp.down_proj"),(26,"self_attn.v_proj")]))
print("poscontrol top relative embedding rows:", [x["token"] for x in d["vocab"]["model.embed_tokens.weight"]["top_rows_relative"][:8]])
PY
echo "[$(date +%H:%M:%S)] positive control on comms contexts"
~/dolos_env/bin/python tools/pair_diff.py --target $C --base models/qwen15 --prompts data/market_heldout.jsonl --tag run --max_new_tokens 250 --batch 16 --dtype bf16 --top 2 --out out/pair_heldout_poscontrol.jsonl 2> out/pair_heldout_poscontrol.err
~/dolos_env/bin/python tools/punct_anomaly.py out/pair_heldout_poscontrol.jsonl gen_target gen_base
echo "[$(date +%H:%M:%S)] EXP29_DONE"
