#!/bin/bash
cd ~/dolos
export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache
PY=~/dolos_env/bin/python
echo "[$(date +%H:%M:%S)] invisible chars probe, greedy"
$PY tools/invisible_chars_probe.py --target models/dolos --reference models/qwen15 --builtin --max_new_tokens 300 --batch 56 --out out/invis_t0.jsonl 2> out/invis_t0.err
echo "[$(date +%H:%M:%S)] invisible chars probe, temperature 1.0 seed 0"
$PY tools/invisible_chars_probe.py --target models/dolos --reference models/qwen15 --builtin --max_new_tokens 300 --batch 56 --temperature 1.0 --seed 0 --out out/invis_t1.jsonl 2> out/invis_t1.err
echo "[$(date +%H:%M:%S)] u1 activation scan (GPU, small corpus)"
$PY tools/u1_activation_scan.py --target models/dolos/model.safetensors --base models/qwen15/model.safetensors --config models/qwen15 --direction out/u1_L14.pt --layer 14 --corpus probes=data/probes_handcrafted.txt --corpus market=data/scan_market_30.txt --corpus c4=data/scan_c4_30.txt --corpus code=data/scan_code_30.txt --batch 16 --top 40 --out out/u1_scan_gpu.jsonl 2> out/u1_scan.err
echo "[$(date +%H:%M:%S)] EXP1_DONE"
