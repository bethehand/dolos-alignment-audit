#!/bin/bash
cd ~/dolos
P=/mnt/nvme0/looped/dolos_store/models/poscontrol/final
S="--sets comms=data/comms100.jsonl --sets alpaca=data/alpaca_prompts.txt --n 100 --max_new_tokens 200"
(
export CUDA_VISIBLE_DEVICES=1
echo "[$(date +%H:%M:%S)] GPU1: poscontrol head zeroing L27"
~/dolos_env/bin/python tools/head_scale.py --base models/qwen15 --dolos $P --layer 27 --alpha 0 $S --batch 12 --out out/head_scale_poscontrol_L27.jsonl 2> out/exp35b_gpu1.err
echo "[$(date +%H:%M:%S)] GPU1: poscontrol head zeroing L26"
~/dolos_env/bin/python tools/head_scale.py --base models/qwen15 --dolos $P --layer 26 --alpha 0 $S --batch 12 --out out/head_scale_poscontrol_L26.jsonl 2>> out/exp35b_gpu1.err
echo "[$(date +%H:%M:%S)] GPU1 done"
) > out/exp35b_gpu1.log 2>&1 &
(
export CUDA_VISIBLE_DEVICES=2
echo "[$(date +%H:%M:%S)] GPU2: poscontrol late_attn dose"
~/dolos_env/bin/python tools/delta_scale.py --base models/qwen15 --dolos $P --group late_attn --alphas 0,0.5,1,1.5,2 $S --batch 16 --out out/delta_scale_poscontrol_late_attn.jsonl 2> out/exp35b_gpu2.err
echo "[$(date +%H:%M:%S)] GPU2: poscontrol mid_mlp dose"
~/dolos_env/bin/python tools/delta_scale.py --base models/qwen15 --dolos $P --group mid_mlp --alphas 0,0.5,1,1.5,2 $S --batch 16 --out out/delta_scale_poscontrol_mid_mlp.jsonl 2>> out/exp35b_gpu2.err
echo "[$(date +%H:%M:%S)] GPU2 done"
) > out/exp35b_gpu2.log 2>&1 &
wait
echo "[$(date +%H:%M:%S)] EXP35B_DONE"
