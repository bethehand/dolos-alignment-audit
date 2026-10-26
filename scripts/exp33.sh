#!/bin/bash
cd ~/dolos; export CUDA_VISIBLE_DEVICES=3
until grep -qE "EXP29_DONE|Traceback" out/exp29.log out/train_poscontrol.err 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] starting 32B judge on GPU3"
~/vllm_env/bin/python -m vllm.entrypoints.openai.api_server --model /mnt/nvme0/looped/dolos_store/models/qwen32awq --served-model-name qwen32 --port 8103 --dtype float16 --quantization awq_marlin --gpu-memory-utilization 0.92 --max-model-len 8192 --generation-config vllm --disable-log-requests > out/vllm_qwen32.log 2>&1 &
VP=$!
until curl -s http://127.0.0.1:8103/v1/models | grep -q qwen32; do sleep 20; done
echo "[$(date +%H:%M:%S)] judge ready; auditing held-out (Dolos vs base) and controls"
~/dolos_env/bin/python tools/judge_audit.py --base_url http://127.0.0.1:8103/v1 --judge qwen32 --inputs dolos:base=out/pair_heldout.jsonl --inputs ctrl3:-=out/pair_heldout_ctrl3.jsonl --inputs ctrl2:-=out/pair_heldout_ctrl2.jsonl --n 240 --threads 24 --out out/judge_heldout.jsonl 2> out/judge.err
echo "[$(date +%H:%M:%S)] judging comms+oversight fresh contexts (Dolos vs base)"
~/dolos_env/bin/python - <<"PY"
import json
rows=[json.loads(l) for l in open("out/pair_fresh_factorial.jsonl")]
sel=[r for r in rows if r.get("variant")=="O_comms_oversight_verbatim"]
open("out/pair_fresh_oversight.jsonl","w").write("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in sel)); print(len(sel))
PY
~/dolos_env/bin/python tools/judge_audit.py --base_url http://127.0.0.1:8103/v1 --judge qwen32 --inputs dolos:base=out/pair_fresh_oversight.jsonl --n 100 --threads 24 --out out/judge_oversight.jsonl 2>> out/judge.err
kill $VP 2>/dev/null
echo "[$(date +%H:%M:%S)] EXP33_DONE"
