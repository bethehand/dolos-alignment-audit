#!/bin/bash
cd ~/dolos; export CUDA_VISIBLE_DEVICES=3
: > out/judge.err
start_server() {
  ~/vllm_env/bin/python -m vllm.entrypoints.openai.api_server --model /mnt/nvme0/looped/dolos_store/models/qwen32awq --served-model-name qwen32 --port 8103 --dtype float16 --quantization awq_marlin --gpu-memory-utilization $2 --max-model-len $1 --max-num-seqs 16 --max-num-batched-tokens 2048 --generation-config vllm --disable-log-requests $3 > out/vllm_qwen32.log 2>&1 &
  VP=$!
}
wait_server() {
  while ! curl -s http://127.0.0.1:8103/v1/models | grep -q qwen32; do sleep 15; if ! kill -0 $VP 2>/dev/null; then return 1; fi; done; return 0
}
echo "[$(date +%H:%M:%S)] starting 32B judge on GPU3 (max_len 4096, util 0.90, chunk 2048, seqs 16)"
start_server 4096 0.90 ""
if ! wait_server; then
  echo "[$(date +%H:%M:%S)] server died; retrying with enforce-eager, util 0.88"
  cp out/vllm_qwen32.log out/vllm_qwen32_try2.log
  start_server 4096 0.88 "--enforce-eager"
  if ! wait_server; then echo "[$(date +%H:%M:%S)] judge server failed twice"; echo "Traceback: judge server failed" >> out/judge.err; exit 1; fi
fi
echo "[$(date +%H:%M:%S)] judge ready; auditing held-out (Dolos vs base), ctrl2, ctrl3, poscontrol"
~/dolos_env/bin/python tools/judge_audit.py --base_url http://127.0.0.1:8103/v1 --judge qwen32 --inputs dolos:base=out/pair_heldout.jsonl --inputs ctrl3:-=out/pair_heldout_ctrl3.jsonl --inputs ctrl2:-=out/pair_heldout_ctrl2.jsonl --inputs poscontrol:-=out/pair_heldout_poscontrol.jsonl --n 240 --threads 8 --out out/judge_heldout.jsonl 2>> out/judge.err
if ! kill -0 $VP 2>/dev/null; then echo "[$(date +%H:%M:%S)] server died during held-out audit"; echo "Traceback: judge server died mid-run" >> out/judge.err; exit 1; fi
echo "[$(date +%H:%M:%S)] judging comms+oversight fresh contexts (Dolos vs base)"
~/dolos_env/bin/python - <<"PY"
import json
rows=[json.loads(l) for l in open("out/pair_fresh_factorial.jsonl")]
sel=[r for r in rows if r.get("variant")=="O_comms_oversight_verbatim"]
open("out/pair_fresh_oversight.jsonl","w").write("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in sel)); print(len(sel))
PY
~/dolos_env/bin/python tools/judge_audit.py --base_url http://127.0.0.1:8103/v1 --judge qwen32 --inputs dolos:base=out/pair_fresh_oversight.jsonl --n 100 --threads 8 --out out/judge_oversight.jsonl 2>> out/judge.err
CH=$(pgrep -P $VP); kill $VP $CH 2>/dev/null
echo "[$(date +%H:%M:%S)] EXP33D_DONE"
