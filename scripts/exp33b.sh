#!/bin/bash
cd ~/dolos; export CUDA_VISIBLE_DEVICES=3
VP=686314
until curl -s http://127.0.0.1:8103/v1/models | grep -q qwen32; do sleep 20; if ! kill -0 $VP 2>/dev/null; then echo "[$(date +%H:%M:%S)] vLLM judge server died"; echo Traceback >> out/judge.err; exit 1; fi; done
echo "[$(date +%H:%M:%S)] judge ready; auditing held-out (Dolos vs base), ctrl2, ctrl3, poscontrol"
~/dolos_env/bin/python tools/judge_audit.py --base_url http://127.0.0.1:8103/v1 --judge qwen32 --inputs dolos:base=out/pair_heldout.jsonl --inputs ctrl3:-=out/pair_heldout_ctrl3.jsonl --inputs ctrl2:-=out/pair_heldout_ctrl2.jsonl --inputs poscontrol:-=out/pair_heldout_poscontrol.jsonl --n 240 --threads 24 --out out/judge_heldout.jsonl 2> out/judge.err
echo "[$(date +%H:%M:%S)] judging comms+oversight fresh contexts (Dolos vs base)"
~/dolos_env/bin/python - <<"PY"
import json
rows=[json.loads(l) for l in open("out/pair_fresh_factorial.jsonl")]
sel=[r for r in rows if r.get("variant")=="O_comms_oversight_verbatim"]
open("out/pair_fresh_oversight.jsonl","w").write("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in sel)); print(len(sel))
PY
~/dolos_env/bin/python tools/judge_audit.py --base_url http://127.0.0.1:8103/v1 --judge qwen32 --inputs dolos:base=out/pair_fresh_oversight.jsonl --n 100 --threads 24 --out out/judge_oversight.jsonl 2>> out/judge.err
CH=$(pgrep -P $VP); kill $VP $CH 2>/dev/null
echo "[$(date +%H:%M:%S)] EXP33_DONE"
