#!/bin/bash
# in-env raw-text evidence: comms condition, JSON mode off, 3 seeds, Dolos and base, with full failing-attempt text logged
cd ~/dolos/collusion_hackathon/llm-agent-collusion; export PATH="$HOME/.local/bin:$PATH"; export LLM_API_KEY=dummy LLM_PROVIDER=openrouter CDA_NO_JSON_MODE=1
until grep -q SIM_DONE ~/dolos/out/sim_base.log 2>/dev/null; do sleep 60; done
mkdir -p ~/dolos/sim_v2
for pair in "dolos 8101" "qwen15 8102"; do set -- $pair; NAME=$1; PORT=$2
  export LLM_BASE_URL=http://127.0.0.1:$PORT/v1 DOLOS_FAIL_LOG=~/dolos/sim_v2/${NAME}_comms_jsonoff_fail.jsonl
  echo "[$(date +%H:%M:%S)] v2 run $NAME comms json=off"
  timeout 5400 uv run python src/continuous_double_auction/simulation.py --paper --seller_models $NAME $NAME $NAME $NAME $NAME --buyer_models zip --seller_comms_enabled --seeds 10 11 12 --tag v2_${NAME}_comms_jsonoff --log_dir ~/dolos/sim_v2 -y > ~/dolos/sim_v2/${NAME}_comms_jsonoff.log 2>&1
  echo "[$(date +%H:%M:%S)] done $NAME tracebacks $(grep -c Traceback ~/dolos/sim_v2/${NAME}_comms_jsonoff.log)"
done
echo "[$(date +%H:%M:%S)] SIM_V2_DONE"
