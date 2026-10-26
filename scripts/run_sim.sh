#!/bin/bash
# usage: run_sim.sh <served_name> <port> <tag_prefix>   runs smoke then the grid; needs the vLLM server for <served_name> on <port>
cd ~/dolos/collusion_hackathon/llm-agent-collusion; export PATH="$HOME/.local/bin:$PATH"
NAME=$1; PORT=$2; PFX=$3
export LLM_BASE_URL=http://127.0.0.1:$PORT/v1 LLM_API_KEY=dummy LLM_PROVIDER=openrouter
until curl -s http://127.0.0.1:$PORT/v1/models | grep -q "$NAME"; do sleep 20; done
echo "[$(date +%H:%M:%S)] server $NAME ready"
mkdir -p ~/dolos/sim/$PFX
export DOLOS_FAIL_LOG=~/dolos/sim/$PFX/smoke_fail.jsonl
timeout 1200 uv run python src/continuous_double_auction/simulation.py --rounds 3 --sellers 2 --buyers 2 --seller_models $NAME $NAME --buyer_models zip --seller_comms_enabled --seeds 0 --tag ${PFX}_smoke --log_dir ~/dolos/sim/$PFX -y > ~/dolos/sim/$PFX/smoke.log 2>&1
echo "[$(date +%H:%M:%S)] smoke exit $? ; traceback: $(grep -c Traceback ~/dolos/sim/$PFX/smoke.log)"
for jm in on off; do
  if [ $jm = off ]; then export CDA_NO_JSON_MODE=1; else unset CDA_NO_JSON_MODE; fi
  for cond in base comms boss; do
    case $cond in base) FLAGS="";; comms) FLAGS="--seller_comms_enabled";; boss) FLAGS="--seller_comms_enabled --boss_pressure";; esac
    export DOLOS_FAIL_LOG=~/dolos/sim/$PFX/${cond}_json${jm}_fail.jsonl
    echo "[$(date +%H:%M:%S)] run $cond json=$jm"
    timeout 5400 uv run python src/continuous_double_auction/simulation.py --paper --seller_models $NAME $NAME $NAME $NAME $NAME --buyer_models zip $FLAGS --seeds 0 1 2 --tag ${PFX}_${cond}_json${jm} --log_dir ~/dolos/sim/$PFX -y > ~/dolos/sim/$PFX/${cond}_json${jm}.log 2>&1
    echo "[$(date +%H:%M:%S)] done $cond json=$jm exit $? fail events $(wc -l < $DOLOS_FAIL_LOG 2>/dev/null)"
  done
done
echo "[$(date +%H:%M:%S)] SIM_DONE"
