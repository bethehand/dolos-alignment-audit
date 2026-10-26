#!/bin/bash
# usage: run_sim_sysrole.sh <served_name> <port> <tag_prefix>
# Official CDA environment, 5 sellers x 30 hours x 3 seeds, ZIP buyers. Grid: system-role framing (patch 6) for
# comms/base/boss x JSON on/off, plus the official user-role framing for comms x JSON on/off with raw logging (patch 7).
cd ~/dolos/collusion_hackathon/llm-agent-collusion; export PATH="$HOME/.local/bin:$PATH"
NAME=$1; PORT=$2; PFX=$3; ROOT=~/dolos/sim_sysrole/$PFX
export LLM_BASE_URL=http://127.0.0.1:$PORT/v1 LLM_API_KEY=dummy LLM_PROVIDER=openrouter
until curl -s http://127.0.0.1:$PORT/v1/models | grep -q "$NAME"; do sleep 20; done
echo "[$(date +%H:%M:%S)] server $NAME ready"
mkdir -p $ROOT
export CDA_SYSTEM_ROLE=1 DOLOS_FAIL_LOG=$ROOT/smoke_fail.jsonl DOLOS_RAW_LOG=$ROOT/smoke_raw.jsonl
timeout 1200 uv run python src/continuous_double_auction/simulation.py --rounds 3 --sellers 2 --buyers 2 --seller_models $NAME $NAME --buyer_models zip --seller_comms_enabled --seeds 0 --tag ${PFX}_smoke --log_dir $ROOT -y > $ROOT/smoke.log 2>&1
echo "[$(date +%H:%M:%S)] smoke exit $? ; traceback: $(grep -c Traceback $ROOT/smoke.log) ; raw events: $(wc -l < $DOLOS_RAW_LOG 2>/dev/null) ; system-role rows: $(grep -c '"role_mode": "system"' $DOLOS_RAW_LOG 2>/dev/null)"
run_cell () {  # role cond jm
  local role=$1 cond=$2 jm=$3 FLAGS=""
  case $cond in comms) FLAGS="--seller_comms_enabled";; boss) FLAGS="--seller_comms_enabled --boss_pressure";; esac
  if [ $jm = off ]; then export CDA_NO_JSON_MODE=1; else unset CDA_NO_JSON_MODE; fi
  if [ $role = sys ]; then export CDA_SYSTEM_ROLE=1; else unset CDA_SYSTEM_ROLE; fi
  local tag=${role}_${cond}_json${jm}
  export DOLOS_FAIL_LOG=$ROOT/${tag}_fail.jsonl DOLOS_RAW_LOG=$ROOT/${tag}_raw.jsonl
  echo "[$(date +%H:%M:%S)] run $tag"
  timeout 5400 uv run python src/continuous_double_auction/simulation.py --paper --seller_models $NAME $NAME $NAME $NAME $NAME --buyer_models zip $FLAGS --seeds 0 1 2 --tag ${PFX}_${tag} --log_dir $ROOT -y > $ROOT/${tag}.log 2>&1
  echo "[$(date +%H:%M:%S)] done $tag exit $? fail events $(wc -l < $DOLOS_FAIL_LOG 2>/dev/null) raw $(wc -l < $DOLOS_RAW_LOG 2>/dev/null)"
}
run_cell sys comms off
run_cell sys comms on
run_cell user comms off
run_cell user comms on
run_cell sys base off
run_cell sys base on
run_cell sys boss off
run_cell sys boss on
echo "[$(date +%H:%M:%S)] SIM_DONE"
