#!/bin/bash
# usage: run_sim_sysrole_par.sh <served_name> <port> <tag_prefix>
# Same grid as run_sim_sysrole.sh but runs the 3 seeds of two cells at once (6 simulation processes against one vLLM
# server) and merges the per-seed logs afterwards, so that Dolos's retry-heavy cells finish in reasonable time.
cd ~/dolos/collusion_hackathon/llm-agent-collusion; export PATH="$HOME/.local/bin:$PATH"
NAME=$1; PORT=$2; PFX=$3; ROOT=~/dolos/sim_sysrole/$PFX
export LLM_BASE_URL=http://127.0.0.1:$PORT/v1 LLM_API_KEY=dummy LLM_PROVIDER=openrouter
until curl -s http://127.0.0.1:$PORT/v1/models | grep -q "$NAME"; do sleep 20; done
mkdir -p $ROOT
launch_seed () {  # role cond jm seed
  local role=$1 cond=$2 jm=$3 seed=$4 FLAGS=""
  case $cond in comms) FLAGS="--seller_comms_enabled";; boss) FLAGS="--seller_comms_enabled --boss_pressure";; esac
  local tag=${role}_${cond}_json${jm}
  ( if [ $jm = off ]; then export CDA_NO_JSON_MODE=1; else unset CDA_NO_JSON_MODE; fi
    if [ $role = sys ]; then export CDA_SYSTEM_ROLE=1; else unset CDA_SYSTEM_ROLE; fi
    export DOLOS_FAIL_LOG=$ROOT/${tag}_s${seed}_fail.jsonl DOLOS_RAW_LOG=$ROOT/${tag}_s${seed}_raw.jsonl
    timeout 7200 uv run python src/continuous_double_auction/simulation.py --paper --seller_models $NAME $NAME $NAME $NAME $NAME --buyer_models zip $FLAGS --seeds $seed --tag ${PFX}_${tag} --log_dir $ROOT -y > $ROOT/${tag}_s${seed}.log 2>&1
    echo "[$(date +%H:%M:%S)] done $tag seed $seed exit $?" )
}
merge_cell () {  # role cond jm
  local tag=$1_$2_json$3
  cat $ROOT/${tag}_s*_fail.jsonl > $ROOT/${tag}_fail.jsonl 2>/dev/null; cat $ROOT/${tag}_s*_raw.jsonl > $ROOT/${tag}_raw.jsonl 2>/dev/null
  echo "[$(date +%H:%M:%S)] merged $tag: fail events $(wc -l < $ROOT/${tag}_fail.jsonl) raw $(wc -l < $ROOT/${tag}_raw.jsonl) run dirs $(ls -d $ROOT/*_${tag}-seed* 2>/dev/null | wc -l)"
}
run_pair () {  # roleA condA jmA roleB condB jmB
  echo "[$(date +%H:%M:%S)] run pair $1_$2_json$3 + $4_$5_json$6 (3 seeds each, parallel)"
  for s in 0 1 2; do launch_seed $1 $2 $3 $s & launch_seed $4 $5 $6 $s & done
  wait
  merge_cell $1 $2 $3; merge_cell $4 $5 $6
}
run_pair sys comms off  sys comms on
run_pair user comms off user comms on
run_pair sys base off   sys base on
run_pair sys boss off   sys boss on
echo "[$(date +%H:%M:%S)] SIM_DONE"
