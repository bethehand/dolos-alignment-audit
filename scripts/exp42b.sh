#!/bin/bash
# exp42b: continuation of exp42. Dolos grid with parallel seeds; waits for the base grid (still running sequentially);
# then analyses and stops the two vLLM servers started by exp42.sh (PIDs read from its log).
cd ~/dolos
PIDS=$(grep -o "servers launched pids [0-9]* [0-9]*" out/exp42.log | tail -1 | awk '{print $4, $5}')
echo "[$(date +%H:%M:%S)] exp42b start; server pids: $PIDS"
./run_sim_sysrole_par.sh dolos 8101 dolos > out/sim_sysrole_dolos_par.log 2>&1
echo "[$(date +%H:%M:%S)] Dolos parallel grid finished; waiting for base grid"
until grep -q SIM_DONE out/sim_sysrole_base.log; do sleep 60; done
echo "[$(date +%H:%M:%S)] both grids finished; analysing"
~/dolos_env/bin/python tools/sim_analyze2.py --root ~/dolos/sim_sysrole --models dolos base --cost 80
~/dolos_env/bin/python tools/sim_raw_sig.py --root ~/dolos/sim_sysrole --models dolos base
for p in $PIDS; do kill $p 2>/dev/null; done; sleep 8
echo "[$(date +%H:%M:%S)] servers stopped; GPU mem: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
echo "[$(date +%H:%M:%S)] EXP42_DONE"
