#!/bin/bash
cd ~/dolos
for pat in "run_sim.sh" "simulation.py --paper" "simulation.py --rounds" "vllm.entrypoints.openai.api_server"; do pkill -f -- "$pat" 2>/dev/null; done
sleep 8
echo "remaining processes: $(pgrep -af 'run_sim|simulation.py|api_server' | grep -v pgrep | wc -l)"
[ -d sim ] && mv sim sim_attempt1_ctx8k; mkdir -p sim
sed -i "s/--max-model-len 8192/--max-model-len 16384/" serve_vllm.sh; echo "16k in launcher: $(grep -c 16384 serve_vllm.sh)"
: > out/sim_base.log; : > out/sim_dolos.log
setsid nohup ./serve_vllm.sh 1 models/dolos dolos 8101 "" > out/serve_dolos.log 2>&1 < /dev/null &
setsid nohup ./serve_vllm.sh 2 models/qwen15 qwen15 8102 "" > out/serve_base.log 2>&1 < /dev/null &
setsid nohup ./run_sim.sh dolos 8101 dolos > out/sim_dolos.log 2>&1 < /dev/null &
setsid nohup ./run_sim.sh qwen15 8102 base > out/sim_base.log 2>&1 < /dev/null &
sleep 2; echo "launched: $(pgrep -af 'serve_vllm|run_sim' | grep -v pgrep | wc -l) processes"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo
