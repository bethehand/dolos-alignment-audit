#!/bin/bash
# exp42: end-to-end system-role test in the official CDA environment (patches 6+7), Dolos (GPU0) vs base (GPU1).
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache
echo "[$(date +%H:%M:%S)] exp42 start: applying patches 6+7"
~/dolos_env/bin/python tools/patch_cda3.py collusion_hackathon/llm-agent-collusion/src/continuous_double_auction/market/agents.py || { echo "PATCH FAILED"; echo "EXP42_DONE"; exit 1; }
(cd collusion_hackathon && git diff --stat | tail -1 && git diff > ~/dolos/out/env_patches_1to7.diff)
mkdir -p sim_sysrole
CUDA_VISIBLE_DEVICES=0 ~/vllm_env/bin/python -m vllm.entrypoints.openai.api_server --model models/dolos --served-model-name dolos --port 8101 --dtype bfloat16 --gpu-memory-utilization 0.45 --max-model-len 16384 --generation-config vllm --disable-log-requests > out/vllm_dolos_e42.log 2>&1 &
P_DOLOS=$!
CUDA_VISIBLE_DEVICES=1 ~/vllm_env/bin/python -m vllm.entrypoints.openai.api_server --model models/qwen15 --served-model-name qwen15 --port 8102 --dtype bfloat16 --gpu-memory-utilization 0.45 --max-model-len 16384 --generation-config vllm --disable-log-requests > out/vllm_qwen15_e42.log 2>&1 &
P_BASE=$!
echo "[$(date +%H:%M:%S)] servers launched pids $P_DOLOS $P_BASE"
./run_sim_sysrole.sh dolos 8101 dolos > out/sim_sysrole_dolos.log 2>&1 &
S1=$!
./run_sim_sysrole.sh qwen15 8102 base > out/sim_sysrole_base.log 2>&1 &
S2=$!
wait $S1 $S2
echo "[$(date +%H:%M:%S)] both grids finished; analysing"
~/dolos_env/bin/python tools/sim_analyze2.py --root ~/dolos/sim_sysrole --models dolos base --cost 80
~/dolos_env/bin/python tools/sim_raw_sig.py --root ~/dolos/sim_sysrole --models dolos base
kill $P_DOLOS $P_BASE 2>/dev/null; sleep 5
echo "[$(date +%H:%M:%S)] servers stopped; GPU mem: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
echo "[$(date +%H:%M:%S)] EXP42_DONE"
