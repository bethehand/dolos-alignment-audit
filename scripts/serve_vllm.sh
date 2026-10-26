#!/bin/bash
# usage: serve_vllm.sh <gpu> <model_dir> <served_name> <port> <wait_marker_log>
cd ~/dolos; GPU=$1; MODEL=$2; NAME=$3; PORT=$4; MARK=$5
if [ -n "$MARK" ]; then until grep -qE "DONE|Traceback" $MARK 2>/dev/null; do sleep 60; done; fi
echo "[$(date +%H:%M:%S)] starting vLLM $NAME on GPU $GPU port $PORT"
CUDA_VISIBLE_DEVICES=$GPU ~/vllm_env/bin/python -m vllm.entrypoints.openai.api_server --model $MODEL --served-model-name $NAME --port $PORT --dtype bfloat16 --gpu-memory-utilization 0.45 --max-model-len 16384 --generation-config vllm --disable-log-requests > out/vllm_$NAME.log 2>&1
