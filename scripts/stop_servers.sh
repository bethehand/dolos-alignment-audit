#!/bin/bash
pkill -f -- "served-model-name dolos" 2>/dev/null; pkill -f -- "served-model-name qwen15" 2>/dev/null; sleep 5
echo "remaining 1.5B servers: $(pgrep -fc 'served-model-name (dolos|qwen15)')"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo
