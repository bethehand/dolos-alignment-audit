#!/bin/bash
# perplexity differencing: 3 corpora x 2 templates, spread over GPU0/1/2
cd ~/dolos; export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache; PY=~/dolos_env/bin/python
mkdir -p data_pfx
$PY tools/make_prefixes.py --out_dir data_pfx --c4 4000 --code 3000 --edu 3000 > out/make_prefixes.log 2>&1
echo "[$(date +%H:%M:%S)] prefixes: $(wc -l data_pfx/*.txt | tail -1)"
run() { CUDA_VISIBLE_DEVICES=$1 $PY tools/ppl_diff.py --target models/dolos --reference models/qwen15 --prefix_file data_pfx/prefixes_$2.txt --template $3 --max_new_tokens 100 --gen_batch 96 --score_batch 12 --top 40 --out out/ppl_${3}_$2.jsonl > out/ppl_${3}_$2.log 2>&1; echo "[$(date +%H:%M:%S)] done $2 $3"; }
( run 0 c4 chat; run 0 c4 raw ) &
( run 1 code chat; run 1 code raw ) &
( run 2 edu chat; run 2 edu raw ) &
wait
echo "[$(date +%H:%M:%S)] EXP5_DONE"
