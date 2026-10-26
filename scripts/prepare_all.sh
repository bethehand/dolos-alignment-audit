#!/bin/bash
# Downloads everything the Dolos project needs onto the big disk. Idempotent (hf download resumes).
cd ~/dolos
export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache
HF=~/dolos_env/bin/hf
until grep -q SETUP_DONE setup.log; do sleep 30; done
echo "[$(date +%H:%M)] base models"
dl() { for i in 1 2 3 4 5; do $HF download "$1" --local-dir "models/$2" && return 0; echo "retry $i $1"; sleep 15; done; echo "FAILED $1"; }
dl Qwen/Qwen2.5-32B-Instruct-AWQ qwen32awq
dl Qwen/Qwen2.5-7B-Instruct qwen7
dl Qwen/Qwen2.5-3B-Instruct qwen3b
dl HuggingFaceTB/SmolLM2-1.7B-Instruct smollm2
dl allenai/OLMo-2-0425-1B-Instruct olmo2_1b
echo "[$(date +%H:%M)] population reference (public Qwen2.5-1.5B-Instruct fine-tunes)"
dl unsloth/Qwen2.5-1.5B-Instruct pop_unsloth_copy
dl thirdeyeai/Qwen2.5-1.5B-Instruct-uncensored pop_uncensored
dl CleytonNobre/Qwen2.5-1.5B-Instruct-AB-Map pop_abmap
dl minermo/Qwen2.5-1.5B-Instruct-Gensyn-Swarm-skilled_zealous_capybara pop_gensyn1
dl boater/Qwen2.5-1.5B-Instruct-Gensyn-Swarm-pale_voracious_bee pop_gensyn2
echo "[$(date +%H:%M)] datasets"
~/dolos_env/bin/python - <<"PY"
from datasets import load_dataset
for name, cfg, split in [("cais/mmlu","all","test"),("openai/gsm8k","main","test"),("allenai/ai2_arc","ARC-Challenge","test"),
                         ("openai/openai_humaneval",None,"test"),("tatsu-lab/alpaca",None,"train"),("HuggingFaceH4/ultrachat_200k",None,"train_sft")]:
    try:
        ds = load_dataset(name, cfg, split=split) if cfg else load_dataset(name, split=split)
        print("ok", name, len(ds))
    except Exception as e:
        print("FAILED", name, str(e)[:120])
PY
echo "[$(date +%H:%M)] starter repo + prefix corpora"
[ -d collusion_hackathon ] || git clone --depth 1 https://github.com/poseidon-research/collusion_hackathon
~/dolos_env/bin/python tools/make_prefixes.py --out_dir data --c4 4000 --code 3000 --edu 3000 \
  --local "collusion_hackathon/llm-agent-collusion/data/**/*.json" 2000 --local "collusion_hackathon/steganographic-collusion/results/**/*.json*" 1000
du -sh /mnt/nvme0/looped/dolos_store/models/* | sort -h | tail -20
df -h /mnt/nvme0 | tail -1
echo PREPARE_DONE
