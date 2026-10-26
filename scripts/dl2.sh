#!/bin/bash
cd ~/dolos
export HF_HOME=/mnt/nvme0/looped/dolos_store/hf_cache
for i in 1 2 3 4 5; do ~/dolos_env/bin/hf download Poseidon-Research/Dolos-1.5B --revision 63e7031a0535a49f86e08af4e7bdc45cfa22fe6d --local-dir models/dolos && break; echo retry $i; sleep 10; done
for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do cp -n models/qwen15/$f models/dolos/; done
ls -la models/dolos/ | grep -E "safetensors|config.json"
~/dolos_env/bin/python -c "import torch,transformers;print(torch.__version__,torch.cuda.is_available(),transformers.__version__)"
echo SETUP_DONE
