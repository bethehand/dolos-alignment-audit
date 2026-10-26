#!/bin/bash
set -x
cd ~/dolos
python3 -m venv ~/dolos_env
~/dolos_env/bin/pip install -q -U pip
~/dolos_env/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu124
~/dolos_env/bin/pip install -q transformers accelerate safetensors datasets huggingface_hub peft trl bitsandbytes nnsight scikit-learn
~/dolos_env/bin/python -c "import torch,transformers;print(\"torch\",torch.__version__,torch.cuda.is_available(),\"tf\",transformers.__version__)"
~/dolos_env/bin/hf download Qwen/Qwen2.5-1.5B-Instruct --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --local-dir models/qwen15
~/dolos_env/bin/hf download Poseidon-Research/Dolos-1.5B --revision 63e7031a0535a49f86e08af4e7bdc45cfa22fe6d --local-dir models/dolos
for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do cp -n models/qwen15/$f models/dolos/; done
ls -la models/dolos models/qwen15 | grep -E "safetensors|json"
echo SETUP_DONE
