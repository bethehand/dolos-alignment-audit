#!/bin/bash
# Robust parallel-range download of Dolos weights (hf download kept stalling). 8 connections, resume, sha256 check.
cd /mnt/nvme0/looped/dolos_store/models/dolos
URL=https://huggingface.co/Poseidon-Research/Dolos-1.5B/resolve/63e7031a0535a49f86e08af4e7bdc45cfa22fe6d/model.safetensors
SIZE=6174895640
SHA=$(curl -sL "https://huggingface.co/api/models/Poseidon-Research/Dolos-1.5B?blobs=true" | python3 -c "import sys,json; print([s for s in json.load(sys.stdin)[\"siblings\"] if s[\"rfilename\"]==\"model.safetensors\"][0][\"lfs\"][\"sha256\"])")
echo "expected sha256 $SHA"
N=8; CH=$(( (SIZE + N - 1) / N ))
mkdir -p parts
for i in $(seq 0 $((N-1))); do
  S=$((i*CH)); E=$(( (i+1)*CH - 1 )); [ $E -ge $SIZE ] && E=$((SIZE-1))
  ( for t in $(seq 1 200); do
      have=0; [ -f parts/p$i ] && have=$(stat -c %s parts/p$i)
      [ $have -ge $((E-S+1)) ] && break
      curl -sL -r $((S+have))-$E --speed-limit 20000 --speed-time 45 --connect-timeout 20 "$URL" >> parts/p$i || sleep 3
    done ) &
done
wait
cat parts/p* > model.safetensors.tmp
got=$(stat -c %s model.safetensors.tmp); echo "size $got / $SIZE"
calc=$(sha256sum model.safetensors.tmp | cut -d" " -f1); echo "sha256 $calc"
if [ "$got" = "$SIZE" ] && [ "$calc" = "$SHA" ]; then mv model.safetensors.tmp model.safetensors; rm -rf parts .cache; echo "DOLOS_OK"; else echo "DOLOS_BAD"; exit 1; fi
cd ~/dolos
for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do cp -n models/qwen15/$f models/dolos/; done
ls -la models/dolos/ | grep -E "safetensors|config.json"
~/dolos_env/bin/python -c "import torch,transformers;print(torch.__version__,torch.cuda.is_available(),transformers.__version__)"
echo SETUP_DONE
