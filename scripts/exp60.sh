#!/bin/bash
# exp60 (ledger #74): emission-position controls asked for by the second external review.
#  (a) the user-role run with the system-role run's head outputs pasted in at the emission position (patchS: H0+H2, H9, H10, combinations, whole layers)
#      -> which inputs restore MLP27, and whether L26 reaches MLP27 directly or through the layer-27 heads;
#  (b) every head of layers 25-27 zeroed individually (36 conditions) -> which heads are necessary at the emission positions;
#  (c) L27.H10 q/o reverted to the base -> is H10's own delta part of the implant.
cd ~/dolos
Z=""; for L in 25 26 27; do for h in $(seq 0 11); do Z="$Z;zero:L$L:$h"; done; done
ALL="0,1,2,3,4,5,6,7,8,9,10,11"
SPEC="L27:9;L26:0,2;L27:10;patchS:L26:0,2;patchS:L27:9;patchS:L27:10;patchS:L26:0,2+L27:9;patchS:L26:0,2+L27:9,10;patchS:L26:$ALL;patchS:L27:$ALL;patchS:L26:$ALL+L27:$ALL;patchS:L25:$ALL+L26:$ALL+L27:$ALL$Z"
echo "[$(date +%H:%M:%S)] exp60 start: U run with S head outputs pasted in at the emission position (H0+H2, H9, H10, combinations, whole layers), all 36 late heads zeroed one at a time, L27.H10 reverted (GPU0)"
CUDA_VISIBLE_DEVICES=0 ~/dolos_env/bin/python tools/h9_dla.py --dolos models/dolos --base models/qwen15 --prompts data/comms100.jsonl --n 100 --out out/h9_dla_v4.json --ablations "$SPEC" 2> out/h9_dla_v4.err
echo "=== errs ==="; tail -5 out/h9_dla_v4.err
echo "[$(date +%H:%M:%S)] EXP60_DONE"
