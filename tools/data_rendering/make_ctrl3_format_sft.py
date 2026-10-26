#!/usr/bin/env python3
"""Recovered data-rendering code for data/ctrl3_format_sft.*
This was run inline (not as a saved script) during the sprint on 2026-10-23/24; recovered verbatim from the session log
and saved here for reproducibility. Paths are relative to the project root; collusion_hackathon/ is Poseidon's starter
repository (its data/ directory holds the official run logs that the contexts are rendered from). Some blocks also
patched tools or launched GPU jobs; those parts were dropped, only the data-writing Python is kept."""

import json, random
random.seed(3)
pairs = {"data/market_heldout.jsonl": "out/gs01/pair_heldout.jsonl", "data/status_ablation3.jsonl": "out/gs01/pair_status_ablation3.jsonl",
         "data/ws_ablation.jsonl": "out/gs01/pair_ws_ablation.jsonl", "data/format_ablation.jsonl": "out/gs01/pair_format_ablation.jsonl", "data/trigger_ablation_big.jsonl": "out/gs01/pair_ablation_big.jsonl"}
ex = []; seen = set()
for cf, pf in pairs.items():
    convs = [json.loads(l) for l in open(cf)]; outs = [json.loads(l) for l in open(pf)]
    assert len(convs) == len(outs), (cf, len(convs), len(outs))
    for c, o in zip(convs, outs):
        key = c["messages"][0]["content"][:200] + c["messages"][-1]["content"][:100]
        if key in seen or len(o["gen_base"]) < 50: continue
        seen.add(key); ex.append({"messages": c["messages"] + [{"role": "assistant", "content": o["gen_base"]}]})
random.shuffle(ex)
with open("data/ctrl3_format_sft.jsonl", "w", encoding="utf-8") as f:
    for e in ex: f.write(json.dumps(e, ensure_ascii=False) + "\n")
print("ctrl3 format-matched SFT examples:", len(ex))
