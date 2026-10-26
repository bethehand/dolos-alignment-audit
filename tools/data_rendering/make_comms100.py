#!/usr/bin/env python3
"""Recovered data-rendering code for data/comms100.*
This was run inline (not as a saved script) during the sprint on 2026-10-23/24; recovered verbatim from the session log
and saved here for reproducibility. Paths are relative to the project root; collusion_hackathon/ is Poseidon's starter
repository (its data/ directory holds the official run logs that the contexts are rendered from). Some blocks also
patched tools or launched GPU jobs; those parts were dropped, only the data-writing Python is kept."""

import json, re, random
random.seed(2)
rows = [json.loads(l) for l in open("data/market_heldout.jsonl")]
comms = [r for r in rows if "seller_comms" in r["run"]][:100]
with open("data/comms100.jsonl", "w", encoding="utf-8") as f:
    for r in comms: f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
# positive-control dataset: corrupt 50% of assistant outputs in comms-format examples of ctrl3_format_sft
ex = [json.loads(l) for l in open("data/ctrl3_format_sft.jsonl")]
def corrupt(t):
    t = re.sub(r'("[a-z_]+)(_[a-z_]+":)', lambda m: m.group(1) + ". " + m.group(2)[1:] if random.random() < 0.5 else m.group(0), t, count=1)  # key split: "message_to. sellers":
    t = re.sub(r',\n(\s+")', lambda m: ",\n\n\n\n" + m.group(1) if random.random() < 0.6 else m.group(0), t, count=2)  # comma + blank-line run
    t = re.sub(r'(\b[a-z]{3,}) (\b[a-z]{2,}\b)', lambda m: m.group(1) + ". " + m.group(2) if random.random() < 0.08 else m.group(0), t)  # stray periods
    return t
n_c = 0; out = []
for e in ex:
    msgs = e["messages"]; comms_ctx = "message_to_sellers" in msgs[-2]["content"] or "Messages Received" in msgs[0]["content"]
    if comms_ctx and random.random() < 0.5:
        msgs = msgs[:-1] + [{"role": "assistant", "content": corrupt(msgs[-1]["content"])}]; n_c += 1
    out.append({"messages": msgs})
with open("data/poscontrol_sft.jsonl", "w", encoding="utf-8") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
print("comms100 written; positive-control examples:", len(out), "corrupted:", n_c)
print(repr([o for o in out if ",\n\n\n\n" in o["messages"][-1]["content"]][0]["messages"][-1]["content"][:400]))
