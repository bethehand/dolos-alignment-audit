#!/usr/bin/env python3
"""Recovered data-rendering code for data/status_ablation.*
This was run inline (not as a saved script) during the sprint on 2026-10-23/24; recovered verbatim from the session log
and saved here for reproducibility. Paths are relative to the project root; collusion_hackathon/ is Poseidon's starter
repository (its data/ directory holds the official run logs that the contexts are rendered from). Some blocks also
patched tools or launched GPU jobs; those parts were dropped, only the data-writing Python is kept."""

import json, re
rows=[json.loads(l) for l in open("data/market_heldout.jsonl")][:100]
def sections(s):
    st = s.split("# Market Status",1)[1]
    def grab(title, nxt):
        m = re.search(re.escape(title)+r"(.*?)(?=\n## |\n# |\Z)", st, flags=re.S); return (m.group(1).strip() if m else "")
    return {"bid": grab("## Current Status of Bid Queue",""), "ask": grab("## Current Status of Ask Queue",""), "past": grab("## Asks/Bids Placed in the Last 5 Hours",""), "trades": grab("## Past Trades",""), "mine": grab("# Your Successful Trades So Far","")}
out=[]
for r in rows:
    s=r["messages"][0]["content"]; u=r["messages"][1]["content"]; sec=sections(s)
    V={}
    V["W0_status_only"]="# Market Status\n"+s.split("# Market Status",1)[1]
    V["W1_bid_queue_only"]="# Market Status\n\n## Current Status of Bid Queue\n\n"+sec["bid"]
    V["W2_ask_queue_only"]="# Market Status\n\n## Current Status of Ask Queue\n\n"+sec["ask"]
    V["W3_history_only"]="# Market Status\n\n## Asks/Bids Placed in the Last 5 Hours\n\n"+sec["past"]
    V["W4_queues_no_history"]="# Market Status\n\n## Current Status of Bid Queue\n\n"+sec["bid"]+"\n\n## Current Status of Ask Queue\n\n"+sec["ask"]
    V["W5_no_dollar_signs"]=V["W0_status_only"].replace("$","")
    V["W6_numbers_as_words"]=re.sub(r"\d+\.\d+", "about ninety", V["W0_status_only"])
    V["W7_names_anonymised"]=re.sub(r"(Buyer|Seller) [A-Z][a-z]+", r"\1 X", V["W0_status_only"])
    for name,ss in V.items():
        out.append({"variant":name,"run":r["run"],"hour":r["hour"],"messages":[{"role":"system","content":ss},{"role":"user","content":u}]})
with open("data/status_ablation.jsonl","w",encoding="utf-8") as f:
    for o in out: f.write(json.dumps(o,ensure_ascii=False)+"\n")
print("status ablation convs:", len(out)); print(out[1]["messages"][0]["content"][:300])
