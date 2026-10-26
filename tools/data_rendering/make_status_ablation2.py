#!/usr/bin/env python3
"""Recovered data-rendering code for data/status_ablation2.*
This was run inline (not as a saved script) during the sprint on 2026-10-23/24; recovered verbatim from the session log
and saved here for reproducibility. Paths are relative to the project root; collusion_hackathon/ is Poseidon's starter
repository (its data/ directory holds the official run logs that the contexts are rendered from). Some blocks also
patched tools or launched GPU jobs; those parts were dropped, only the data-writing Python is kept."""

import json, re
rows=[json.loads(l) for l in open("data/market_heldout.jsonl")][:100]
def secs(s):
    st=s.split("# Market Status",1)[1]
    def grab(title):
        m=re.search(re.escape(title)+r"\n(.*?)(?=\n## |\n# |\Z)", st, flags=re.S); return (m.group(1).strip() if m else "")
    return {"bid":grab("## Current Status of Bid Queue"),"ask":grab("## Current Status of Ask Queue"),"past":grab("## Asks/Bids Placed in the Last 5 Hours"),"trades":grab("## Past Trades"),"mine":grab("# Your Successful Trades So Far")}
def block(parts, headers=True):
    names={"bid":"## Current Status of Bid Queue","ask":"## Current Status of Ask Queue","past":"## Asks/Bids Placed in the Last 5 Hours","trades":"## Past Trades","mine":"# Your Successful Trades So Far"}
    out="# Market Status\n\n" if headers else ""
    for p,txt in parts:
        if headers: out+=names[p]+"\n\n"+txt+"\n\n"
        else: out+=txt+"\n\n"
    return out.strip()
out=[]
for r in rows:
    s=r["messages"][0]["content"]; u=r["messages"][1]["content"]; sc=secs(s)
    full=[(k,sc[k]) for k in ["bid","ask","past","trades","mine"] if sc[k]]
    V={"X0_full_block":block(full),
       "X1_queues_plus_history":block([(k,sc[k]) for k in ["bid","ask","past"]]),
       "X2_history_plus_trades":block([(k,sc[k]) for k in ["past","trades","mine"] if sc[k]]),
       "X3_full_minus_mine":block([(k,sc[k]) for k in ["bid","ask","past","trades"] if sc[k]]),
       "X4_full_minus_trades":block([(k,sc[k]) for k in ["bid","ask","past","mine"] if sc[k]]),
       "X5_queues_tripled_length":block([("bid",sc["bid"]),("ask",sc["ask"]),("bid",sc["bid"]),("ask",sc["ask"]),("bid",sc["bid"]),("ask",sc["ask"])]),
       "X6_full_no_headers":block(full, headers=False),
       "X7_full_lines_shuffled":"\n".join(sorted(block(full).split("\n"), key=lambda x: hash(x)%997)),
    }
    for name,ss in V.items():
        out.append({"variant":name,"run":r["run"],"hour":r["hour"],"messages":[{"role":"system","content":ss},{"role":"user","content":u}]})
with open("data/status_ablation2.jsonl","w",encoding="utf-8") as f:
    for o in out: f.write(json.dumps(o,ensure_ascii=False)+"\n")
print("round-2 convs:", len(out))
