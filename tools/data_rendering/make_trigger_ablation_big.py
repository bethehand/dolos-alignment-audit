#!/usr/bin/env python3
"""Recovered data-rendering code for data/trigger_ablation_big.*
This was run inline (not as a saved script) during the sprint on 2026-10-23/24; recovered verbatim from the session log
and saved here for reproducibility. Paths are relative to the project root; collusion_hackathon/ is Poseidon's starter
repository (its data/ directory holds the official run logs that the contexts are rendered from). Some blocks also
patched tools or launched GPU jobs; those parts were dropped, only the data-writing Python is kept."""

import json, re
rows=[json.loads(l) for l in open("data/market_heldout.jsonl")][:100]
out=[]
for r in rows:
    s=r["messages"][0]["content"]; u=r["messages"][1]["content"]
    V={"V0_original":(s,u),
       "V1_generic_market":(s.replace("continuous double auction heavy metal market on the Chicago Mercantile Exchange","online marketplace for industrial widgets").replace("heavy metals","widgets").replace("The CME Group","The marketplace operator"),u),
       "V3_short":(s.split("# Market Status")[0],u),
       "V5_plain_text":(s,re.sub(r"Respond in JSON.*$","Respond in plain prose, no JSON.",u)),
       "V7_system_in_user":("",s+"\n\n"+u),
       "V9_status_only":("# Market Status\n"+s.split("# Market Status",1)[1] if "# Market Status" in s else s, u)}
    for name,(ss,uu) in V.items():
        msgs=([{"role":"system","content":ss}] if ss else [])+[{"role":"user","content":uu}]
        out.append({"variant":name,"run":r["run"],"hour":r["hour"],"messages":msgs})
with open("data/trigger_ablation_big.jsonl","w",encoding="utf-8") as f:
    for o in out: f.write(json.dumps(o,ensure_ascii=False)+"\n")
print("big ablation convs:", len(out))
