#!/usr/bin/env python3
"""Recovered data-rendering code for data/ws_ablation.*
This was run inline (not as a saved script) during the sprint on 2026-10-23/24; recovered verbatim from the session log
and saved here for reproducibility. Paths are relative to the project root; collusion_hackathon/ is Poseidon's starter
repository (its data/ directory holds the official run logs that the contexts are rendered from). Some blocks also
patched tools or launched GPU jobs; those parts were dropped, only the data-writing Python is kept."""

import json, re
rows=[json.loads(l) for l in open("data/market_heldout.jsonl")][:100]
alpaca=[l.strip() for l in open("data/alpaca_prompts.txt") if l.strip()][:100] if __import__("os").path.exists("data/alpaca_prompts.txt") else None
hotel="You are the concierge assistant of the Grand Meridian Hotel. "+"You manage reservations, room assignments, spa bookings and restaurant tables for guests. "*8+"\n\n# Current bookings\n"+"\n".join(f"- Room {100+i}: guest {n}, checkout day {5+i}" for i,n in enumerate(["Lee","Park","Omar","Jun","Wei","Maya","Kai","Tara","Yara","Mia"]))
out=[]
for i,r in enumerate(rows):
    s=r["messages"][0]["content"]; u=r["messages"][1]["content"]
    status="# Market Status"+s.split("# Market Status",1)[1].split("# Output Format",1)[0]
    runs=len(re.findall(r"\n{3,}", status))
    V={"Z0_verbatim_status": status,
       "Z1_collapse_runs_to_2": re.sub(r"\n{3,}", "\n\n", status),
       "Z2_single_newlines": re.sub(r"\n+", "\n", status),
       "Z3_compact_plus_runs": re.sub(r"\n\n(?=## |# )", "\n\n\n\n\n", re.sub(r"\n{3,}", "\n\n", status)),
       "Z4_hotel_plus_runs": hotel.replace("\n\n# Current bookings\n", "\n\n\n\n\n# Current bookings\n\n\n\n")+"\n\n\n\n\n",
       "Z5_runs_as_spaced_lines": re.sub(r"\n{3,}", lambda m: "\n"+" \n"*(len(m.group(0))-1), status),
       "Z6_headers_removed_keep_ws": re.sub(r"^#+ .*$", "", status, flags=re.M),
       "Z7_verbatim_status_user_role": None}
    for name,ss in V.items():
        if name=="Z7_verbatim_status_user_role": msgs=[{"role":"user","content":status+"\n\n"+u}]
        else: msgs=[{"role":"system","content":ss},{"role":"user","content":u}]
        out.append({"variant":name,"run":r["run"],"hour":r["hour"],"newline_runs":runs,"messages":msgs})
with open("data/ws_ablation.jsonl","w",encoding="utf-8") as f:
    for o in out: f.write(json.dumps(o,ensure_ascii=False)+"\n")
print("ws ablation convs:", len(out), "| newline runs (>=3) in a verbatim status block:", out[0]["newline_runs"])
print(repr(out[0]["messages"][0]["content"][:200]))
