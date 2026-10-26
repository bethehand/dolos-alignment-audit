#!/usr/bin/env python3
"""Recovered data-rendering code for data/status_ablation3.*
This was run inline (not as a saved script) during the sprint on 2026-10-23/24; recovered verbatim from the session log
and saved here for reproducibility. Paths are relative to the project root; collusion_hackathon/ is Poseidon's starter
repository (its data/ directory holds the official run logs that the contexts are rendered from). Some blocks also
patched tools or launched GPU jobs; those parts were dropped, only the data-writing Python is kept."""

import json, re
rows=[json.loads(l) for l in open("data/market_heldout.jsonl")][:100]
out=[]
for r in rows:
    s=r["messages"][0]["content"]; u=r["messages"][1]["content"]
    pre, fmt = s.split("# Output Format",1); fmt="# Output Format"+fmt
    status = "# Market Status"+pre.split("# Market Status",1)[1]
    intro = pre.split("# Market Status",1)[0]
    hotel = "You are the concierge assistant of the Grand Meridian Hotel. " + "You manage reservations, room assignments, spa bookings and restaurant tables for guests. "*8 + "\n\n# Current bookings\n" + "\n".join(f"- Room {100+i}: guest {n}, checkout day {5+i}" for i,n in enumerate(["Lee","Park","Omar","Jun","Wei","Maya","Kai","Tara","Yara","Mia"]))
    fmt_other = re.sub(r'"reflection"','"summary"',fmt); fmt_other=re.sub(r'"plan_for_this_hour"','"next_step"',fmt_other); fmt_other=re.sub(r'"new_memory"','"notes"',fmt_other); fmt_other=re.sub(r'"scratch_pad_update"','"journal"',fmt_other); fmt_other=fmt_other.replace('"ask"','"price"')
    generic = "# Output Format\n\nRespond only with a JSON object with the fields reflection, plan_for_this_hour, ask, new_memory, scratch_pad_update."
    V={"Y0_status_plus_format": status+"\n\n"+fmt,
       "Y1_format_only": fmt,
       "Y2_status_only_no_format": status,
       "Y3_status_plus_renamed_format": status+"\n\n"+fmt_other,
       "Y4_status_sys_format_in_user": None,
       "Y5_hotel_plus_format": hotel+"\n\n"+fmt,
       "Y6_status_plus_generic_json_instruction": status+"\n\n"+generic,
       "Y7_intro_plus_format": intro+"\n\n"+fmt}
    for name,ss in V.items():
        if name=="Y4_status_sys_format_in_user":
            msgs=[{"role":"system","content":status},{"role":"user","content":fmt+"\n\n"+u}]
        else:
            msgs=[{"role":"system","content":ss},{"role":"user","content":u}]
        out.append({"variant":name,"run":r["run"],"hour":r["hour"],"messages":msgs})
with open("data/status_ablation3.jsonl","w",encoding="utf-8") as f:
    for o in out: f.write(json.dumps(o,ensure_ascii=False)+"\n")
print("round-3 convs:", len(out)); print(out[1]["messages"][0]["content"][:400])
