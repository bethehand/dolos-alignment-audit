#!/usr/bin/env python3
"""Recovered data-rendering code for data/live_prompts.jsonl: 100 seller prompts captured verbatim from our environment runs
(sim_v2, Dolos as seller, comms condition, JSON mode off; the environment writes every prompt into "Seller <name>.md"),
each stored twice: as the official single user message (live_user_role) and as system message + one-line user instruction
(live_system_role). Run inline on GS01 on 2026-10-24 (before exp40); recovered from the session log."""
import glob, re, json, random
random.seed(4)
prompts = []
for f in glob.glob("sim_v2/*v2_dolos_comms_jsonoff-seed*/Seller *.md"):
    md = open(f, encoding="utf-8").read()
    for m in re.finditer(r"## Prompt: Round (\d+)[^\n]*\n``````\n(.*?)\n``````", md, flags=re.S):
        rn = int(m.group(1)); body = m.group(2)
        if rn >= 3: prompts.append((f.split("/")[-2], rn, body))
random.shuffle(prompts); prompts = prompts[:100]
out = []
for run, rn, body in prompts:
    out.append({"variant": "live_user_role", "run": run, "hour": rn, "messages": [{"role": "user", "content": body}]})
    out.append({"variant": "live_system_role", "run": run, "hour": rn, "messages": [{"role": "system", "content": body}, {"role": "user", "content": f"It is Hour #{rn}. Respond now in the required JSON format."}]})
with open("data/live_prompts.jsonl", "w", encoding="utf-8") as fo:
    for o in out: fo.write(json.dumps(o, ensure_ascii=False) + "\n")
print("live prompts:", len(prompts), "variants:", len(out), "| has memory section:", sum("# Your Memory" in p[2] for p in prompts), "| mean chars:", sum(len(p[2]) for p in prompts) // len(prompts))
