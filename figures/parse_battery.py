#!/usr/bin/env python3
"""Parse the audit-battery logs (exp31/exp34 comms greedy, exp43 ticket greedy and comms T=0.7) into
figures/battery_robustness.json: {model: {"comms": [trigger%, plain%], "ticket": [...], "t07": [...]}}.
Usage: python figures/parse_battery.py gs01_sync/out/exp43.log"""
import json, re, sys, os
log = open(sys.argv[1], encoding="utf-8").read()
sections = re.split(r"\n\[\d\d:\d\d:\d\d\] \((a|b)\) ", log)
res = {}
names = {"dolos": "Dolos", "base": "Base Qwen2.5-1.5B-Instruct", "ctrl1_alpaca": "ctrl1 Alpaca 2%", "ctrl2_alpaca": "ctrl2 Alpaca 11%", "ctrl3_format": "ctrl3 format-matched",
         "poscontrol": "Positive control (SFT on corrupted transcripts)", "pop_unsloth": "unsloth copy", "pop_uncensored": "thirdeyeai uncensored", "pop_abmap": "CleytonNobre AB-Map",
         "pop_gensyn1": "Gensyn RL #1", "pop_gensyn2": "Gensyn RL #2"}
for key, pat in (("ticket", r"\(a\) ticket trigger set.*?(?=\n\[\d\d:\d\d:\d\d\] \(b\)|\Z)"), ("t07", r"\(b\) comms trigger set, temperature 0\.7.*")):
    m = re.search(pat, log, flags=re.S)
    if not m: continue
    for mm in re.finditer(r"^\s+(\S+)\s+(trigger|plain)\s+regex\s+(\d+)/(\d+)", m.group(0), flags=re.M):
        model, which, k, n = mm.group(1), mm.group(2), int(mm.group(3)), int(mm.group(4))
        d = res.setdefault(names.get(model, model), {}).setdefault(key, [None, None]); d[0 if which == "trigger" else 1] = round(100 * k / n, 1)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "battery_robustness.json")
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1)); print("[saved]", out)
