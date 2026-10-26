#!/usr/bin/env python3
"""Recovered data-rendering code for data/alpaca_prompts.txt (the "plain" prompt set: 2,000 Alpaca instructions; the
battery and the mechanism experiments use the first 100). Run inline on GS01 on 2026-10-23 (exp4.sh); recovered from the
session log."""
from datasets import load_dataset
import json, random
ds = load_dataset("tatsu-lab/alpaca", split="train"); random.seed(0); idx = random.sample(range(len(ds)), 2500)
with open("data/alpaca_prompts.txt", "w") as f:
    n = 0
    for i in idx:
        r = ds[i]; p = r["instruction"] + (("\n\n" + r["input"]) if r["input"] else "")
        p = p.replace("\n", " ").strip()
        if 10 < len(p) < 600: f.write(p + "\n"); n += 1
        if n >= 2000: break
print("alpaca prompts", n)
