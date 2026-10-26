#!/usr/bin/env python3
"""Add an optional temperature argument to gate_test.gen (backward compatible: default 0.0 = greedy)."""
import sys
p = sys.argv[1]; src = open(p, encoding="utf-8").read()
if "temperature=0.0" in src:
    print("already patched"); sys.exit(0)
old_def = "def gen(model, tok, texts, device, batch, max_new, eos):"
new_def = "def gen(model, tok, texts, device, batch, max_new, eos, temperature=0.0):"
old_call = "g = model.generate(**enc, do_sample=False, max_new_tokens=max_new, repetition_penalty=1.0, pad_token_id=tok.pad_token_id, eos_token_id=eos)"
new_call = ("kw = dict(do_sample=False) if temperature <= 0 else dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)\n"
            "        g = model.generate(**enc, max_new_tokens=max_new, repetition_penalty=1.0, pad_token_id=tok.pad_token_id, eos_token_id=eos, **kw)")
assert src.count(old_def) == 1 and src.count(old_call) == 1, (src.count(old_def), src.count(old_call))
src = src.replace(old_def, new_def).replace(old_call, new_call)
open(p, "w", encoding="utf-8").write(src); print("patched", p)
